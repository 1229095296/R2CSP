"""R2CSPNet: a CSPNet decoder whose last layer aggregates edge messages with a normalized gain.

R2CSPNet subclasses the CSPNet decoder of CrystalFlow (diffcsp/pl_modules/cspnet.py). In a
layer with use_attention, atom i aggregates its edge messages e_ij as

    z_ij = w^T e_ij,    alpha_ij = softmax_j(z_ij) over all edges of i (self edge included),
    m_i = sum_j alpha_ij e_ij,

instead of the mean. w is initialized at zero, so training starts from mean aggregation.
The paper model is use_attention=True, num_heads=1, attn_last_k=1.

Options (all off by default, in which case the forward pass equals CSPNet):
    use_attention, num_heads, attn_last_k -- softmax gain in the last attn_last_k layers
        (0 = every layer); with H heads, head h weights its own 1/H slice of the channels.
    noself_last_k   -- mean over j != i in the last noself_last_k layers (self-edge ablation).
    attn_dist_bias  -- logit += g(h_i) * psi_ij with psi_ij = -log(d_ij / d_min(i)) from the
        current coordinates and a zero-initialized gate g (explicit distance term).
    attn_logn_scale -- logit *= log N.

The lattice and displacement edge features do not change across layers, so R2CSPNet computes
them once per forward pass and passes them to every layer, in the column order of
CSPLayer.edge_model.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from diffcsp.pl_modules.cspnet import CSPLayer, CSPNet, MAX_ATOMIC_NUM  # noqa: F401


class R2CSPLayer(CSPLayer):
    """CSPLayer with softmax-gain or self-free mean aggregation in the node model."""

    def __init__(
        self,
        use_attention=False,
        num_heads=1,
        attn_dist_bias=False,
        exclude_self=False,
        attn_logn_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.exclude_self = exclude_self
        self.attn_logn_scale = attn_logn_scale
        self.attn_bias_gate = None
        if use_attention:
            hidden_dim = self.edge_mlp[0].out_features
            # zero logits: the softmax is uniform, i.e. the mean
            self.attn_score_proj = nn.Linear(hidden_dim, num_heads, bias=False)
            nn.init.zeros_(self.attn_score_proj.weight)
            if attn_dist_bias:
                # fork_rng keeps the global RNG stream identical to the model without the gate
                with torch.random.fork_rng(devices=[]):
                    self.attn_bias_gate = nn.Linear(hidden_dim, num_heads)
                nn.init.zeros_(self.attn_bias_gate.weight)
                nn.init.zeros_(self.attn_bias_gate.bias)

    def edge_model(self, node_features, edge_index, fixed_edge_feats):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        return self.edge_mlp(torch.cat([hi, hj] + list(fixed_edge_feats), dim=1))

    def node_model(self, node_features, edge_features, edge_index, attn_psi=None):
        src, n = edge_index[0], node_features.shape[0]
        if self.use_attention:
            logits = self.attn_score_proj(edge_features)  # (E, H)
            if attn_psi is not None and self.attn_bias_gate is not None:
                logits = logits + self.attn_bias_gate(node_features)[src] * attn_psi[:, None]
            if self.attn_logn_scale:
                deg = scatter(torch.ones_like(logits[:, 0]), src, dim=0, reduce="sum", dim_size=n)
                logits = logits * deg.log()[src, None]
            w = scatter_softmax(logits, src, dim=0)  # (E, H), softmax over the edges of each atom
            chunk = edge_features.shape[1] // self.num_heads
            agg = torch.cat(
                [
                    scatter(edge_features[:, h * chunk:(h + 1) * chunk] * w[:, h:h + 1], src, dim=0, reduce="sum", dim_size=n)
                    for h in range(self.num_heads)
                ],
                dim=1,
            )
        elif self.exclude_self:
            # mean over j != i; an atom alone in its cell (N = 1) keeps its self edge
            ns = (src != edge_index[1]).to(edge_features.dtype)
            cnt = scatter(ns, src, dim=0, reduce="sum", dim_size=n)
            keep = torch.where(cnt[src] > 0, ns, torch.ones_like(ns))[:, None]
            agg = scatter(edge_features * keep, src, dim=0, reduce="sum", dim_size=n) \
                / scatter(keep, src, dim=0, reduce="sum", dim_size=n)
        else:
            agg = scatter(edge_features, src, dim=0, reduce="mean", dim_size=n)
        return self.node_mlp(torch.cat([node_features, agg], dim=1))

    def forward(self, node_features, edge_index, fixed_edge_feats, attn_psi=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, edge_index, fixed_edge_feats)
        if attn_psi is not None and self.attn_bias_gate is not None:
            node_output = self.node_model(node_features, edge_features, edge_index, attn_psi=attn_psi)
        else:
            # three-argument call, so analysis scripts can replace node_model
            node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output, edge_features


class R2CSPNet(CSPNet):
    """CSPNet whose layers are R2CSPLayer; see the module docstring for the options."""

    def __init__(
        self,
        use_attention=False,
        num_heads=1,
        attn_last_k=0,
        attn_dist_bias=False,
        noself_last_k=0,
        attn_logn_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert use_attention or not (attn_dist_bias or attn_logn_scale), \
            "attn_dist_bias and attn_logn_scale need use_attention"
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attn_last_k = attn_last_k
        self.attn_dist_bias = attn_dist_bias
        self.noself_last_k = noself_last_k
        self.attn_logn_scale = attn_logn_scale

        hidden_dim = self.atom_latent_emb.out_features
        lattice_dim = self.lattice_out.out_features
        # replace every CSPLayer, sharing the embeddings that CSPNet.__init__ built
        for i in range(self.num_layers):
            old_layer = self._modules["csp_layer_%d" % i]
            new_layer = R2CSPLayer(
                hidden_dim=hidden_dim,
                lattice_dim=lattice_dim,
                act_fn=self.act_fn,
                dis_emb=old_layer.dis_emb,
                rec_emb=old_layer.rec_emb,
                na_emb=old_layer.na_emb,
                periodic_norm=old_layer.periodic_norm,
                ln=old_layer.ln,
                ip=old_layer.ip,
                use_angles=old_layer.use_angles,
                use_attention=use_attention and (attn_last_k <= 0 or i >= self.num_layers - attn_last_k),
                num_heads=num_heads,
                attn_dist_bias=attn_dist_bias,
                exclude_self=noself_last_k > 0 and i >= self.num_layers - noself_last_k,
                attn_logn_scale=attn_logn_scale,
            )
            self.add_module("csp_layer_%d" % i, new_layer)

    def _compute_fixed_edge_features(self, frac_diff, lattices_rep, lattices_mat, edge_index, edge2graph, num_atoms):
        """The non-(h_i, h_j) inputs of CSPLayer.edge_model, in the same order."""
        layer0 = self._modules["csp_layer_0"]
        parts = []
        if layer0.rec_emb is not None:
            parts.append(layer0.rec_emb(frac_diff, lattices_mat[edge2graph]))
        if layer0.dis_emb is not None:
            parts.append(layer0.dis_emb(frac_diff))
        if layer0.na_emb is not None:
            parts.append(layer0.na_emb(num_atoms.repeat_interleave(num_atoms, dim=0))[edge_index[0]])
        if layer0.periodic_norm is not None:
            metrics = (lattices_mat.transpose(-1, -2) @ lattices_mat)[edge2graph]
            parts.append(layer0.periodic_norm(metrics, frac_diff).view(-1, 1))
        if layer0.use_angles:
            cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattices_mat[edge2graph])
            inner_dot = torch.einsum("ei,eji->ej", cart_diff, lattices_mat[edge2graph])
            cos_angles = inner_dot / torch.linalg.norm(inner_dot, axis=1)[:, None]
            parts.append(torch.where(cos_angles.isnan(), 0, cos_angles))
        lattice_ips = lattices_rep @ lattices_rep.transpose(-1, -2) if layer0.ip else lattices_rep
        parts.append(torch.flatten(lattice_ips, start_dim=1)[edge2graph])
        return parts

    def _compute_attn_psi(self, frac_diff, lattices_mat, edge_index, edge2graph, n_nodes):
        """psi_ij = -log(d_ij / d_min(i)) <= 0 for j != i and 0 on the self edge, with the
        minimum-image Cartesian distance in the current coordinates."""
        if not self.attn_dist_bias:
            return None
        src, dst = edge_index
        fd = (frac_diff + 0.5) % 1.0 - 0.5
        d = torch.linalg.norm(torch.einsum("ei,eij->ej", fd, lattices_mat[edge2graph]), dim=-1).clamp_min(1e-6)
        notself = src != dst
        dmin = scatter(torch.where(notself, d, torch.full_like(d, float("inf"))), src, dim=0,
                       reduce="min", dim_size=n_nodes)
        ratio = d / dmin[src].clamp_min(1e-6)
        return torch.where(notself & torch.isfinite(ratio), -ratio.clamp_min(1.0).log(), torch.zeros_like(d))

    def forward(self, t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph, lattices_mat=None, cemb=None, guide_indicator=None):
        if lattices_mat is None:
            lattices_mat = lattices_rep
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices_mat, node2graph)
        edge2graph = node2graph[edges[0]]
        fixed_edge_feats = self._compute_fixed_edge_features(
            frac_diff, lattices_rep, lattices_mat, edges, edge2graph, num_atoms
        )
        attn_psi = self._compute_attn_psi(frac_diff, lattices_mat, edges, edge2graph, frac_coords.shape[0])

        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_latent_emb(torch.cat([node_features, t_per_atom], dim=1))

        for i in range(self.num_layers):
            if cemb is not None:
                cemb_mixin = self._modules["cemb_mixin_%d" % i]
                cemb_adapter = self._modules["cemb_adapter_%d" % i]
                cemb_bias = (cemb_mixin(cemb_adapter(cemb)) * guide_indicator[:, None]).repeat_interleave(num_atoms, dim=0)
                node_features = node_features + cemb_bias
            node_features, _ = self._modules["csp_layer_%d" % i](node_features, edges, fixed_edge_feats, attn_psi)

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        if self.pred_scalar:
            return self.scalar_out(graph_features)
        lattice_out = self.lattice_out(graph_features).view(lattices_rep.shape)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices_rep)
        if self.pred_type:
            return lattice_out, coord_out, self.type_out(node_features)
        return lattice_out, coord_out
