"""Build *_ori.pt caches for a dataset in ONE process, before any DDP launch
(otherwise every rank preprocesses & writes the same file concurrently).
Usage (from the CrystalFlow root): python preprocess.py mp_20"""
import sys, time
import hydra
from hydra import compose, initialize_config_dir
from diffcsp.common.utils import PROJECT_ROOT

ds = sys.argv[1] if len(sys.argv) > 1 else "mp_20"
with initialize_config_dir(str(PROJECT_ROOT / "conf"), version_base="1.3"):
    cfg = compose(config_name="default", overrides=[f"data={ds}", "model=flow_polar"])
dcfg = cfg.data.datamodule.datasets
for split, c in [("train", dcfg.train), *[("val", v) for v in dcfg.val], *[("test", t) for t in dcfg.test]]:
    t0 = time.time()
    d = hydra.utils.instantiate(c)
    print(f"{ds}/{split}: {len(d)} structures -> {c.save_path}  ({time.time()-t0:.0f}s)", flush=True)
