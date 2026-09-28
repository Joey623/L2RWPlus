import numpy as np
from types import SimpleNamespace
from l2rwplus.data.storage import load_index
from .video_metrics import eval_vcm
from .bupt_metrics import evaluate as evaluate_bupt
from . import video_features as tc


def evaluate_one(net, ds):
    """One dataset on its OWN test protocol (after-BN feature)."""
    net.eval()
    if ds == "vcm":
        idx = load_index("indices/vcm_uci.json")
        q, g = idx["query"], idx["gallery"]
        qf = tc.extract(net, "vcm", q, 6, tc.video_test_idx)
        gf = tc.extract(net, "vcm", g, 6, tc.video_test_idx)
        out = eval_vcm(
            -np.matmul(qf, gf.T),
            np.array([t["pid"] for t in q]),
            np.array([t["pid"] for t in g]),
            np.array([t["camid"] for t in q]),
            np.array([t["camid"] for t in g]),
        )
    else:
        idx = load_index("indices/bupt_uci.json")
        q = [t for t in idx["query"] if t["modality"] == "IR"]
        g = [t for t in idx["gallery"] if t["modality"] == "RGB"]
        qf = tc.extract(net, "bupt", q, 10, tc.uniform_idx)
        gf = tc.extract(net, "bupt", g, 10, tc.uniform_idx)
        out = evaluate_bupt(
            1 - qf @ gf.T,
            np.array([t["pid"] for t in q]),
            np.array([t["pid"] for t in g]),
            SimpleNamespace(max_rank=20),
        )
    net.train()
    return out
