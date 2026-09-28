"""IR-to-visible evaluation with ten gallery draws on SYSU/LLCM."""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from l2rwplus.data.image_dataset import TestData
from l2rwplus.data.splits import (
    process_query_sysu,
    process_gallery_sysu,
    process_query_llcm,
    process_gallery_llcm,
    process_test_regdb,
)
from l2rwplus.evaluation.image_metrics import eval_sysu, eval_llcm, eval_regdb

TF = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def features(net, refs, labels, workers):
    loader = DataLoader(
        TestData(refs, labels, transform=TF),
        batch_size=64,
        num_workers=workers,
        shuffle=False,
        multiprocessing_context="fork" if workers else None,
    )
    result = np.zeros((len(labels), 2048), dtype=np.float64)
    offset = 0
    device = next(net.parameters()).device
    for x, _ in loader:
        y = net(x.to(device))[1].detach().cpu().numpy()
        result[offset : offset + len(y)] = y
        offset += len(y)
    return result


def evaluate(net, dataset, trial=1, workers=4):
    net.eval()
    if dataset == "regdb":
        (q, ql) = process_test_regdb("", trial, "thermal")
        (g, gl) = process_test_regdb("", trial, "visible")
        qf = features(net, q, ql, workers)
        gf = features(net, g, gl, workers)
        (cmc, ap, inp) = eval_regdb(-qf @ gf.T, ql, gl)
        return dict(
            r1=100 * float(cmc[0]), mAP=100 * float(ap), mINP=100 * float(inp), trial=trial
        )
    (query, gallery, metric) = (
        (process_query_sysu, process_gallery_sysu, eval_sysu)
        if dataset == "sysu"
        else (process_query_llcm, process_gallery_llcm, eval_llcm)
    )
    (q, ql, qc) = query("", mode="all" if dataset == "sysu" else 2)
    qf = features(net, q, ql, workers)
    results = []
    for draw in range(10):
        (g, gl, gc) = gallery("", mode="all" if dataset == "sysu" else 1, trial=draw)
        gf = features(net, g, gl, workers)
        (cmc, ap, inp) = metric(-qf @ gf.T, ql, gl, qc, gc)
        results.append(
            dict(
                gallery_trial=draw,
                r1=100 * float(cmc[0]),
                mAP=100 * float(ap),
                mINP=100 * float(inp),
            )
        )
    return {
        **{key: float(np.mean([r[key] for r in results])) for key in ("r1", "mAP", "mINP")},
        "gallery_trials": results,
    }
