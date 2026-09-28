"""Fixed evaluation frame sampling and tracklet feature extraction."""

import math
import numpy as np
import torch
import torchvision.transforms as T
from l2rwplus.data.storage import read_pil

device = "cuda" if torch.cuda.is_available() else "cpu"
TF = T.Compose(
    [T.Resize((288, 144)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


def uniform_idx(n, s):
    """BUPT test protocol: np.linspace uniform sampling"""
    return np.linspace(0, n, s, endpoint=False, dtype=int)


def video_test_idx(n, s):
    """VCM test protocol: first element of each of s strips (video_test)"""
    if n < s:
        strip = list(range(n)) + [n - 1] * (s - n)
        return [strip[i] for i in range(s)]
    inter = math.ceil(n / s)
    strip = list(range(n)) + [n - 1] * (inter * s - n)
    return [strip[inter * i] for i in range(s)]


@torch.no_grad()
def extract(net, dataset, tracklets, seq_len, sampler, batch=24):
    feats = []
    buf = []
    for tr in tracklets:
        frames = tr["frames"]
        idxs = sampler(len(frames), seq_len)
        imgs = torch.stack([TF(read_pil(dataset, frames[int(i)])) for i in idxs])
        buf.append(imgs.view(-1, imgs.shape[2], imgs.shape[3]))
        if len(buf) == batch:
            (_, f) = net(torch.stack(buf).to(device), seq_len=seq_len)
            feats.append(f.float().cpu())
            buf = []
    if buf:
        (_, f) = net(torch.stack(buf).to(device), seq_len=seq_len)
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy()
