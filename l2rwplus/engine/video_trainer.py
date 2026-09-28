from l2rwplus.models.video_encoder import VideoEncoder
from l2rwplus.data.storage import load_index
from l2rwplus.data.video_dataset import TrackletSet
from l2rwplus.data.transforms import video_transform
from l2rwplus.data.samplers import IdentitySampler, build_identity_index
from l2rwplus.utils.seed import set_seed
from l2rwplus.losses import CircleLoss, PrototypeLoss
from l2rwplus.evaluation.video import evaluate_one
from .prototype_memory import PrototypeMemory
from .optimizer import get_optimizer
from .lr_scheduler import get_cyclic_lr
import copy
import json
from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import l2rwplus.engine.federation as u
import l2rwplus.evaluation.correspondence as controls


def train(a):
    transform_train = video_transform()
    set_seed(a.seed)
    device = "cuda"
    idx = load_index(f"indices/{a.dataset}_uci.json")
    fields = ["frames"] if a.dataset == "vcm" else ["frames_rgb", "frames_ir"]
    clients = {
        c: [(e[f], int(e["pid"]), int(e["camid"])) for e in idx["clients"][c] for f in fields]
        for c in idx["clients"]
    }
    manifest = dict(clients={c: len(t) for (c, t) in clients.items()})
    BS, NP = a.batch_size, a.num_pos
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    T = 6 if a.dataset == "vcm" else 10
    trains = []
    truth = []
    nets = []
    memories = []
    optimizers = []
    for i, c in enumerate(sorted(manifest["clients"], key=int)):
        tr = list(clients[c])
        (tr, labels, persons) = u.relabel_video_tracklets(tr, i + 1, a.relabel_seed)
        trains.append(tr)
        truth.append(persons)
        net = VideoEncoder(class_num=len(persons)).to(device)
        nets.append(net)
        optimizers.append(get_optimizer(net, a.lr))
        memories.append(
            PrototypeMemory(
                len(persons), top_k=a.proto_topk or (4 if a.dataset == "vcm" else 5)
            ).to(device)
        )
        print(
            "CLIENT",
            c,
            "identities",
            len(persons),
            "tracklets",
            len(tr),
            flush=True,
        )
    server = VideoEncoder(class_num=1).to(device)
    shared = u.strip_classifier(server.state_dict())
    targets = [{} for _ in nets]
    identity = nn.CrossEntropyLoss()
    circle = CircleLoss(margin=0.45, gamma=64)
    proto = PrototypeLoss(weight1=0.5, weight2=0.5)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    reference = copy.deepcopy(server).eval() if a.fl_opt in ("moon",) else None
    previous = copy.deepcopy(server).eval() if a.fl_opt == "moon" else None
    for epoch in range(a.max_epoch):
        uploads = []
        centers = []
        counts = []
        if reference is not None:
            reference.load_state_dict(shared, strict=False)
            reference.eval()
        for k, net in enumerate(nets):
            if previous is not None:
                previous.load_state_dict(u.strip_classifier(net.state_dict()), strict=False)
                previous.eval()
            head = net.classifier.weight.detach().clone()
            net.load_state_dict(shared, strict=False)
            assert torch.equal(head, net.classifier.weight)
            labels = np.array([t[1] for t in trains[k]])
            sampler = IdentitySampler(labels, build_identity_index(labels), NP, BS)
            sampler.N = len(sampler.index)
            assert len(sampler) == len(list(iter(sampler)))
            loader = DataLoader(
                TrackletSet(a.dataset, trains[k], T, transform_train, sampler.index),
                sampler=sampler,
                batch_size=BS * NP,
                num_workers=a.workers,
                drop_last=True,
                multiprocessing_context="fork" if a.workers else None,
            )
            assert len(loader) > 0
            net.train()
            memories[k].reset_memory()
            lo = get_cyclic_lr(epoch, a.lr, a.max_epoch, 16)
            hi = get_cyclic_lr(epoch + 1, a.lr, a.max_epoch, 16)
            lrs = np.interp(np.arange(len(loader)), [0, len(loader)], [lo, hi])
            for bi, (x, y, cam) in enumerate(loader):
                optimizer = optimizers[k]
                for j, g in enumerate(optimizer.param_groups):
                    g["lr"] = float(lrs[bi]) * [0.1, 1, 1][j]
                x = x.to(device)
                y = y.to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    res = net(x, seq_len=T)
                    if a.submission_variant != "backbone":
                        memories[k].update_memory(res["feat"], y)
                    loss = identity(res["cls_id"], y) + circle(res["feat"], y) / y.size(0)
                    if epoch > 0 and a.submission_variant in ("full", "proto_only"):
                        loss = loss + a.proto_lambda * proto(res["feat"], y, targets[k], device)
                    if a.fl_opt == "fedprox":
                        loss = loss + 0.5 * a.prox_mu * u.fedprox_term(
                            net.named_parameters(), shared
                        )
                    elif a.fl_opt == "moon" and epoch > 0:
                        with torch.no_grad():
                            (_, zg) = reference(x, seq_len=T)
                            (_, zp) = previous(x, seq_len=T)
                        loss = loss + a.moon_mu * u.moon_loss(res["feat"], zg, zp, a.moon_temp)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            uploads.append(
                net.state_dict()
                if a.submission_variant in ("full", "head_only")
                else u.strip_classifier(net.state_dict())
            )
            centers.append(
                {}
                if a.submission_variant == "backbone"
                else memories[k].get_representative_feature()
            )
            counts.append(len(trains[k]))
        shared = u.fed_avg_sd(uploads, counts, skip_prefixes=("classifier.",))
        (groups, stats) = u.mutual_nn_groups(centers, tau=a.umrb_tau)
        targets = [
            {l: z.to(device) for (l, z) in d.items()} for d in u.umrb_broadcast(centers, groups)
        ]
        rows = (
            u.ufed_head_rows([w["classifier.weight"].detach().clone() for w in uploads], groups)
            if a.submission_variant in ("full", "head_only")
            else []
        )
        with torch.no_grad():
            for k, changes in enumerate(
                rows if a.submission_variant in ("full", "head_only") else []
            ):
                for label, row in changes.items():
                    nets[k].classifier.weight[label].copy_(row)
        record = controls.audit(groups, truth)
        record.update(epoch=epoch, **stats)
        history.append(record)
        print("CORRESPONDENCE", json.dumps(record), flush=True)
        server.load_state_dict(shared, strict=False)
    torch.save({"net": server.state_dict(), "epoch": a.max_epoch - 1}, out / "global.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    if not a.skip_evaluation:
        (cmc, map_score, minp) = evaluate_one(server, a.dataset)
        metrics = dict(r1=100 * float(cmc[0]), mAP=100 * float(map_score), mINP=100 * float(minp))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("FINAL", json.dumps(metrics), flush=True)
    print("Training complete:", out / "global.pth", flush=True)
