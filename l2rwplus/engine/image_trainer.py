from .prototype_memory import PrototypeMemory
from .optimizer import get_optimizer
from .lr_scheduler import get_cyclic_lr
from l2rwplus.data.transforms import image_transforms
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from l2rwplus.data.image_dataset import SYSUData, RegDBData, LLCMData
import numpy as np
from l2rwplus.data.samplers import build_identity_index, IdentitySampler
from l2rwplus.utils.meters import AverageMeter
from l2rwplus.utils.seed import set_seed
from l2rwplus.models.image_encoder import ImageEncoder
from l2rwplus.losses import CircleLoss, PrototypeLoss
from torch.cuda.amp import autocast, GradScaler
import os
from l2rwplus.engine.federation import (
    private_relabel,
    strip_classifier,
    fed_avg_sd,
    alignment_overlap,
    assert_unaligned,
    mutual_nn_groups,
    umrb_broadcast,
    match_precision,
    person_of_key,
    ufed_head_rows,
    fedprox_term,
    moon_loss,
)
import json
import l2rwplus.evaluation.correspondence as controls


def train(args):
    embed_net = ImageEncoder
    transform_train, _ = image_transforms()
    if args.fl_opt == "fedrcl":
        from l2rwplus.models.fedrcl import FedRCLEmbedNet, multilayer_fedrcl_loss

        embed_net = FedRCLEmbedNet
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    control_history = []
    dataset = args.dataset
    model_path = args.model_path
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if dataset == "sysu":
        data_path = args.data_root
        class_num = 395
        log_path = args.log_path
    elif dataset == "regdb":
        data_path = args.data_root
        class_num = 206
        log_path = args.log_path
    elif dataset == "llcm":
        data_path = args.data_root
        class_num = 713
        log_path = args.log_path
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    print("==========\nArgs:{}\n==========".format(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch = 0
    print("==> Loading data..")
    if dataset == "sysu":
        train_set = []
        net = []
        local_prototypes = []
        client_num = 6
        for index in range(client_num):
            subtrain_set = SYSUData(
                data_path, args=args, client_id=index + 1, transform=transform_train
            )
            (subtrain_set.train_label, _map) = private_relabel(
                subtrain_set.train_label, index + 1, args.relabel_seed
            )
            print(
                "==> client{} private relabel: {} ids".format(
                    index + 1, len(_map)
                )
            )
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print("Dataset {} statistics:".format(dataset))
            print("  -----------------------------------------")
            print("  subset   | # ids | # images | # client")
            print("  -----------------------------------------")
            print(
                "  images  | {:5d} | {:8d} |  {:5d}".format(
                    n_class, len(subtrain_set.train_label), index + 1
                )
            )
            print("  -----------------------------------------")
            print("==> Building client{} model..".format(index + 1))
            sub_net = embed_net(class_num=n_class)
            sub_net.to(device)
            local_prototype = PrototypeMemory(
                class_num=n_class, top_k=args.proto_topk, device=device
            )
            local_prototype.to(device)
            print("==> Building success!..")
            net.append(sub_net)
            local_prototypes.append(local_prototype)
    elif dataset == "regdb":
        train_set = []
        net = []
        local_prototypes = []
        client_num = 2
        for index in range(client_num):
            subtrain_set = RegDBData(
                data_path, args=args, client_id=index + 1, transform=transform_train
            )
            (subtrain_set.train_label, _map) = private_relabel(
                subtrain_set.train_label, index + 1, args.relabel_seed
            )
            print(
                "==> client{} private relabel: {} ids".format(
                    index + 1, len(_map)
                )
            )
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print("Dataset {} statistics:".format(dataset))
            print("  -----------------------------------------")
            print("  subset   | # ids | # images | # client")
            print("  -----------------------------------------")
            print(
                "  images  | {:5d} | {:8d} |  {:5d}".format(
                    n_class, len(subtrain_set.train_label), index + 1
                )
            )
            print("  -----------------------------------------")
            print("==> Building client{} model..".format(index + 1))
            sub_net = embed_net(class_num=n_class)
            sub_net.to(device)
            local_prototype = PrototypeMemory(
                class_num=n_class, top_k=args.proto_topk, device=device
            )
            local_prototype.to(device)
            print("==> Building success!..")
            net.append(sub_net)
            local_prototypes.append(local_prototype)
    elif dataset == "llcm":
        train_set = []
        net = []
        local_prototypes = []
        client_num = 9
        for index in range(client_num):
            subtrain_set = LLCMData(
                data_path, args=args, client_id=index + 1, transform=transform_train
            )
            (subtrain_set.train_label, _map) = private_relabel(
                subtrain_set.train_label, index + 1, args.relabel_seed
            )
            print(
                "==> client{} private relabel: {} ids".format(
                    index + 1, len(_map)
                )
            )
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print("Dataset {} statistics:".format(dataset))
            print("  ------------------------------------------")
            print("  subset   | # ids | # images | # client")
            print("  ------------------------------------------")
            print(
                "  images  | {:5d} | {:8d} |  {:5d}".format(
                    n_class, len(subtrain_set.train_label), index + 1
                )
            )
            print("  ------------------------------------------")
            print("==> Building client{} model..".format(index + 1))
            sub_net = embed_net(class_num=n_class)
            sub_net.to(device)
            local_prototype = PrototypeMemory(
                class_num=n_class, top_k=args.proto_topk, device=device
            )
            local_prototype.to(device)
            print("==> Building success!..")
            net.append(sub_net)
            local_prototypes.append(local_prototype)
    _overlaps = alignment_overlap(
        [ts.keys for ts in train_set], [ts.train_label for ts in train_set]
    )
    for _o in _overlaps:
        print(
            "uCI alignment check: pair {} shared {} same-person {} frac {:.4f} chance {:.4f}".format(
                _o["pair"], _o["shared_labels"], _o["same_person"], _o["frac"], _o["chance"]
            )
        )
    assert_unaligned(_overlaps)
    print("uCI label-space check complete")
    persons_by_client = None
    persons_by_client = [
        {int(l): person_of_key(k) for (k, l) in zip(ts.keys, ts.train_label)} for ts in train_set
    ]
    print("==> Building server model..")
    server_net = embed_net(class_num=class_num)
    server_net.to(device)
    global_weight = server_net.state_dict()
    global_weight = strip_classifier(global_weight)
    if args.umrb:
        global_prototype = [dict() for _ in range(len(net))]
    else:
        global_prototype = {i: torch.zeros(2048).to(device) for i in range(class_num)}
    moon_prev_net = None
    if args.fl_opt == "moon":
        server_net.eval()
    if args.fl_opt == "moon":
        moon_prev_net = embed_net(class_num=1)
        moon_prev_net.to(device)
        moon_prev_net.eval()
    if args.fl_opt != "fedavg":
        print("==> FL baseline: {}".format(args.fl_opt))
    if args.umrb:
        print("==> CGPR correspondence rule: {}".format("mnn"))
    print("==> Building success!..")
    cudnn.benchmark = not args.deterministic
    criterion_id = nn.CrossEntropyLoss()
    criterion_tri = CircleLoss(margin=0.45, gamma=64)
    criterion_pro = PrototypeLoss(weight1=args.pro_w1, weight2=args.pro_w2, mse_scale=1.0)
    criterion_id.to(device)
    criterion_tri.to(device)
    criterion_pro.to(device)
    client_optimizer = []
    for i in range(len(net)):
        sub_optim = get_optimizer(net[i], args.lr)
        client_optimizer.append(sub_optim)

    def adjust_lr(optimizer, batch_idx, lrs):
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lrs[batch_idx - 1] * args.lr_multipliers[idx]
        return (optimizer, optimizer.param_groups[-1]["lr"], optimizer.param_groups[0]["lr"])

    def train_client(
        epoch,
        optimizer,
        net,
        args,
        global_weight,
        local_prototype,
        global_prototype,
        expected_cls=None,
    ):
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        pro_loss = AverageMeter()
        baseline_loss = AverageMeter() if args.fl_opt != "fedavg" else None
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0
        net.to(device)
        if moon_prev_net is not None and epoch > start_epoch:
            moon_prev_net.load_state_dict(strip_classifier(net.state_dict()), strict=False)
            moon_prev_net.eval()
        prox_ref = None
        if args.fl_opt == "fedprox":
            prox_ref = {k: v.detach().clone() for (k, v) in global_weight.items()}
        net.load_state_dict(global_weight, strict=False)
        assert not any(
            (k.startswith("classifier.") for k in global_weight)
        ), "global weight still carries a classification head"
        if expected_cls is not None:
            assert torch.equal(
                net.classifier.weight, expected_cls
            ), "client head was clobbered between rounds"
        net.train()
        end = time.time()
        lr_start = get_cyclic_lr(epoch, args.lr, args.max_epoch, args.decay_step)
        lr_end = get_cyclic_lr(epoch + 1, args.lr, args.max_epoch, args.decay_step)
        iters = len(trainloader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])
        local_prototype.reset_memory()
        for batch_idx, (input, label, cam) in enumerate(trainloader):
            (optimizer, current_lr, base_lr) = adjust_lr(optimizer, batch_idx, lrs)
            optimizer.zero_grad()
            label = torch.tensor(label, dtype=torch.long).to(device)
            data_time.update(time.time() - end)
            with autocast():
                inp = input.to(device)
                res = net(inp)
                (B, C) = res["feat"].shape
                if not args.no_mrb:
                    local_prototype.update_memory(res["feat"], label)
                loss_id = criterion_id(res["cls_id"], label)
                loss_tri = criterion_tri(res["feat"], label) / B
                if epoch == 0 or args.no_mrb:
                    loss = loss_id + loss_tri
                else:
                    feat_for_pro = res["feat"]
                    loss_pro = criterion_pro(feat_for_pro, label, global_prototype, device)
                    loss = (
                        loss_id
                        + loss_tri
                        + (0.0 if args.submission_variant == "head_only" else args.proto_lambda)
                        * loss_pro
                    )
                # These regularizers belong only to the external baselines.
                loss_baseline = None
                if args.fl_opt == "fedprox":
                    loss_baseline = 0.5 * args.prox_mu * fedprox_term(net.named_parameters(), prox_ref)
                elif args.fl_opt == "moon" and epoch > start_epoch:
                    with torch.no_grad():
                        (_, z_glob) = server_net(inp)
                        (_, z_prev) = moon_prev_net(inp)
                    loss_baseline = args.moon_mu * moon_loss(
                        res["feat"], z_glob, z_prev, args.moon_temp
                    )
                elif args.fl_opt == "fedrcl":
                    loss_baseline = args.fedrcl_weight * multilayer_fedrcl_loss(
                        res["rcl_features"],
                        label,
                        temperature=args.fedrcl_temp,
                        threshold=args.fedrcl_threshold,
                    )
                if loss_baseline is not None:
                    loss = loss + loss_baseline
                (_, predicted) = res["cls_id"].max(1)
                correct += predicted.eq(label).sum().item()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            if args.fl_opt == "fedrcl":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.fedrcl_clip)
            scaler.step(optimizer)
            scaler.update()
            if epoch == 0 or args.no_mrb:
                train_loss.update(loss.item(), input.size(0))
                id_loss.update(loss_id.item(), input.size(0))
                tri_loss.update(loss_tri.item(), input.size(0))
            else:
                train_loss.update(loss.item(), input.size(0))
                id_loss.update(loss_id.item(), input.size(0))
                tri_loss.update(loss_tri.item(), input.size(0))
                pro_loss.update(loss_pro.item(), input.size(0))
            if loss_baseline is not None:
                baseline_loss.update(loss_baseline.item(), input.size(0))
            total += label.size(0)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 20 == 0:
                baseline_log = ""
                if baseline_loss is not None and baseline_loss.count:
                    baseline_log = " {}Loss: {:.4f} ({:.4f})".format(
                        {"fedprox": "FedProx", "moon": "MOON", "fedrcl": "FedRCL"}[args.fl_opt],
                        baseline_loss.val,
                        baseline_loss.avg,
                    )
                print(
                    "Epoch: [{}][{}/{}] Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) lr:{:.6f} base_lr:{:.6f} Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) CLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) PLoss: {pro_loss.val:.4f} ({pro_loss.avg:.4f}){baseline_log} Accu: {:.2f}".format(
                        epoch,
                        batch_idx,
                        len(trainloader),
                        current_lr,
                        base_lr,
                        100.0 * correct / total,
                        batch_time=batch_time,
                        train_loss=train_loss,
                        id_loss=id_loss,
                        tri_loss=tri_loss,
                        pro_loss=pro_loss,
                        baseline_log=baseline_log,
                    )
                )
        extra = {}

        def _upload_sd():
            sd = net.state_dict()
            return strip_classifier(sd) if not args.ufed_head else sd

        if args.no_mrb:
            return (_upload_sd(), {}, extra)
        representative_local_prototypes = local_prototype.get_representative_feature()
        return (_upload_sd(), representative_local_prototypes, extra)

    def fed_avg(weight_local, identity_num):
        return fed_avg_sd(weight_local, identity_num, skip_prefixes=("classifier.",))

    print("==> Start Training...")
    scaler = GradScaler()
    for epoch in range(start_epoch, args.max_epoch):
        weight_local = []
        identity_num = []
        representative_local_prototype = []
        for index in range(len(train_set)):
            color_pos = build_identity_index(train_set[index].train_label)
            sampler = IdentitySampler(
                train_set[index].train_label, color_pos, args.num_pos, args.batch_size
            )
            train_set[index].Index = sampler.index
            loader_batch = args.batch_size * args.num_pos
            id_num = len(train_set[index])
            id_num = id_num - id_num % loader_batch
            identity_num.append(id_num)
            trainloader = data.DataLoader(
                train_set[index],
                batch_size=loader_batch,
                sampler=sampler,
                drop_last=True,
                num_workers=args.workers,
                multiprocessing_context="fork" if args.workers else None,
            )
            print("Client{} training".format(index + 1))
            expected_cls = None
            if epoch > start_epoch:
                expected_cls = net[index].classifier.weight.detach().clone()
            (w, rep_local_proto, extra) = train_client(
                epoch,
                client_optimizer[index],
                net=net[index],
                args=args,
                global_weight=global_weight,
                local_prototype=local_prototypes[index],
                global_prototype=global_prototype[index] if args.umrb else global_prototype,
                expected_cls=expected_cls,
            )
            weight_local.append(w)
            representative_local_prototype.append(rep_local_proto)
        if epoch >= 0:
            print("Start Communication...")
            global_weight = fed_avg(weight_local, identity_num)
            if args.umrb:
                (groups, gstats) = mutual_nn_groups(
                    representative_local_prototype, tau=args.umrb_tau
                )
                audit = controls.audit(groups, persons_by_client)
                audit.update(epoch=epoch, **gstats)
                control_history.append(audit)
                print("CORRESPONDENCE " + json.dumps(audit, sort_keys=True))
                global_prototype = umrb_broadcast(
                    representative_local_prototype, groups, prev=None
                )
                global_prototype = [
                    {l: v.to(device) for (l, v) in d.items()} for d in global_prototype
                ]
                (prec, npairs) = match_precision(groups, persons_by_client)
                print(
                    "CGPR epoch {}: protos={} multi_groups={} matched_frac={:.3f} max_group={} match_precision={} pairs={}".format(
                        epoch,
                        gstats["n_protos"],
                        gstats["n_groups_multi"],
                        gstats["matched_frac"],
                        gstats["max_group"],
                        "{:.3f}".format(prec) if prec is not None else "n/a",
                        npairs,
                    )
                )
                if args.ufed_head:
                    _cls = [w["classifier.weight"].detach().clone() for w in weight_local]
                    _upd = ufed_head_rows(_cls, groups)
                    with torch.no_grad():
                        for _c, _rows in enumerate(_upd):
                            for _l, _row in _rows.items():
                                net[_c].classifier.weight.data[_l] = _row.to(device)
                    print(
                        "CGCF epoch {}: rows federated per client {}".format(
                            epoch, [len(r) for r in _upd]
                        )
                    )
            else:
                record = controls.audit([], persons_by_client)
                record.update(epoch=epoch)
                control_history.append(record)
                print("CORRESPONDENCE " + json.dumps(record, sort_keys=True))
            server_net.load_state_dict(global_weight, strict=False)
            print("Success!")
            print("-" * 30)
        if epoch == args.max_epoch - 1:
            state = {"net": server_net.state_dict(), "epoch": epoch}
            torch.save(state, model_path + "global.pth")
    with open(model_path + "history.json", "w") as f:
        json.dump(control_history, f, indent=2)
    if not args.skip_evaluation:
        from l2rwplus.evaluation.image import evaluate

        metrics = evaluate(server_net, args.dataset, args.trial, args.workers)
        with open(model_path + "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("FINAL", json.dumps(metrics))
    print("Training complete:", model_path + "global.pth")
