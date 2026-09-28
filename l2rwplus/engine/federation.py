"""Independent client labels and correspondence-guided federation."""

import copy
import numpy as np

CLASSIFIER_PREFIXES = ("classifier.",)


def private_relabel(train_label, client_id, seed):
    """Map a client's global labels onto a PRIVATE contiguous space 0..n_c-1."""
    labels = np.asarray(train_label)
    uniq = np.unique(labels)
    rng = np.random.RandomState(seed * 10007 + client_id)
    perm = rng.permutation(len(uniq))
    mapping = {int(g): int(p) for (g, p) in zip(uniq, perm)}
    new = np.array([mapping[int(l)] for l in labels], dtype=np.int64)
    return (new, mapping)


def strip_classifier(state_dict, prefixes=CLASSIFIER_PREFIXES):
    """Copy of state_dict without the classification-head keys."""
    return {k: v for (k, v) in state_dict.items() if not k.startswith(tuple(prefixes))}


def fed_avg_sd(weight_local, identity_num, skip_prefixes=()):
    """Sample-count-weighted FedAvg, optionally skipping key prefixes."""
    w_avg = copy.deepcopy(weight_local[0])
    if skip_prefixes:
        for k in list(w_avg.keys()):
            if k.startswith(tuple(skip_prefixes)):
                del w_avg[k]
    sample_num = sum(identity_num)
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * (identity_num[0] / sample_num)
    for k in w_avg.keys():
        for i in range(1, len(weight_local)):
            w_avg[k] = w_avg[k] + weight_local[i][k] * identity_num[i] / sample_num
    return w_avg


def person_of_key(key):
    """Read the person directory for evaluation-only association diagnostics."""
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError("key has no person directory: {}".format(key))
    return parts[-2]


def alignment_overlap_maps(persons_by_client):
    """Core of the alignment check, on precomputed {local_label: person} maps."""
    persons = persons_by_client
    out = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            shared = set(persons[i]) & set(persons[j])
            same = sum((1 for l in shared if persons[i][l] == persons[j][l]))
            (si, sj) = (set(persons[i].values()), set(persons[j].values()))
            chance = len(si & sj) / (len(si) * len(sj)) if si and sj else 0.0
            out.append(
                {
                    "pair": (i + 1, j + 1),
                    "shared_labels": len(shared),
                    "same_person": same,
                    "frac": same / len(shared) if shared else 0.0,
                    "chance": chance,
                }
            )
    return out


def relabel_video_tracklets(train_tuples, client_id, seed):
    """Privately relabel (frames, pid, camera) tracklets at one client."""
    pids = [t[1] for t in train_tuples]
    (new_labels, mapping) = private_relabel(pids, client_id, seed)
    new_tuples = [(t[0], int(nl), t[2]) for (t, nl) in zip(train_tuples, new_labels)]
    persons = {int(nl): str(op) for (op, nl) in mapping.items()}
    return (new_tuples, [int(x) for x in new_labels], persons)


def alignment_overlap(keys_by_client, labels_by_client):
    """Report agreement between numerical labels, for diagnostics only."""
    persons = []
    for keys, labels in zip(keys_by_client, labels_by_client):
        m = {}
        for k, l in zip(keys, labels):
            p = person_of_key(k)
            prev = m.setdefault(int(l), p)
            if prev != p:
                raise ValueError("label {} maps to persons {} and {}".format(l, prev, p))
        persons.append(m)
    return alignment_overlap_maps(persons)


def mutual_nn_groups(protos_by_client, tau=0.0):
    """Infer cross-client MNN groups using nonzero prototypes only."""
    import torch
    import torch.nn.functional as Fn

    items = []
    vecs = []
    for c, d in enumerate(protos_by_client):
        for l in sorted(d):
            v = d[l].detach().float().cpu()
            if float(v.norm()) > 0:
                items.append((c, int(l)))
                vecs.append(v)
    if not items:
        return (
            [],
            {
                "n_protos": 0,
                "n_edges": 0,
                "n_groups_multi": 0,
                "matched_frac": 0.0,
                "max_group": 0,
            },
        )
    X = Fn.normalize(torch.stack(vecs), dim=1)
    by_client = {}
    for idx, (c, _) in enumerate(items):
        by_client.setdefault(c, []).append(idx)
    edges = []
    clients = sorted(by_client)
    for ai in range(len(clients)):
        for bi in range(ai + 1, len(clients)):
            (ia, ib) = (by_client[clients[ai]], by_client[clients[bi]])
            S = X[ia] @ X[ib].t()
            fwd = S.argmax(dim=1)
            bwd = S.argmax(dim=0)
            for i in range(len(ia)):
                j = int(fwd[i])
                if int(bwd[j]) == i and float(S[i, j]) >= tau:
                    edges.append((float(S[i, j]), ia[i], ib[j]))
    edges.sort(key=lambda e: (-e[0], items[e[1]], items[e[2]]))
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    group_clients = {i: {items[i][0]} for i in range(len(items))}
    n_used = 0
    for _, u, v in edges:
        (ru, rv) = (find(u), find(v))
        if ru == rv or group_clients[ru] & group_clients[rv]:
            continue
        parent[rv] = ru
        group_clients[ru] |= group_clients[rv]
        n_used += 1
    by_root = {}
    for i in range(len(items)):
        by_root.setdefault(find(i), []).append(items[i])
    groups = [sorted(v) for v in by_root.values()]
    groups.sort()
    multi = [g for g in groups if len(g) > 1]
    stats = {
        "n_protos": len(items),
        "n_edges": n_used,
        "n_groups_multi": len(multi),
        "matched_frac": sum((len(g) for g in multi)) / len(items),
        "max_group": max((len(g) for g in groups), default=0),
    }
    return (groups, stats)


def umrb_broadcast(protos_by_client, groups, prev=None):
    """Group means, indexed back into each client's PRIVATE label space."""
    import torch

    out = [dict() for _ in protos_by_client]
    for g in groups:
        vs = [protos_by_client[c][l].detach().float() for (c, l) in g]
        center = torch.stack([v.cpu() for v in vs]).mean(0)
        for c, l in g:
            out[c][l] = center.to(protos_by_client[c][l].device)
    for c, d in enumerate(protos_by_client):
        for l in d:
            if int(l) not in out[c]:
                v = d[l].detach().float()
                if prev is not None and float(v.norm()) == 0.0 and (int(l) in prev[c]):
                    v = prev[c][int(l)].detach().float().to(v.device)
                out[c][int(l)] = v
    return out


def ufed_head_rows(cls_weights, groups):
    """CGCF: average classifier rows within matched groups."""
    import torch

    out = [dict() for _ in cls_weights]
    for g in groups:
        if len(g) < 2:
            continue
        rows = torch.stack([cls_weights[c][l].detach().float() for (c, l) in g])
        avg = rows.mean(0)
        for c, l in g:
            out[c][l] = avg.to(cls_weights[c].dtype)
    return out


def match_precision(groups, persons_by_client):
    """Evaluate inferred pairs; ground truth is not an input to matching."""
    same = total = 0
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                (ca, la) = g[a]
                (cb, lb) = g[b]
                total += 1
                if persons_by_client[ca][la] == persons_by_client[cb][lb]:
                    same += 1
    return (same / total if total else None, total)


def assert_unaligned(overlaps, threshold=0.05):
    """Check that relabeling removed a shared numerical identity index."""
    bad = [o for o in overlaps if o["frac"] > threshold]
    if bad:
        raise AssertionError("label spaces still aligned: {}".format(bad))
    return True


def fedprox_term(named_params, global_sd, skip_prefixes=CLASSIFIER_PREFIXES):
    """FedProx (Li et al., MLSys 2020) proximal term sum_k ||w_k - w_k^g||^2"""
    import torch

    total = None
    for k, p in named_params:
        if k.startswith(tuple(skip_prefixes)) or k not in global_sd:
            continue
        d = (p.float() - global_sd[k].detach().float()).pow(2).sum()
        total = d if total is None else total + d
    if total is None:
        return torch.zeros(())
    return total


def moon_loss(z, z_glob, z_prev, temperature=0.5):
    """MOON model-contrastive loss against global and previous local encoders."""
    import torch
    import torch.nn.functional as Fn

    z = Fn.normalize(z.float(), dim=1)
    zg = Fn.normalize(z_glob.detach().float(), dim=1)
    zp = Fn.normalize(z_prev.detach().float(), dim=1)
    pos = (z * zg).sum(1, keepdim=True) / temperature
    neg = (z * zp).sum(1, keepdim=True) / temperature
    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return Fn.cross_entropy(logits, target)
