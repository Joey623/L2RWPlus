import torch
from torch import nn


class PrototypeMemory(nn.Module):
    def __init__(self, class_num, feature_dim=2048, top_k=4, device="cuda"):
        self.device = device
        super().__init__()
        self.memory = {i: [] for i in range(class_num)}
        self.n_class, self.feature_dim, self.top_k = class_num, feature_dim, top_k

    def update_memory(self, features, labels):
        for i in range(features.size(0)):
            self.memory[labels[i].item()].append(features[i].detach())

    def get_representative_feature(self):
        out = {i: torch.zeros(self.feature_dim).to(self.device) for i in range(self.n_class)}
        for label in sorted(self.memory):
            feats = self.memory[label]
            if not feats:
                continue
            if len(feats) < self.top_k:
                out[label] = torch.stack(feats).mean(0)
            else:
                mean = torch.stack(feats).mean(0)
                d = torch.stack(
                    [
                        torch.nn.functional.pairwise_distance(mean.unsqueeze(0), f.unsqueeze(0))
                        for f in feats
                    ]
                )
                sel = torch.argsort(d, dim=0)[: self.top_k]
                out[label] = torch.stack([feats[k] for k in sel]).mean(0)
        return out

    def reset_memory(self):
        self.memory = {i: [] for i in range(self.n_class)}
