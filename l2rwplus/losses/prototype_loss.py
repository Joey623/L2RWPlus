"""Feature alignment to the server-provided identity prototypes."""

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeLoss(nn.Module):
    def __init__(self, weight1=0.5, weight2=0.5, mse_scale=1.0):
        """Combine mean squared feature error and cosine distance."""
        super(PrototypeLoss, self).__init__()

        self.weight1 = weight1
        self.weight2 = weight2
        self.mse_scale = mse_scale

    def forward(self, x, label, global_prototype, device):
        B, _ = x.shape
        edu_dist = []
        cos_dist = []
        for i in range(B):
            class_label = label[i].item()
            prototype = torch.tensor(global_prototype[class_label]).to(device)
            feature = x[i]
            prototype_distance = F.mse_loss(feature, prototype)
            prototype_cosine = 1 - F.cosine_similarity(feature, prototype, dim=0)
            edu_dist.append(prototype_distance)
            cos_dist.append(prototype_cosine)
        edu_dist = torch.stack(edu_dist)
        cos_dist = torch.stack(cos_dist)
        loss = self.weight1 * self.mse_scale * edu_dist.mean() + self.weight2 * cos_dist.mean()

        return loss

