"""FedRCL (CVPR 2024) adapted to the existing ResNet-50 VI-ReID encoder.

Reference: https://github.com/skynbe/FedRCL, commit
378beb409add17838567ea25238afadb3258514f. Implements its released defaults:
one hardest positive, two hardest inter-class negatives, and the thresholded
intra-class penalty. Both terms are averaged over five pre-ReLU layers.
"""

import torch
import torch.nn.functional as F

from .image_encoder import ImageEncoder


def fedrcl_terms(features, labels, temperature=0.05, threshold=0.7):
    """Return the released implementation's SCL and divergence terms.

    The reference keeps the diagonal among SCL positive candidates; its
    penalty uses the least-similar other same-class example (topk_pos=1),
    replacing absent/below-threshold candidates with cosine -1. These details
    intentionally follow the code rather than substituting ordinary SupCon.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=1)
        sim = z @ z.t()
        same = labels[:, None].eq(labels[None, :])
        diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive = sim.masked_fill(~same, float("inf")).min(1, keepdim=True).values
        negative = sim.masked_fill(same, -float("inf")).topk(min(2, len(labels)), dim=1).values
        target = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
        scl = F.cross_entropy(torch.cat([positive, negative], dim=1) / temperature, target)
        within = sim.masked_fill(~same | diagonal, float("inf")).min(1, keepdim=True).values
        within = torch.where(
            torch.isfinite(within) & (within >= threshold), within, torch.full_like(within, -1)
        )
        self_sim = sim.diag().unsqueeze(1)
        penalty = F.cross_entropy(torch.cat([self_sim, within], dim=1) / temperature, target)
    return scl, penalty


def multilayer_fedrcl_loss(features, labels, temperature=0.05, threshold=0.7):
    if len(features) != 5:
        raise ValueError("FedRCL requires stem and four residual-stage features")
    terms = [fedrcl_terms(z, labels, temperature, threshold) for z in features]
    return torch.stack([scl + penalty for scl, penalty in terms]).mean()


class FedRCLEmbedNet(ImageEncoder):
    """Observe pre-activation features without changing the retrieval path.

    The inherited model, parameters, state-dict keys and inference are intact.
    Pre-hooks pool the stem BN output and each stage's final residual sum
    before the in-place ReLU. No extra forward pass or global model is needed:
    the official default loss only uses local--local (``nn``) pairs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rcl_features = {}
        self._rcl_counts = {}
        backbone = self.base.base

        def hook_for(layer, calls):
            def capture(module, inputs):
                if not self.training:
                    return
                self._rcl_counts[layer] = self._rcl_counts.get(layer, 0) + 1
                if self._rcl_counts[layer] == calls:
                    self._rcl_features[layer] = F.adaptive_avg_pool2d(inputs[0], 1).flatten(1)

            return capture

        backbone.relu.register_forward_pre_hook(hook_for(0, 1))
        for index in range(1, 5):
            # ResNet-50 Bottleneck reuses its ReLU three times; the last call
            # follows bn3 + residual and supplies the pre-activation feature.
            stage = getattr(backbone, "layer" + str(index))
            stage[-1].relu.register_forward_pre_hook(hook_for(index, 3))

    def forward(self, x):
        self._rcl_features = {}
        self._rcl_counts = {}
        result = super().forward(x)
        if self.training:
            result["rcl_features"] = [self._rcl_features[i] for i in range(5)]
        self._rcl_features = {}
        return result
