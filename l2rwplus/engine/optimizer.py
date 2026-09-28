"""SGD parameter groups for the backbone, neck and classifier."""

import torch.optim as optim


def get_optimizer(net, lr):
    ignored = list(map(id, net.bottleneck.parameters())) + list(
        map(id, net.classifier.parameters())
    )
    base = filter(lambda p: id(p) not in ignored, net.parameters())
    return optim.SGD(
        [
            {"params": base, "lr": 0.1 * lr},
            {"params": net.bottleneck.parameters(), "lr": lr},
            {"params": net.classifier.parameters(), "lr": lr},
        ],
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )

