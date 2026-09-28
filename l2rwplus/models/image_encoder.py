from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from .backbone import ResNetBackbone
from .pooling import GeMP
from .weight_init import weights_init_classifier, weights_init_kaiming


class ImageEncoder(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.base = ResNetBackbone(pretrained=pretrained)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()

    @autocast()
    def forward(self, x):

        feat = self.base(x)
        if self.training:
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            cls_id = self.classifier(feat_after_BN)

            return {
                "cls_id": cls_id,
                "feat": feat_after_BN,
            }
        else:
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            return F.normalize(feat, p=2.0, dim=1), F.normalize(feat_after_BN, p=2.0, dim=1)
