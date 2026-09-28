from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from .backbone import ResNetBackbone
from .pooling import GeMP, TemporalMeanPooling
from .weight_init import weights_init_classifier, weights_init_kaiming


class VideoEncoder(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(VideoEncoder, self).__init__()
        self.base = ResNetBackbone(pretrained=pretrained)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()
        self.temporal_module = TemporalMeanPooling(feat_dim=pool_dim)

    @autocast()
    def forward(self, x, seq_len=6):
        # [b,tc,h,w]
        b, c, h, w = x.shape
        t = seq_len
        # [bt,c,h,w]
        x = x.view(int(b * t), int(c / t), h, w)
        feat = self.base(x)
        b, c, h, w = feat.shape
        feat = self.relu(feat)
        feat = feat.view(b, c, h * w)
        # [bt,c]
        frame_feat = self.pool(feat)
        # [b,t,c]
        feat = frame_feat.view(frame_feat.size(0) // t, t, -1)
        # [b,c]
        feat = self.temporal_module(feat)
        feat_after_BN = self.bottleneck(feat)

        if self.training:
            cls_id = self.classifier(feat_after_BN)

            res = {
                "cls_id": cls_id,
                "feat": feat_after_BN,
            }
            return res
        else:
            return F.normalize(feat, p=2.0, dim=1), F.normalize(feat_after_BN, p=2.0, dim=1)
