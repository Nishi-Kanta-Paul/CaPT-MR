import math

import torch
import torch.nn as nn

from .backbones import build_backbone
from .heads import ALLOCATION, AllocationHead
from .baselines import ProtoNetHead, DN4Head, DMN4Head, DeepEMDHead, FRNHead, BiFRNHead, BSNetHead, TDMHead, SFMRHead


BASELINES = {
    'ProtoNet': lambda channels, cfg: ProtoNetHead(),
    'DN4': lambda channels, cfg: DN4Head(cfg['dn4_neighbours'], cfg['floor']),
    'DMN4': lambda channels, cfg: DMN4Head(cfg['floor']),
    'DeepEMD': lambda channels, cfg: DeepEMDHead(cfg),
    'FRN': lambda channels, cfg: FRNHead(channels),
    'Bi-FRN': lambda channels, cfg: BiFRNHead(channels),
    'BSNet': lambda channels, cfg: BSNetHead(channels),
    'TDM': lambda channels, cfg: TDMHead(channels),
    'SFMR': lambda channels, cfg: SFMRHead(channels, cfg['sfmr_radius']),
}
METHODS = tuple(ALLOCATION) + tuple(BASELINES)


class FewShotModel(nn.Module):
    def __init__(self, backbone, head, tau):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    def forward(self, support, query, timer=None, **kwargs):
        n, k = support.shape[:2]
        feats = self.backbone(torch.cat([support.flatten(0, 1), query]))
        if timer is not None:
            timer.mark('encoder')
        fs = feats[:n * k].view(n, k, *feats.shape[1:])
        out = self.head(feats[n * k:], fs, timer=timer, **kwargs)
        out['logits'] = self.log_tau.exp() * out['score']
        return out


def build_model(method, backbone_name, cfg):
    backbone = build_backbone(backbone_name)
    if method in ALLOCATION:
        head = AllocationHead({**cfg, **ALLOCATION[method]})
    else:
        head = BASELINES[method](backbone.channels, cfg)
    return FewShotModel(backbone, head, cfg['tau_init'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
