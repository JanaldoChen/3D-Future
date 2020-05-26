import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.gbottleneck import GBottleneck
from networks.graph_unpooling import GraphUnpooling
from networks.graph_projection import GraphProjection

class Pixel2Mesh(nn.Module):
    def __init__(self, ellipsoid, device=None, hidden_dim=192, feat_dim=1280, coor_dim=3, adjust_ellipsoid=False):
        super(Pixel2Mesh, self).__init__()