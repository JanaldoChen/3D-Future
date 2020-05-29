import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnet import resnet18
from networks.gbottleneck import GBottleneck
from networks.graph_unpooling import GraphUnpooling
from networks.graph_projection import GraphProjection

class Pixel2Mesh(nn.Module):
    def __init__(self, ellipsoid, hidden_dim=192, feat_dim=1280, coord_dim=3, adjust_ellipsoid=False):
        super(Pixel2Mesh, self).__init__()
        self.hidden_dim  = hidden_dim
        self.feat_dim = feat_dim
        self.coord_dim = coord_dim

        # Encoder
        self.encoder = resnet18(pretrained=True)

        # Save necessary helper matrices in respective variables
        self.initial_coordinates = nn.Parameter(ellipsoid.coord, requires_grad=False)
        if adjust_ellipsoid:
            ''' This is the inverse of the operation the Pixel2mesh authors'
            performed to original CAT model; it ensures that the ellipsoid
            has the same size and scale in the not-transformed coordinate
            system we are using. '''
            print("Adjusting ellipsoid.")
            self.initial_coordinates = self.initial_coordinates / 0.57
            self.initial_coordinates[:, 1] = -self.initial_coordinates[:, 1]
            self.initial_coordinates[:, 2] = -self.initial_coordinates[:, 2]

        gconv_activation = nn.ReLU()

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.feat_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=gconv_activation),
            GBottleneck(6, self.feat_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=gconv_activation),
            GBottleneck(6, self.feat_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[2], activation=gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        self.projection = GraphProjection()

    def forward(self, img, camera_mat):
        batch_size = img.size(0)
        img_feat = self.encoder(img)

        init_pts = self.initial_coordinates.data.unsqueeze(0).expand(batch_size, -1, -1)
        
        # GCN Block 1
        x = self.projection(init_pts, img_feat, camera_mat)
        x1, x_hidden = self.gcns[0](x)

        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(x1, img_feat, cam)
        x = self.unpooling[0](torch.cat([x, x_hidden], dim=2))
        x2, x_hidden = self.gcns[1](x)

        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(x2, img_feat, cam)
        x = self.unpooling[0](torch.cat([x, x_hidden], dim=2))
        x3, _ = self.gcns[2](x)

        out = {
            'pred_coord': [x1, x2, x3],
            'pred_coord_before_deform': [init_pts, x1_up, x2_up]
        }
        return out


