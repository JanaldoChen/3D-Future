import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.encoder = get_backbone()

        # Save necessary helper matrices in respective variables
        self.initial_coordinates = nn.Parameter(torch.tensor(ellipsoid[0]), requires_grad=False)
        if adjust_ellipsoid:
            ''' This is the inverse of the operation the Pixel2mesh authors'
            performed to original CAT model; it ensures that the ellipsoid
            has the same size and scale in the not-transformed coordinate
            system we are using. '''
            print("Adjusting ellipsoid.")
            self.initial_coordinates = self.initial_coordinates / 0.57
            self.initial_coordinates[:, 1] = -self.initial_coordinates[:, 1]
            self.initial_coordinates[:, 2] = -self.initial_coordinates[:, 2]

        pool_idx_1 = torch.tensor(ellipsoid[4][0])  # IDs for the first unpooling operation
        pool_idx_2 = torch.tensor(ellipsoid[4][1])  # IDs for the second unpooling operation

        # sparse support matrices for graph convolution; the indices need to
        # be transposed to match pytorch standards
        ell_1 = ellipsoid[1][1]
        e1, e2, e3 = torch.tensor(ell_1[0]).transpose_(0, 1), torch.tensor(ell_1[1]), torch.tensor(ell_1[2])
        adj_mat_1 = torch.sparse.FloatTensor(e1.long(), e2, torch.Size(e3))

        ell_2 = ellipsoid[2][1]
        e1, e2, e3 = torch.tensor(ell_2[0]).transpose_(0, 1), torch.tensor(ell_2[1]), torch.tensor(ell_2[2])
        adj_mat_2 = torch.sparse.FloatTensor(e1.long(), e2, torch.Size(e3))
        
        ell_3 = ellipsoid[3][1]
        e1, e2, e3 = torch.tensor(ell_3[0]).transpose_(0, 1), torch.tensor(ell_3[1]), torch.tensor(ell_3[2])
        adj_mat_3 = torch.sparse.FloatTensor(e1.long(), e2, torch.Size(e3))

        gconv_activation = nn.ReLU()

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.feat_dim, self.hidden_dim, self.coord_dim,
                        adj_mat_1, activation=gconv_activation),
            GBottleneck(6, self.feat_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        adj_mat_2, activation=gconv_activation),
            GBottleneck(6, self.feat_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        adj_mat_3, activation=gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(pool_idx_1.long()),
            GUnpooling(pool_idx_2.long())
        ])

        self.projection = GraphProjection()

    def forward(self, img, cam):
        batch_size = img.size(0)
        img_feat = self.encoder(img)

        init_pts = self.initial_coordinates.data.unsqueeze(0).expand(batch_size, -1, -1)
        
        # GCN Block 1
        x = self.projection(init_pts, img_feat, cam)
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

        return {'pred_coord': [x1, x2, x3], 'pred_coord_before_deform': [init_pts, x1_up, x2_up]}


