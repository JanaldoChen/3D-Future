import torch
import torch.nn as nn

class GraphUnpooling(nn.Module):
    ''' Graph Unpooling Layer.
        Unpools additional vertices following the helper file and uses the
        average feature vector from the two adjacent vertices
    '''

    def __init__(self, pool_idx_array):
        ''' Initialisation
        Args:
            pool_idx_array (tensor): vertex IDs that should be comined to new
            vertices
        '''
        super(GraphUnpooling, self).__init__()
        self.pool_x1 = pool_idx_array[:, 0]
        self.pool_x2 = pool_idx_array[:, 1]

    def forward(self, x):
        num_new_v = len(self.pool_x1)
        batch_size = x.shape[0]
        num_feats = x.shape[2]

        x1 = x[:, self.pool_x1.long(), :]
        x2 = x[:, self.pool_x2.long(), :]
        new_v = torch.add(x1, x2).mul(0.5)
        assert(new_v.shape == (batch_size, num_new_v, num_feats))
        out = torch.cat([x, new_v], dim=1)
        return out