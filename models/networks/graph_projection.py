import torch
import torch.nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class GraphProjection(nn.Module):
    """Graph Projection layer.
        Projects the predicted point cloud to the respective 2D coordinates
        given the camera and world matrix, and returns the concatenated
        features from the respective locations for each point
    """

    def __init__(self):
        super(GraphProjection, self).__init__()

    def visualise_projection(self, points_img, img, output_file='./out.png'):
        ''' Visualises the vertex projection to the image plane.
            Args:
                points_img (numpy array): points projected to the image plane
                img (numpy array): image
                output_file (string): where the result should be saved
        '''
        plt.imshow(img.transpose(1, 2, 0))
        plt.plot(
            (points_img[:, 0] + 1)*img.shape[1]/2,
            (points_img[:, 1] + 1) * img.shape[2]/2, 'x')
        plt.savefig(output_file)

    def forward(self, x, fm, camera_mat, img=None, visualise=False):
        ''' Performs a forward pass through the GP layer.
        Args:
            x (tensor): coordinates of shape (batch_size, num_vertices, 3)
            f (list): list of feature maps from where the image features
                        should be pooled
            camera_mat (tensor): camera matrices for transformation to 2D
                        image plane
            img (tensor): images (just fo visualisation purposes)
        '''
        points_img = common.project_to_camera(x, camera_mat)
        points_img = points_img.unsqueeze(1)
        feats = []
        feats.append(x)
        for fmap in fm:
            # bilinearly interpolate to get the corresponding features
            feat_pts = F.grid_sample(fmap, points_img)
            feat_pts = feat_pts.squeeze(2)
            feats.append(feat_pts.transpose(1, 2))
        # Just for visualisation purposes
        if visualise and (img is not None):
            self.visualise_projection(
                points_img.squeeze(1)[0].detach().cpu().numpy(),
                img[0].cpu().numpy())

        outputs = torch.cat([proj for proj in feats], dim=2)
        return outputs