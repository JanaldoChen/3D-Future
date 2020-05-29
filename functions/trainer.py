import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.mesh import Ellipsoid
from models.pixel2mesh import Pixel2Mesh
from models.networks.loss import P2MLoss

class Trainer(object):
    def __init__(self, opt):
        self.options = opt
        self.ellipsoid = Ellipsoid(file=opt.ellipsoid_file)
        self.model = Pixel2Mesh(self.ellipsoid, hidden_dim=opt.hidden_dim, feat_dim=opt.feat_dim, coord_dim=opt.coord_dim, adjust_ellipsoid=opt.adjust_ellipsoid)
        self.criterion = P2MLoss(self.ellipsoid)

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=opt.lr,
            betas=(opt.adam_beta1, 0.999),
            weight_decay=opt.weight_decay
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, opt.lr_step, opt.lr_factor)

    def train_step(self, input):
        self.model.train()
        img = input['image'].to(self.device)
        points = input['points'].to(self.device)
        normals = input['normals'].to(self.device)

        world_mat, camera_mat = camera_args['Rt'], camera_args['K']
        out = self.model(img, camera_mat)
        loss, loss_summary = self.criterion(out, input)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        