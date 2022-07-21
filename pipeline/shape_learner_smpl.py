import os
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from model.smpl.body_model import BodyModel

from pipeline.shape_learner import ShapeLearner


class ShapeLearnerSMPL(ShapeLearner):
    """A generic object for neural network experiments."""

    def _init_model(self):
        self.model = nn.Module()

        # init parameters
        self.model.register_parameter('beta', nn.Parameter(torch.zeros(self.cfg_model.num_beta)))
        self.model.register_parameter('trans', nn.Parameter(torch.zeros(3)))
        self.model.register_parameter('theta', nn.Parameter(torch.zeros(24*3)))

        # smpl-1.0
        self.bm_dict = {
            'female': os.path.join(self.cfg_model.smpl_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'),
            'male': os.path.join(self.cfg_model.smpl_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        }
    
    def smpl_forward(self, batch):
        gender = batch['gender'][0]
        if isinstance(self.bm_dict[gender], str):
            # body model from VPoser
            logging.info(f'Loading SMPL body model from: {self.bm_dict[gender]}.')
            self.bm_dict[gender] = BodyModel(bm_fname=self.bm_dict[gender], num_betas=self.cfg_model.num_beta, device='cuda')

        # for body model from VPoser
        body_parms = {
            'betas': self.model.beta[None, :],
            'root_orient': self.model.theta[None, :3],  # 0
            'trans': self.model.trans[None, :],  # 0
            'pose_body': self.model.theta[None, 3:66],  # 0
            'pose_hand': self.model.theta[None, 66:],  # 0
        }

        clip_data = self.bm_dict[gender](**body_parms)
        pred_verts = clip_data.v

        return pred_verts
    
    def forward(self, batch, is_testing=False):
        # prepare data
        gt_verts = batch['verts'].cuda()
        pred_verts = self.smpl_forward(batch)

        result_dict = {
            'pred_verts': pred_verts,
            'gt_verts': gt_verts,
        }

        # loss computation
        loss_dict = {}
        
        ## vertex loss
        loss_vert = (pred_verts - gt_verts).abs().mean()
        # loss_vert = ((pred_verts - gt_verts)**2).mean()
        loss_dict['loss_vert'] = loss_vert
        
        ## total loss
        loss_dict['loss'] = sum(map(lambda x: x[1], loss_dict.items()))

        # metrics computation
        if is_testing:
            with torch.no_grad():
                pass
        
        return result_dict, loss_dict


    def _save_results(self, mode, result_dict, epoch, step, num_epochs, steps_per_epoch):
        torch.cuda.synchronize()
        tic_save = time.time()

        if self.cfg_exp.save_points:
            self._save_pts_3d(result_dict['pred_verts'], mode, epoch, step, name='verts_pred')
            self._save_pts_3d(result_dict['gt_verts'], mode, epoch, step, name='verts_gt')
        
        self._save_beta(self.model.beta, mode, epoch, step)

        self._write_logs_save(mode, epoch, step, tic_save, num_epochs, steps_per_epoch)
    
    def _save_beta(self, beta, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'beta', name)

        out_path = base_path + '-%06d%s.npy' % (base_idx, name)

        beta = beta.detach().cpu().numpy()
        np.save(out_path, beta)
        
    