import os
import logging
import time

import numpy as np
import cv2 as cv
import trimesh
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from pipeline.learner import Learner
    

class ShapeLearner(Learner):
    """A generic object for neural network experiments."""

    def _save_results(self, mode, result_dict, epoch, step, num_epochs, steps_per_epoch):
        torch.cuda.synchronize()
        tic_save = time.time()

        if self.cfg_exp.save_field:
            self._save_field_3d(result_dict, mode, epoch, step, name=None)

        if self.cfg_exp.save_surface:
            self._save_surface_3d(result_dict, mode, epoch, step, name=None)

        if self.cfg_exp.save_points:
            self._save_pts_3d(result_dict['pts_surface'], mode, epoch, step, name='surface')

        self._write_logs_save(mode, epoch, step, tic_save, num_epochs, steps_per_epoch)
    
    def _get_grid_pred(self, result_dict):
        pts = result_dict['pts_surface'].detach()
        priors = result_dict['priors']

        result_dict['grid_pts'], result_dict['grid_pred'] = self.vis.get_grid_pred(self.model, priors, pts)
    
    def _get_mesh_pred(self, result_dict):
        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        pts = result_dict['grid_pts']
        val = result_dict['grid_pred']

        B = pts.shape[0]
        result_dict['mesh_pred'] = []
        for idx in range(B):
            mesh = self.vis.get_mesh_from_grid(pts[idx], val[idx])
            result_dict['mesh_pred'].append(mesh)
    
    def _save_field_3d(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'field_3d', name)

        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        pts = result_dict['grid_pts']
        val = result_dict['grid_pred']

        cmap = 'RdYlBu_r'
        num_levels = 20

        B = pts.shape[0]
        for idx in range(B):
            out_path = base_path + '-%06d%s.png' % (base_idx+idx, name)

            fig, ax = plt.subplots(2, 2, dpi=150)

            v = val[idx, :, :, val.shape[3]//2, 0]
            x = pts[idx, :, :, val.shape[3]//2, 0]
            y = pts[idx, :, :, val.shape[3]//2, 1]
            ct = ax[0, 0].contourf(x, y, v, num_levels, cmap=cmap)
            ax[0, 0].axis('equal')
            ax[0, 0].axis('off')
            fig.colorbar(ct, ax=ax[0, 0])

            v = val[idx, val.shape[1]//2, :, :, 0]
            z = pts[idx, val.shape[1]//2, :, :, 2]
            y = pts[idx, val.shape[1]//2, :, :, 1]
            ct = ax[0, 1].contourf(z, y, v, num_levels, cmap=cmap)
            ax[0, 1].axis('equal')
            ax[0, 1].axis('off')
            fig.colorbar(ct, ax=ax[0, 1])

            v = val[idx, :, val.shape[2]//2, :, 0]
            x = pts[idx, :, val.shape[2]//2, :, 0]
            z = pts[idx, :, val.shape[2]//2, :, 2]
            ct = ax[1, 0].contourf(x, z, v, num_levels, cmap=cmap)
            ax[1, 0].axis('equal')
            ax[1, 0].axis('off')
            fig.colorbar(ct, ax=ax[1, 0])

            ax[1, 1].axis('equal')
            ax[1, 1].axis('off')

            fig.tight_layout()
            fig.savefig(out_path)
            fig.clf()

        plt.close('all')
    
    def _save_field_2d(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'field_2d', name)

        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        pts = result_dict['grid_pts']
        val = result_dict['grid_pred']
        
        cmap = 'RdYlBu_r'
        num_levels = 40

        B = pts.shape[0]
        for idx in range(B):
            out_path = base_path + '-%06d%s.png' % (base_idx+idx, name)

            fig = plt.figure(dpi=150)
            ax = plt.gca()

            v = val[idx, :, :, 0]
            x = pts[idx, :, :, 0]
            y = -pts[idx, :, :, 1]  # the y axis of maplotlib is the inverse of opencv
            ct = ax.contourf(x, y, v, num_levels, cmap=cmap)
            ax.axis('equal')
            ax.axis('off')

            fig.colorbar(ct)
            fig.tight_layout()
            fig.savefig(out_path)
            fig.clf()

        plt.close('all')

    def _save_surface_3d(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'surface_3d', name)

        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        if 'mesh_pred' not in result_dict:
            self._get_mesh_pred(result_dict)

        for idx, mesh in enumerate(result_dict['mesh_pred']):
            out_path = base_path + '-%06d%s' % (base_idx+idx, name)

            if mesh is None:
                logging.warn('The set value for marching cubes is not among the given volume.')
            else:
                mesh.export(f'{out_path}.ply')
                img = self.vis.render_mesh(mesh)
                cv.imwrite(f'{out_path}.jpg', img)
    
    def _save_surface_2d(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'surface_2d', name)

        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        pts = result_dict['grid_pts']
        val = result_dict['grid_pred']

        B = pts.shape[0]
        for idx in range(B):
            out_path = base_path + '-%06d%s.png' % (base_idx+idx, name)
            if 'INR' in self.cfg_model.name:
                surf = (val[idx] <= 0).astype(np.uint8) * 160 + 30
            else:
                raise NotImplementedError
            plt.figure(dpi=150)
            plt.imshow(surf.transpose(1,0,2), cmap='Greys', vmin=0, vmax=255)
            plt.axis('off')
            plt.savefig(out_path)
            plt.clf()

        plt.close('all')

    def _save_pts_3d(self, pts_batch, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'points_3d', name)

        B = pts_batch.shape[0]
        for idx in range(B):
            out_path = base_path + '-%06d%s.ply' % (base_idx+idx, name)

            pts = pts_batch[idx].detach().cpu().numpy()
            assert pts.shape[-1] == 3
            pc = trimesh.points.PointCloud(pts)
            pc.export(out_path)
        
    def _save_pts_2d(self, pts_batch, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'points_2d', name)

        B = pts_batch.shape[0]
        for idx in range(B):
            out_path = base_path + '-%06d%s.jpg' % (base_idx+idx, name)

            pts = pts_batch[idx].detach().cpu().numpy()
            assert pts.shape[-1] == 2
            img = self.vis.scatter_pts(pts)
            cv.imwrite(out_path, img)
        