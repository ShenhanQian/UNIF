import logging
import time

import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F

from pipeline.shape_learner import ShapeLearner
from utils.geometry import gradient
from utils.metrics import IOU, chamfer_and_score


class ShapeLearnerUNIF(ShapeLearner):
    """A generic object for neural network experiments."""

    def forward(self, batch, is_testing=False):
        # prepare data
        pts_surface = batch['pts_surface'].cuda().requires_grad_()
        pts_near = batch['pts_near'].cuda().requires_grad_()
        pts_uniform = batch['pts_uniform'].cuda().requires_grad_()

        priors = {}
        if 'Jtr' in batch:
            priors['Jtr'] = batch['Jtr'].cuda()
        if 'Brot' in batch and 'Btr' in batch:
            priors['Brot'] = batch['Brot'].cuda()
            priors['Btr'] = batch['Btr'].cuda()
        if 'Bneigh' in batch:
            priors['Bneigh'] = batch['Bneigh'].cuda()
        if 'Bcond' in batch:
            priors['Bcond'] = batch['Bcond'].cuda()
        if 'partId' in batch:
            priors['partId'] = batch['partId'].cuda()
        
        if is_testing and self.cfg_exp.TEST.external_query:
            result_dict = {
                'priors': priors,
                'pts_surface': pts_surface.detach(),
                'pts_near': pts_near.detach(),
                'pts_uniform': pts_uniform.detach(),
                'pts_smpl': batch['pts_smpl'],
            }
            return result_dict, {}

        # model forward
        if self.cfg_model.name == 'INR':
            pred_surface = self.model(pts_surface, **priors)
            pred_near = self.model(pts_near, **priors)
            pred_uniform = self.model(pts_uniform, **priors)
        elif self.cfg_model.name == 'UNIF':
            pred_surface, _, pred_surface_parts, pts_surface_parts = self.model(pts_surface, is_testing, return_parts=True, **priors)
            pred_near, _, pred_near_parts, pts_near_parts = self.model(pts_near, is_testing, return_parts=True, **priors)
            pred_uniform, _, pred_uniform_parts, pts_uniform_parts = self.model(pts_uniform, is_testing, return_parts=True, **priors)
        else:
            raise NotImplementedError

        result_dict = {
            'priors': priors,
            'pts_surface': pts_surface.detach(),
            'pts_near': pts_near.detach(),
            'pts_uniform': pts_uniform.detach(),
            'pts_smpl': batch['pts_smpl'],
            'pred_surface': pred_surface.detach(),
            'pred_near': pred_near.detach(),
            'pred_uniform': pred_uniform.detach(),
        }

        # loss computation
        loss_dict = {}
        
        ## reconstruction (surface) loss
        if self.cfg_pipeline.lambda_surface is not None:
            loss_surface = pred_surface.abs().mean()
            loss_dict['loss_surface'] = loss_surface * self.cfg_pipeline.lambda_surface
        
        ## reconstruction (normal) loss
        if self.cfg_pipeline.lambda_normal is not None:
            pred_normal = gradient(pts_surface, pred_surface)
            v_normal = batch['normal_surf'].cuda()
            v_normal_norm = v_normal.norm(p=2, dim=-1)
            mask = v_normal_norm > 0

            loss_normal = ((pred_normal - v_normal).norm(2, dim=-1) * mask).mean()  # mask out those normals with zero norm
            loss_dict['loss_normal'] = loss_normal * self.cfg_pipeline.lambda_normal

        ## bone limit loss and section normal loss
        if self.cfg_model.name == 'UNIF' and self.cfg_pipeline.lambda_lim is not None or self.cfg_pipeline.lambda_sec is not None:
            # pts_lim is padded with nan in the dataset
            pts_lim = batch['Blim'].cuda().nan_to_num().requires_grad_()
            mask_lim = ~batch['Blim'].cuda().isnan()

            pred_lim = self.model.forward_limit(pts_lim, **priors)

            if self.cfg_pipeline.lambda_lim is not None:
                loss_lim = pred_lim[mask_lim[..., [0]]].abs().mean()
                loss_dict['loss_lim'] = loss_lim * self.cfg_pipeline.lambda_lim
            
            if self.cfg_pipeline.lambda_sec is not None:
                gt_sec = batch['Bsec'].cuda()
                pred_sec = gradient(pts_lim, pred_lim)
                mask_lim_normal = ~batch['Bsec'].cuda().isnan()

                # loss_sec = -F.cosine_similarity(
                #     pred_sec[mask_lim_normal].reshape(-1, mask_lim_normal.shape[-1]), 
                #     gt_sec[mask_lim_normal].reshape(-1, mask_lim_normal.shape[-1]), 
                #     dim=-1).mean() + 1
                loss_sec = (pred_sec - gt_sec)[mask_lim_normal].norm(2, dim=-1).mean()
                loss_dict['loss_sec'] = loss_sec * self.cfg_pipeline.lambda_sec
            
        ## Eikonal term
        if self.cfg_pipeline.lambda_unit is not None:
            grad_near = gradient(pts_near, pred_near)
            grad_uniform = gradient(pts_uniform, pred_uniform)
            loss_unit_near = ((grad_near.norm(2, dim=-1) - 1) ** 2).mean()
            loss_unit_uniform = ((grad_uniform.norm(2, dim=-1) - 1) ** 2).mean()

            if self.cfg_model.name == 'UNIF':
                grad_near_parts = gradient(pts_near_parts, pred_near_parts)
                grad_uniform_parts = gradient(pts_uniform_parts, pred_uniform_parts)

                loss_unit_near = loss_unit_near + ((grad_near_parts.norm(2, dim=-1) - 1) ** 2).mean()
                loss_unit_uniform = loss_unit_uniform + ((grad_uniform_parts.norm(2, dim=-1) - 1) ** 2).mean()

            loss_unit = loss_unit_near + loss_unit_uniform
            loss_dict['loss_unit'] = loss_unit * self.cfg_pipeline.lambda_unit

        ## perimeter term
        if self.cfg_pipeline.lambda_perim is not None:

            def chain_grad_sigmoid(s, grad_s, k=10):
                """ Given the value of s(x) and its gradient grad_s(x), compute 
                    the chained gradient of sigmoid(k * s(x)).
                """
                exp_neg = torch.exp(-s * k)
                return k * (1 + exp_neg)**(-2) * exp_neg * grad_s

            grad_density_near = chain_grad_sigmoid(pred_near, grad_near)
            grad_density_uniform = chain_grad_sigmoid(pred_uniform, grad_uniform)

            # only account points near the predicted surface for a balanced perimeter energy
            # mask_nearby = (pred_uniform.abs() < self.cfg_dataset.kwargs.sigma_local).repeat_interleave(grad_density_uniform.shape[-1], dim=-1)
            # grad_density_uniform[~mask_nearby] *= 0.

            loss_perim_near = (grad_density_near.norm(2, dim=-1)**2).mean()
            loss_perim_uniform = (grad_density_uniform.norm(2, dim=-1)**2).mean()
            
            if self.cfg_model.name == 'UNIF':
                grad_density_near_parts = chain_grad_sigmoid(pred_near_parts, grad_near_parts)
                grad_density_uniform_parts = chain_grad_sigmoid(pred_uniform_parts, grad_uniform_parts)

                loss_perim_near = loss_perim_near + (grad_density_near_parts.norm(2, dim=-1) ** 2).mean()
                loss_perim_uniform = loss_perim_uniform + (grad_density_uniform_parts.norm(2, dim=-1) ** 2).mean()

            loss_perim = loss_perim_near + loss_perim_uniform
            loss_dict['loss_perim'] = loss_perim * self.cfg_pipeline.lambda_perim
        
        ## offsurface regularization
        if self.cfg_pipeline.lambda_offsurf is not None:
            loss_offsurf_near = torch.exp(-100.0*torch.abs(pred_near)).mean()
            loss_offsurf_uniform = torch.exp(-100.0*torch.abs(pred_uniform)).mean()
            loss_offsurf = loss_offsurf_near + loss_offsurf_uniform
            loss_dict['loss_offsurf'] = loss_offsurf * self.cfg_pipeline.lambda_offsurf
        
        ## SDF maximization for regularization
        if self.cfg_pipeline.lambda_maximize is not None:
            loss_maximize = (-pred_near_parts).mean() + (-pred_uniform_parts).mean()
            loss_dict['loss_maximize'] = loss_maximize * self.cfg_pipeline.lambda_maximize
        
        ## total loss
        loss_dict['loss'] = sum(map(lambda x: x[1], loss_dict.items()))

        # value inspection
        if 'rigid_alpha' in self.model.state_dict():
            rigid_alpha = F.elu(self.model.state_dict()['rigid_alpha'] - 1) + 1
            loss_dict['rigid_alpha'] = rigid_alpha.unique().mean().detach()
        
        if 'rigid_beta' in self.model.state_dict():
            rigid_beta = self.model.state_dict()['rigid_beta']
            loss_dict['rigid_beta'] = rigid_beta.unique().mean().detach()

        # metrics computation
        if is_testing:
            with torch.no_grad():
                if 'label_near' in batch and 'label_uniform' in batch:
                    label_near = batch['label_near'].cuda()
                    label_uniform = batch['label_uniform'].cuda()

                    loss_dict['iou_near'] = IOU(pred_near <= 0, label_near)
                    loss_dict['iou_uniform'] = IOU(pred_uniform <= 0, label_uniform)
                
                if result_dict['pts_uniform'].shape[-1] == 3:
                    self._get_mesh_pred(result_dict)
                    pt2surf = torch.empty(len(result_dict['mesh_pred']))
                    CD_sym = torch.empty(len(result_dict['mesh_pred']))
                    recall = torch.empty(len(result_dict['mesh_pred']))
                    F_score = torch.empty(len(result_dict['mesh_pred']))

                    for i, mesh_pred in enumerate(result_dict['mesh_pred']):
                        pt2surf[i], CD_sym[i], recall[i], F_score[i] = chamfer_and_score(batch['pts_metric'][i].numpy(), mesh_pred)
                    loss_dict['pt2surf'] = pt2surf.mean()
                    loss_dict['CD_sym'] = CD_sym.mean()
                    loss_dict['recall'] = recall.mean()
                    loss_dict['F_score'] = F_score.mean()
            
        return result_dict, loss_dict
    
    def _save_results(self, mode, result_dict, epoch, step, num_epochs, steps_per_epoch):
        torch.cuda.synchronize()
        tic_save = time.time()

        # shrink the batch when saving results to save time
        if mode=='TRAIN' and self.cfg_exp.TRAIN.num_save_per_batch is not None:
            self._shrink_tensor_dict_(result_dict, self.cfg_exp.TRAIN.num_save_per_batch)

        if self.cfg_model.kwargs.d_in == 3:
            if self.cfg_exp.save_coord_system:
                self._save_coord_system(result_dict, mode, epoch, step, name=None)
            
            if self.cfg_exp.save_fig:
                self._save_fig(result_dict, mode, epoch, step, name=None)

            if self.cfg_exp.save_field:
                self._save_field_3d(result_dict, mode, epoch, step, name=None)

            if self.cfg_exp.save_surface:
                self._save_surface_3d(result_dict, mode, epoch, step, name=None)
                if self.cfg_model.name == 'UNIF':
                    self._save_surface_3d_colored(result_dict, mode, epoch, step, name=None)
            
            if self.cfg_exp.save_sphere_tracing:
                self._save_sphere_tracing_3d(result_dict, mode, epoch, step, name=None)

            if self.cfg_exp.save_points and not self.cfg_exp.TEST.external_query:
                self._save_pts_3d(result_dict['pts_surface'], mode, epoch, step, name='surface')
                self._save_pts_3d(result_dict['pts_near'], mode, epoch, step, name='near')
                self._save_pts_3d(result_dict['pts_uniform'], mode, epoch, step, name='uniform')
                self._save_pts_3d(result_dict['pts_smpl'], mode, epoch, step, name='smpl')

        elif self.cfg_model.kwargs.d_in ==2:
            if self.cfg_exp.save_field:
                self._save_field_2d(result_dict, mode, epoch, step, name=None)

            if self.cfg_exp.save_surface:
                self._save_surface_2d(result_dict, mode, epoch, step, name='surface')

            if self.cfg_exp.save_points and not self.cfg_exp.TEST.external_query:
                self._save_pts_2d(result_dict['pts_surface'], mode, epoch, step, name='surface')
                self._save_pts_2d(result_dict['pts_near'], mode, epoch, step, name='near')
                self._save_pts_2d(result_dict['pts_uniform'], mode, epoch, step, name='uniform')
        else:
            raise RuntimeError(f'{self.cfg_model.kwargs.d_in}-dimension points not supported.')

        self._write_logs_save(mode, epoch, step, tic_save, num_epochs, steps_per_epoch)
    
    def _color_mesh_by_part(self, result_dict):
        meshes = result_dict['mesh_pred']
        priors = result_dict['priors']

        result_dict['mesh_colored'] = self.vis.color_mesh_by_part(self.model, meshes, priors)
    
    def _save_sphere_tracing_3d(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'sphere_tracing_3d', name)

        for idx in range(result_dict['priors']['Brot'].shape[0]):
            img = self.vis.render_sdf(self.model, self._slice_tensor_dict(result_dict['priors'], slice(idx, idx+1)))

            out_path = base_path + '-%06d%s' % (base_idx+idx, name)
            cv.imwrite(f'{out_path}.jpg', img)
    
    def _save_surface_3d_colored(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'surface_3d_colored', name)

        if 'grid_pts' not in result_dict or 'grid_pred' not in result_dict:
            self._get_grid_pred(result_dict)
        if 'mesh_pred' not in result_dict:
            self._get_mesh_pred(result_dict)
        if 'mesh_colored' not in result_dict:
            self._color_mesh_by_part(result_dict)

        for idx, mesh in enumerate(result_dict['mesh_colored']):
            out_path = base_path + '-%06d%s' % (base_idx+idx, name)

            if mesh is None:
                logging.warn('The set value for marching cubes is not among the given volume.')
            else:
                mesh.export(f'{out_path}.ply')
                img = self.vis.render_mesh(mesh)
                cv.imwrite(f'{out_path}.jpg', img)
    
    def _save_coord_system(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'coord_system', name)

        Jtr = result_dict['priors']['Jtr'].cpu()
        Btr = result_dict['priors']['Btr'].cpu()
        Brot = result_dict['priors']['Brot'].cpu()

        B = Brot.shape[0]
        for idx in range(B):
            mesh = self.vis.get_mesh_of_coord_system(Jtr[idx], Btr[idx], Brot[idx])

            out_path = base_path + '-%06d%s' % (base_idx+idx, name)
            mesh.export(f'{out_path}.obj')

            img = self.vis.render_mesh(mesh, pointlight=False)
            cv.imwrite(f'{out_path}.jpg', img)
    
    def _save_fig(self, result_dict, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'fig', name)
        out_path = base_path + '-%06d%s' % (base_idx, name)
                
        mpl.use('Agg')
        # mpl.rcParams['agg.path.chunksize'] = 10000
        kwargs = {}

        if 'rigid_alpha' in self.model.state_dict():
            # kwargs = {'figsize': [6.4*2.5, 4.8*10], 'dpi': 100}
            kwargs = {}
            fig = plt.figure(**kwargs)
            ax = plt.gca()

            rigid_alpha = (F.elu(self.model.state_dict()['rigid_alpha'] - 1) + 1).cpu().numpy()
            im = ax.matshow(rigid_alpha)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='5%')
            plt.colorbar(im, cax=cax)

            fig.tight_layout()
            fig.savefig(f'{out_path}-rigid_alpha.png')
            plt.close()
        
        if 'rigid_beta' in self.model.state_dict():
            # kwargs = {'figsize': [6.4*2.5, 4.8*10], 'dpi': 100}
            kwargs = {}
            fig = plt.figure(**kwargs)
            ax = plt.gca()

            rigid_beta = self.model.state_dict()['rigid_beta'].cpu().numpy()
            im = ax.matshow(rigid_beta)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='5%')
            plt.colorbar(im, cax=cax)

            fig.tight_layout()
            fig.savefig(f'{out_path}-rigid_beta.png')
            plt.close()
    