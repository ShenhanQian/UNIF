import os
import logging
import time

import numpy as np
import torch
import trimesh
from model.smpl.body_model import BodyModel

from .smpl_based import SmplBased


class ClothSeq(SmplBased):
    def __init__(
        self, 
        split,
        data_root,
        smpl_root,
        num_betas,
        clip_name,
        frame_index=None,
        frame_interval=10,
        use_smpl_verts=False,
        subset_for_train=None,
        test_interpolation=False,
        sigma_knn = None,
        bbox_normalize = False,
        num_pts_surf=5000,
        num_pts_near_per_sample=1,
        num_pts_uniform=5000,
        num_pts_metric=100000,
        sigma_local=0.1,
        sample_on_sphere=False,
        sigma_global=1.5,
        label_for_train=False,
        partId_from_smpl=False,
        rm_bones=[],
        ): 

        self.split = split
        self.subset_for_train = subset_for_train
        self.sigma_knn = sigma_knn
        self.bbox_normalize = bbox_normalize
        self.num_pts_surf = num_pts_surf
        self.num_pts_near_per_sample = num_pts_near_per_sample
        self.num_pts_uniform = num_pts_uniform
        self.num_pts_metric = num_pts_metric
        self.sigma_local = sigma_local
        self.sample_on_sphere = sample_on_sphere
        self.sigma_global = sigma_global if isinstance(sigma_global, int) or isinstance(sigma_global, float) else torch.tensor(sigma_global)
        self.label_for_train = label_for_train
        self.partId_from_smpl = partId_from_smpl

        assert split in ['train', 'val'], f'Unknown dataset split: {split}'
        msg_prefix = (f'[DATASET]({split}) ')
        logging.info(msg_prefix + f'Dataset root: {data_root}.')

        # init SMPL body model
        gender = 'male' if clip_name == 'ShrugsPants' else 'female'
        device='cpu'
        if gender == 'female':
            bm_fname = os.path.join(smpl_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'male':
            bm_fname = os.path.join(smpl_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        else:
            raise ValueError(f'Unknown gender: {gender}.')
        logging.info(msg_prefix + f'Loading SMPL body model from: {bm_fname}.')
        self.bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, device=device)

        # parse the skeleton
        parents, children, Jconnect, Bid_table = self._parse_kintree_table(self.bm.kintree_table.numpy(), rm_bones)

        # collect frames
        assert clip_name is not None, 'DATASET.kwargs.clip_name is required.'
        clip_dir = os.path.join(data_root, clip_name)
        smpl_regis_npz = os.path.join(clip_dir, 'smpl_registration.npz')
        smpl_param_frames = np.load(smpl_regis_npz)
        num_frames = len(smpl_param_frames['frames'])

        if frame_index is not None:
            indices = range(frame_index, frame_index+1, frame_interval)    
        else:
            if split == 'train':
                indices = range(0, int(num_frames*0.8), frame_interval)
            else:
                if test_interpolation:
                    indices = range(frame_interval//2, int(num_frames*0.8), frame_interval)
                else:  # extrapolation
                    indices = range(int(num_frames*0.8), num_frames, frame_interval)
        frames = smpl_param_frames['frames'][indices]     

        # rest pose SMPL
        body_parms = {
            'betas': torch.tensor(smpl_param_frames['betas'][indices, :num_betas]).float().to(device),
        }
        clip_data = self.bm(**body_parms)
        Jtr_rest, Jrot_rest = clip_data.Jtr, clip_data.Jrot  # (N, J, 3, (3))
        smpl_faces = self.bm.f
        smpl_weights = self.bm.weights
        smpl_partId = self._get_smpl_partId(smpl_weights, parents, Bid_table)

        # posed SMPL
        body_parms = {
            'betas': torch.tensor(smpl_param_frames['betas'][indices, :num_betas]).float().to(device),
            'root_orient': torch.tensor(smpl_param_frames['thetas'][indices, :3]).float().to(device),  # 0
            'trans': torch.tensor(smpl_param_frames['trans'][indices]).float().to(device),
            'pose_body': torch.tensor(smpl_param_frames['thetas'][indices, 3:66]).float().to(device),  # 1-21
            'pose_hand': torch.tensor(smpl_param_frames['thetas'][indices, 66:]).float().to(device),  # 22-23
        }
        clip_data = self.bm(**body_parms)
        Jtr, Jrot, Jpose = clip_data.Jtr, clip_data.Jrot, clip_data.Jpose  # (N, J, 3, (3))
        smpl_verts = clip_data.v
        
        
        Btr, Brot, Bneigh, Bcond, Blim, Bsec = self._batch_transform_bones(
            Jtr_rest, Jrot_rest, Jtr, Jrot, Jpose, parents, children, Jconnect, rm_bones, Bid_table)

        logging.info(msg_prefix + f'Begin loading data...')
        tic = time.time()
        self.items = []
        # indices = indices[::20]
        # indices = indices[::5]
        for i in range(len(indices)):
            if use_smpl_verts:
                verts = smpl_verts[i]

                smpl_mesh = trimesh.Trimesh(verts.cpu(), smpl_faces)
                v_normal = torch.tensor(smpl_mesh.vertex_normals)
            else:
                # raw_scan = os.path.join(data_root, clip_name, 'scans', '%s.obj' % frames[i])  # slow
                raw_scan = os.path.join(data_root, clip_name, 'scans-ply', '%s.ply' % frames[i])  # 2x faster
                mesh = trimesh.load_mesh(raw_scan)
                verts = torch.tensor(mesh.vertices)
                v_normal = torch.tensor(mesh.vertex_normals)

                v_normal_norm = np.linalg.norm(v_normal, axis=-1, keepdims=True)
                mask_invalid = (v_normal_norm == 0).repeat(3, axis=-1)
                v_normal = v_normal / (v_normal_norm + 1e-10)
                v_normal[mask_invalid] = 0.

            item = {
                'pts_smpl': smpl_verts[i],
                'weights_smpl': smpl_weights,
                'partId_smpl': smpl_partId,
                'verts': verts,
                'v_normal': v_normal,
                'Jrot': Jrot[i],
                'Jtr': Jtr[i],
                'Brot': Brot[i],
                'Btr': Btr[i],
                'Bneigh': Bneigh[i],
                'Bcond': Bcond[i],
                'Blim': Blim[i],
                'Bsec': Bsec[i],
                'parents': parents,
            }
            item = self._preprocess(item)
            self.items.append(item)

            logging.info(msg_prefix +
                f'clip: {clip_dir}, '
                f'frame: {frames[i]}, '
                f'({i}/{len(indices)}).')
        logging.info(msg_prefix + 'Finish loading in %d s.' % (time.time() - tic))
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.pipeline.default import get_cfg_defaults
    from dataset import get_dataset
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/dataset/clothseq.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    dataset = get_dataset(cfg.DATASET.name)(split='train', **cfg.DATASET.kwargs)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
 
    for batch in dataloader:
        pass
