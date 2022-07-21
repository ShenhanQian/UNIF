import os
import logging
import time

import numpy as np
import torch
import trimesh
from model.smpl.body_model import BodyModel

from .smpl_based import SmplBased


class CAPE(SmplBased):
    def __init__(
        self, 
        split,
        data_root,
        subdir,
        smpl_root,
        num_betas,
        subject_name,
        cloth_type,
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
        self.num_betas = num_betas
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
        assert subject_name is not None, 'DATASET.kwargs.subject_name is required.'
        betas_npy = os.path.join(data_root, 'minimal_body_shape', subject_name, f'{subject_name}_betas.npy')
        betas = np.load(betas_npy)

        gender_pkl = os.path.join(data_root, 'misc', 'subj_genders.pkl')
        gender = np.load(gender_pkl, allow_pickle=True)[subject_name]

        device='cpu'
        # smpl-1.0
        self.bm_dict = {
            'female': os.path.join(smpl_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'),
            'male': os.path.join(smpl_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        }
        if isinstance(self.bm_dict[gender], str):
            logging.info(msg_prefix + f'Loading SMPL body model from: {self.bm_dict[gender]}.')
            self.bm_dict[gender] = BodyModel(bm_fname=self.bm_dict[gender], num_betas=self.num_betas, device=device)
        
        # parse the skeleton
        parents, children, Jconnect, Bid_table = self._parse_kintree_table(self.bm_dict[gender].kintree_table.numpy(), rm_bones)
        
        # rest pose SMPL
        body_parms = {
            'betas': torch.tensor(betas[None, :num_betas]).float().to(device),
        }
        smpl_data = self.bm_dict[gender](**body_parms)
        Jtr_rest, Jrot_rest = smpl_data.Jtr, smpl_data.Jrot
        smpl_faces = self.bm_dict[gender].f
        smpl_weights = self.bm_dict[gender].weights
        smpl_partId = self._get_smpl_partId(smpl_weights, parents, Bid_table)

        # collect clips
        if clip_name is not None:
            clips_list = [clip_name]
        else:
            clips_list = sorted(os.listdir(os.path.join(data_root, subdir, subject_name)))
            if cloth_type is not None:
                clips_list = [clip_name for clip_name in clips_list if cloth_type in clip_name]
                assert len(clips_list) >= 2
            
            num_for_train = int(0.8 * len(clips_list))
            np.random.seed(0)  # reset the seed to ensure the same order for train and val
            np.random.shuffle(clips_list)  # in-place shuffle
            if split == 'train' or test_interpolation:
                clips_list = clips_list[:num_for_train]
            else:
                clips_list = clips_list[num_for_train:]
        scan_clips = []
        for clip_name in clips_list:
            scans_dir = os.path.join(data_root, subdir, subject_name, clip_name)
            assert os.path.exists(scans_dir), 'Directory not exists: %s' % scans_dir
            scan_frames = [os.path.join(scans_dir, f) for f in sorted(os.listdir(scans_dir))]
            scan_clips.append(scan_frames)
        
        logging.info(msg_prefix + f'Begin loading data...')
        tic = time.time()
        self.items = []
        # process clips
        for idx, scan_frames in enumerate(scan_clips):
            num_frames = len(scan_frames)

            if clip_name is not None:
                if frame_index is not None:
                    indices = range(frame_index, frame_index+1)    
                else:
                    if split == 'train':
                        indices = range(0, int(num_frames*0.8), frame_interval)
                    else:
                        if test_interpolation:
                            indices = range(frame_interval//2, int(num_frames*0.8), frame_interval)
                        else:  # extrapolation
                            indices = range(frame_interval//2, num_frames, frame_interval)
            else:
                indices = range(0, num_frames, frame_interval)

            ## frames
            # indices = indices[::20]
            # indices = indices[::5]
            # indices = indices[:2]
            for i, frame_id in enumerate(indices):
                scan_path = scan_frames[frame_id]

                # ---- posed SMPL ----
                if '.ply' in scan_path:
                    regis_npz = scan_path.replace('raw_scans', 'sequences').replace('.ply', '.npz')
                else:
                    regis_npz = scan_path
                regis_npz = dict(np.load(regis_npz))
                body_parms = {
                    'betas': torch.tensor(betas[None, :num_betas]).float().to(device),
                    'root_orient': torch.tensor(regis_npz['pose'][None, :3]).float().to(device),  # 0
                    'trans': torch.tensor(regis_npz['transl'][None]).float().to(device),
                    'pose_body': torch.tensor(regis_npz['pose'][None, 3:66]).float().to(device),  # 1-21
                    # 'pose_hand': torch.tensor(regis_npz['pose'][None, 66:]).float().to(device),  # for CAPE, using pose_hand cause degradation
                }
                smpl_data = self.bm_dict[gender](**body_parms)
                Jtr, Jrot, Jpose = smpl_data.Jtr, smpl_data.Jrot, smpl_data.Jpose
                smpl_verts = smpl_data.v[0]

                # skeleton-based parsing
                Btr, Brot, Bneigh, Bcond, Blim, Bsec = self._batch_transform_bones(
                    Jtr_rest, Jrot_rest, Jtr, Jrot, Jpose, parents, children, Jconnect, rm_bones, Bid_table)

                # ---- scan ----
                # vertices and normals
                if use_smpl_verts:
                    verts = smpl_verts
                    
                    smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_faces)
                    v_normal = smpl_mesh.vertex_normals  # TODO: check the norm
                else:
                    if '.ply' in scan_path:
                        # raw scan
                        mesh = trimesh.load(scan_path, process=False)
                        verts = torch.tensor(mesh.vertices) / 1000  # millimeter -> meter
                        v_normal = torch.tensor(mesh.vertex_normals)
                        
                        # filter outliers by an enlarged bbox of smpl
                        mask = self._filter_pts(verts, smpl_verts)
                        verts = verts[mask]
                        v_normal = v_normal[mask]
                    else:
                        # SMPL-registered mesh
                        verts = regis_npz['v_posed']  # already in millimeter
                        mesh = trimesh.Trimesh(verts, smpl_faces)
                        # resampled vertices
                        verts = torch.tensor(trimesh.sample.sample_surface(mesh, max(num_pts_surf, num_pts_metric))[0])
                        mesh = trimesh.Trimesh(verts)
                        v_normal = torch.tensor(mesh.vertex_normals)

                    v_normal_norm = v_normal.norm(p=2, dim=-1, keepdim=True)
                    mask_invalid = (v_normal_norm == 0).repeat_interleave(3, dim=-1)
                    v_normal = v_normal / (v_normal_norm + 1e-10)
                    v_normal[mask_invalid] = 0.
                
                # ---- collect ----
                item = {
                    'pts_smpl': smpl_verts,
                    'face_smpl': smpl_faces,
                    'weight_smpl': smpl_weights,
                    'partId_smpl': smpl_partId,
                    'verts': verts,
                    'v_normal': v_normal,
                    'Jrot': Jrot[0],
                    'Jtr': Jtr[0],
                    'Brot': Brot[0],
                    'Btr': Btr[0],
                    'Bneigh': Bneigh[0],
                    'Bcond': Bcond[0],
                    'Blim': Blim[0],
                    'Bsec': Bsec[0],
                    'parents': parents,
                }

                item = self._preprocess(item)
                self.items.append(item)

                logging.info(msg_prefix +
                    f'subject: {subject_name}, '
                    f'clip ({idx+1}/{len(scan_clips)}), '
                    f'frame: {i+1}/{len(indices)}: {scan_path}.')
    
        logging.info(msg_prefix + 'Finish loading in %d s.' % (time.time() - tic))
    
    def _filter_pts(self, verts, smpl_verts):
        ## filter with the bbox of smpl vertices
        smpl_bbox_min = smpl_verts.min(0)[0]
        smpl_bbox_max = smpl_verts.max(0)[0]
        smpl_bbox_center = (smpl_bbox_max + smpl_bbox_min) / 2
        smpl_bbox_size = (smpl_bbox_max - smpl_bbox_min) / 2

        bbox_min = smpl_bbox_center - smpl_bbox_size * self.sigma_global
        bbox_max = smpl_bbox_center + smpl_bbox_size * self.sigma_global
        mask_bbox = (verts > bbox_min).all(1) * (verts < bbox_max).all(1)

        ## remove noisy points of the floor
        # mask = (verts[:, 1] >= -0.562)  # by SCANimate
        # mask = (verts[:, 1] >= -0.590)  # the height of the floor is about -0.600
        mask_floor = (verts[:, 1] >= -0.595)  # the height of the floor is about -0.600

        return mask_floor * mask_bbox
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.pipeline.default import get_cfg_defaults
    from dataset import get_dataset
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/dataset/cape.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    dataset = get_dataset(cfg.DATASET.name)(split='train', **cfg.DATASET.kwargs)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
 
    for batch in dataloader:
        pass
