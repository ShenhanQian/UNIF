import os
from copy import deepcopy
import logging

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset, DataLoader


class Surface3D(Dataset):
    def __init__(
        self, 
        split,
        obj_path,
        subset_for_train=None,
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

        assert split in ['train', 'val'], f'Unknown dataset split: {split}'
        msg_prefix = (f'[DATASET]({split}) ')
        logging.info(msg_prefix + f'Object path: {obj_path}.')
        
        mesh = trimesh.load(obj_path)
        self.items = [
            {
                'verts': torch.tensor(mesh.vertices).float(),
                'faces': torch.tensor(mesh.faces),
            }
        ]

        for item in self.items:
            item = self._preprocess(item)
    
    def _preprocess(self, item):
        if self.subset_for_train is not None and self.split == 'train':
            self._subsample_pts(item)
        if self.bbox_normalize:
            item = self._bbox_normalize(item)
        if self.sigma_knn is not None:
            item = self._get_sigma_knn(item)
        return item
    
    def _subsample_pts(self, item):
        pts = item['verts']
        assert self.subset_for_train <= pts.shape[0], \
            f'DATA.subset_for_train ({self.subset_for_train}) should be no more than ' \
            f'the total number of points ({pts.shape[0]}).'

        indices = torch.tensor(np.random.choice(pts.shape[0], self.subset_for_train, False))
        
        item['verts'] = item['verts'][indices]
        return item
    
    def _bbox_normalize(self, item):
        pts = item['verts']
        bbox_min, _ = pts.min(0, keepdim=True)
        bbox_max, _ = pts.max(0, keepdim=True)
        center = (bbox_max + bbox_min) / 2
        scale = (bbox_max - bbox_min).norm(p=2, dim=1) / 2
        item['verts'] = (pts - center) / scale
        return item
    
    def _get_sigma_knn(self, item):
        sigma_set = []
        ptree = cKDTree(item['verts'])

        for p in np.array_split(item['verts'], 100, axis=0):
            d = ptree.query(p, self.sigma_knn + 1)
            sigma_set.append(d[0][:, -1].astype(np.float32))

        item['sigma_knn'] = np.concatenate(sigma_set)[..., None]
        return item

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = deepcopy(self.items[idx])
        item = self._runtime_preprocess(item)

        return item

    def _runtime_preprocess(self, item):
        item = self._spatial_sampling(item)

        if self.label_for_train:
            item = self._get_label(item)
        
        item = self._clean_item(item)
        return item
    
    def _spatial_sampling(self, item):
        # surface
        assert item['verts'].shape[0] >= self.num_pts_surf, 'Too few points: %d' % item['verts'].shape[0]
        indices = torch.tensor(np.random.choice(item['verts'].shape[0], self.num_pts_surf, False))
        pts_surface = item['verts'][indices, ...].float()
        normal_surf = item['v_normal'][indices, ...].float()

        # near
        if 'sigma_knn' in item:
            sigma_local = item['sigma_knn'][indices, ...]
        else:
            sigma_local = self.sigma_local

        num_pts, dim = pts_surface.shape
        offset = torch.randn(num_pts, self.num_pts_near_per_sample, dim)
        if self.sample_on_sphere:
            offset = offset / offset.norm(p=2, dim=-1, keepdim=True)
            sigma_local = (torch.rand(offset.shape[0]) * self.sigma_local)[:, None, None]
        else:
            sigma_local = self.sigma_local
        offset = offset * sigma_local
        pts_near = (pts_surface.unsqueeze(1) + offset).reshape(-1, dim)

        # uniform
        min_xyz = pts_surface.min(0, keepdim=True)[0]
        max_xyz = pts_surface.max(0, keepdim=True)[0]
        center_xyz = (max_xyz + min_xyz) / 2
        size_xyz = (max_xyz - min_xyz) * self.sigma_global
        pts_uniform = torch.rand(self.num_pts_uniform, dim, device=pts_surface.device)
        pts_uniform = (pts_uniform - 0.5) * size_xyz + center_xyz

        item['pts_surface'] = pts_surface
        item['normal_surf'] = normal_surf
        item['pts_near'] = pts_near
        item['num_pts_near_per_sample'] = self.num_pts_near_per_sample
        item['pts_uniform'] = pts_uniform

        if self.split == 'val':
            # sample from the verts when no face is available
            metric_indices = np.random.choice(item['verts'].shape[0], self.num_pts_metric, True)
            pts_metric = item['verts'][metric_indices, ...]
        
            item['pts_metric'] = pts_metric

        return item
    
    def _get_label(self, item):
        assert 'faces' in item
        mesh = trimesh.Trimesh(item['verts'], item['faces'])

        rt = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        label_near = rt.contains_points(item['pts_near'])
        label_uniform = rt.contains_points(item['pts_uniform'])

        item['label_near'] = torch.tensor(label_near).unsqueeze(-1)
        item['label_uniform'] = torch.tensor(label_uniform).unsqueeze(-1)
        return item
    
    def _clean_item(self, item):
        item.pop('verts')
        item.pop('v_normal')
        return item


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.pipeline.default import get_cfg_defaults
    from dataset import get_dataset
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/dataset/surface3d.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    dataset = get_dataset(cfg.DATASET.name)(split='train', **cfg.DATASET.kwargs)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
 
    for batch in dataloader:
        pass
