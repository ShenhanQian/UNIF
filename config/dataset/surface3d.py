from yacs.config import CfgNode as CN


cfg = CN()

cfg.DATASET = CN()

cfg.DATASET.name = 'Surface3D'
cfg.DATASET.kwargs = CN()
cfg.DATASET.kwargs.obj_path = None
cfg.DATASET.kwargs.subset_for_train = None
cfg.DATASET.kwargs.sigma_knn = None
cfg.DATASET.kwargs.bbox_normalize = True
cfg.DATASET.kwargs.num_pts_surf = 5000
cfg.DATASET.kwargs.num_pts_near_per_sample = 1
cfg.DATASET.kwargs.num_pts_uniform = 5000
cfg.DATASET.kwargs.num_pts_metric = 100000
cfg.DATASET.kwargs.sigma_local = 0.1
cfg.DATASET.kwargs.sample_on_sphere = False
cfg.DATASET.kwargs.sigma_global = 1.5
cfg.DATASET.kwargs.label_for_train = False
