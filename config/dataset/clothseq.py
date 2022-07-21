from yacs.config import CfgNode as CN


cfg = CN()

cfg.DATASET = CN()

cfg.DATASET.name = 'ClothSeq'
cfg.DATASET.kwargs = CN()
cfg.DATASET.kwargs.data_root = 'data/ClothSeq'
cfg.DATASET.kwargs.smpl_root = 'data/smpl/models/'
cfg.DATASET.kwargs.num_betas = 10
cfg.DATASET.kwargs.use_smpl_verts = False
cfg.DATASET.kwargs.clip_name = 'JacketPants'
cfg.DATASET.kwargs.frame_index = None
cfg.DATASET.kwargs.frame_interval = 10
cfg.DATASET.kwargs.subset_for_train = None
cfg.DATASET.kwargs.test_interpolation = False
cfg.DATASET.kwargs.sigma_knn = None
cfg.DATASET.kwargs.bbox_normalize = False
cfg.DATASET.kwargs.num_pts_surf = 5000
cfg.DATASET.kwargs.num_pts_near_per_sample = 1
cfg.DATASET.kwargs.num_pts_uniform = 5000
cfg.DATASET.kwargs.num_pts_metric = 100000
cfg.DATASET.kwargs.sigma_local = 0.1
cfg.DATASET.kwargs.sample_on_sphere = False
cfg.DATASET.kwargs.sigma_global = 1.5
cfg.DATASET.kwargs.label_for_train = False
cfg.DATASET.kwargs.partId_from_smpl = False
cfg.DATASET.kwargs.rm_bones = []