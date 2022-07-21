from yacs.config import CfgNode as CN


cfg = CN()

cfg.DATASET = CN()

cfg.DATASET.name = 'CAPE_bodyshape'
cfg.DATASET.kwargs = CN()
cfg.DATASET.kwargs.data_root = 'data/cape_release'
cfg.DATASET.kwargs.subject_name = '00032'