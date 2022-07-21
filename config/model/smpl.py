from yacs.config import CfgNode as CN


cfg = CN()

cfg.MODEL = CN()

cfg.MODEL.smpl_root = 'data/smpl/models/'
cfg.MODEL.num_beta = 10