from yacs.config import CfgNode as CN


cfg = CN()

cfg.MODEL = CN()

cfg.MODEL.name = 'UNIF'
cfg.MODEL.kwargs = CN()
cfg.MODEL.kwargs.num_parts = None
cfg.MODEL.kwargs.d_in = 3
cfg.MODEL.kwargs.dims = [64, 64, 64, 64, 64, 64, 64, 64]
cfg.MODEL.kwargs.skip_in = [4]
cfg.MODEL.kwargs.d_cond = 4
cfg.MODEL.kwargs.multires = 0
cfg.MODEL.kwargs.geometric_init = True
cfg.MODEL.kwargs.radius_init = 0.01
cfg.MODEL.kwargs.beta = 100
cfg.MODEL.kwargs.weight_norm = True
cfg.MODEL.kwargs.blend_alpha = 2.
