from yacs.config import CfgNode as CN


cfg = CN()

cfg.MODEL = CN()

cfg.MODEL.name = 'INR'
cfg.MODEL.kwargs = CN()
cfg.MODEL.kwargs.d_in = 3
cfg.MODEL.kwargs.dims = [512, 512, 512, 512, 512, 512, 512, 512]
cfg.MODEL.kwargs.skip_in = [4]
cfg.MODEL.kwargs.multires = 6
cfg.MODEL.kwargs.geometric_init = True
cfg.MODEL.kwargs.radius_init = 0.01
cfg.MODEL.kwargs.beta = 100
cfg.MODEL.kwargs.weight_norm = True
