from yacs.config import CfgNode as CN


cfg = CN()

cfg.PIPELINE = CN()
cfg.PIPELINE.name = 'ShapeLearnerUNIF'
cfg.PIPELINE.lambda_surface = None
cfg.PIPELINE.lambda_normal = None
cfg.PIPELINE.lambda_lim = None
cfg.PIPELINE.lambda_sec = None
cfg.PIPELINE.lambda_unit = None
cfg.PIPELINE.lambda_perim = None
cfg.PIPELINE.lambda_offsurf = None
cfg.PIPELINE.lambda_maximize = None
