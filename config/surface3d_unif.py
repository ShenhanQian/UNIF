from yacs.config import CfgNode as CN


cfg = CN()

cfg.PIPELINE = CN()
cfg.PIPELINE.parent_cfg = 'config/pipeline/shape_learner_unif.py'
cfg.PIPELINE.lambda_surface = 1.
cfg.PIPELINE.lambda_unit = 0.1
cfg.PIPELINE.lambda_perim = 0.01

cfg.MODEL = CN()
cfg.MODEL.parent_cfg = 'config/model/inr.py'
cfg.MODEL.kwargs = CN()
cfg.MODEL.kwargs.multires = 6

cfg.DATASET = CN()
cfg.DATASET.parent_cfg = 'config/dataset/surface3d.py'
cfg.DATASET.kwargs = CN()

cfg.VISUALIZER = CN()
cfg.VISUALIZER.parent_cfg = 'config/visualizer/vis3d.py'

cfg.EXP = CN()
cfg.EXP.interval_log_step = 100
cfg.EXP.interval_log_epoch = 100
cfg.EXP.interval_result_step = 1000
cfg.EXP.interval_result_epoch = 1000
cfg.EXP.interval_ckpt_epoch = 1000

cfg.EXP.TRAIN = CN()
cfg.EXP.TRAIN.num_epochs = 100000


cfg.SCHEDULER = CN()
cfg.SCHEDULER.type = 'MultiStepLR'
cfg.SCHEDULER.interval = 5000
cfg.SCHEDULER.gamma = 0.3
cfg.SCHEDULER.num_steps = 3
