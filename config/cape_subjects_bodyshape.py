from yacs.config import CfgNode as CN


cfg = CN()

cfg.PIPELINE = CN()
cfg.PIPELINE.parent_cfg = 'config/pipeline/shape_learner_smpl.py'

cfg.MODEL = CN()
cfg.MODEL.parent_cfg = 'config/model/smpl.py'

cfg.DATASET = CN()
cfg.DATASET.parent_cfg = 'config/dataset/cape_bodyshape.py'
cfg.DATASET.kwargs = CN()
cfg.DATASET.kwargs.subject_name = "00032"

cfg.VISUALIZER = CN()
cfg.VISUALIZER.parent_cfg = 'config/visualizer/vis3d.py'
cfg.VISUALIZER.kwargs = CN()
cfg.VISUALIZER.kwargs.cam_position = [0., 0.4, 2.]

cfg.EXP = CN()
cfg.EXP.interval_log_step = 10
cfg.EXP.interval_log_epoch = 100
cfg.EXP.interval_result_step = 500
cfg.EXP.interval_result_epoch = 1000
cfg.EXP.interval_ckpt_epoch = 1000

cfg.EXP.TRAIN = CN()
cfg.EXP.TRAIN.num_epochs = 10000
cfg.EXP.TRAIN.batch_size = 4
cfg.EXP.TRAIN.num_save_per_batch = 1

cfg.EXP.TEST = CN()
cfg.EXP.TEST.batch_size = 4

cfg.SCHEDULER = CN()
cfg.SCHEDULER.type = 'MultiStepLR'
cfg.SCHEDULER.interval = 1000
cfg.SCHEDULER.gamma = 0.3
cfg.SCHEDULER.num_steps = 3
