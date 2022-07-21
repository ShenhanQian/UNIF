from yacs.config import CfgNode as CN


cfg = CN()

cfg.PIPELINE = CN()
cfg.PIPELINE.parent_cfg = 'config/pipeline/shape_learner_unif.py'
cfg.PIPELINE.lambda_surface = 1.
cfg.PIPELINE.lambda_normal = 0.01
cfg.PIPELINE.lambda_lim = 1.
cfg.PIPELINE.lambda_sec = 0.01
cfg.PIPELINE.lambda_unit = 0.1
cfg.PIPELINE.lambda_perim = 0.001

cfg.MODEL = CN()
cfg.MODEL.parent_cfg = 'config/model/unif.py'
cfg.MODEL.kwargs = CN()
cfg.MODEL.kwargs.num_parts = 20
cfg.MODEL.kwargs.dims = [64, 64, 64, 64, 64, 64, 64, 64]
cfg.MODEL.kwargs.radius_init = 0.01

cfg.DATASET = CN()
cfg.DATASET.parent_cfg = 'config/dataset/clothseq.py'
cfg.DATASET.kwargs = CN()
cfg.DATASET.kwargs.clip_name = 'JacketPants'
cfg.DATASET.kwargs.frame_index = None
cfg.DATASET.kwargs.num_pts_surf = 5000
cfg.DATASET.kwargs.rm_bones = [10, 11, 22, 23]

cfg.VISUALIZER = CN()
cfg.VISUALIZER.parent_cfg = 'config/visualizer/vis3d.py'
cfg.VISUALIZER.kwargs = CN()
cfg.VISUALIZER.kwargs.cam_position = [0., 0.25, 2.]

cfg.EXP = CN()
cfg.EXP.interval_log_step = 10
cfg.EXP.interval_log_epoch = 1
cfg.EXP.interval_result_step = 1000
cfg.EXP.interval_result_epoch = 50
cfg.EXP.interval_ckpt_epoch = 100

cfg.EXP.TRAIN = CN()
cfg.EXP.TRAIN.num_epochs = 5000
cfg.EXP.TRAIN.batch_size = 4
cfg.EXP.TRAIN.num_save_per_batch = 1

cfg.EXP.TEST = CN()
cfg.EXP.TEST.batch_size = 4

cfg.SCHEDULER = CN()
cfg.SCHEDULER.type = 'MultiStepLR'
cfg.SCHEDULER.interval = 1000
cfg.SCHEDULER.gamma = 0.3
cfg.SCHEDULER.num_steps = 3
cfg.SCHEDULER.warmup_type = None
