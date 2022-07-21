from yacs.config import CfgNode as CN


cfg = CN()

cfg.PIPELINE = CN(new_allowed=True)
cfg.PIPELINE.name = 'Learner'

cfg.MODEL = CN(new_allowed=True)
cfg.MODEL.strict_load = True
cfg.MODEL.pretrain_map_list = ()
cfg.MODEL.pretrain_blacklist = []
cfg.MODEL.pretrain_whitelist = []

cfg.DATASET = CN(new_allowed=True)

cfg.VISUALIZER = CN(new_allowed=True)

cfg.EXP = CN()
cfg.EXP.tag = ''
cfg.EXP.base_dir = 'output'
cfg.EXP.interval_log_step = 100
cfg.EXP.interval_log_epoch = 1
cfg.EXP.interval_result_step = 1000
cfg.EXP.interval_result_epoch = 1
cfg.EXP.interval_ckpt_epoch = 1
cfg.EXP.save_fig = True
cfg.EXP.save_img = True
cfg.EXP.save_field = True
cfg.EXP.save_surface = True
cfg.EXP.save_sphere_tracing = False
cfg.EXP.save_points = True
cfg.EXP.save_coord_system = True
cfg.EXP.num_workers = 8
cfg.EXP.persistent_workers = True
cfg.EXP.validate = True
cfg.EXP.test_only = False
cfg.EXP.checkpoint = None
cfg.EXP.checkpoint_map = None
cfg.EXP.resume_from = None

cfg.EXP.TRAIN = CN()
cfg.EXP.TRAIN.num_epochs = 15000
cfg.EXP.TRAIN.batch_size = 4
cfg.EXP.TRAIN.num_save_per_batch = None

cfg.EXP.TEST = CN()
cfg.EXP.TEST.batch_size = 4
cfg.EXP.TEST.shuffle = False
cfg.EXP.TEST.save_all_results = True
cfg.EXP.TEST.external_query = False

cfg.OPTIMIZER = CN()
cfg.OPTIMIZER.type = 'adam'
cfg.OPTIMIZER.update_blacklist = []
cfg.OPTIMIZER.update_whitelist = []
cfg.OPTIMIZER.lr = 1e-3

cfg.SCHEDULER = CN()
cfg.SCHEDULER.type = None
cfg.SCHEDULER.interval = None
cfg.SCHEDULER.gamma = None
cfg.SCHEDULER.num_steps = None
cfg.SCHEDULER.warmup_type = None


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()