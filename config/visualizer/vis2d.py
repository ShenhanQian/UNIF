from yacs.config import CfgNode as CN


cfg = CN()

cfg.VISUALIZER = CN()
cfg.VISUALIZER.name = 'Visualizer2D'
cfg.VISUALIZER.kwargs = CN()
cfg.VISUALIZER.kwargs.resolution = 512
cfg.VISUALIZER.kwargs.surface_value = 0.  # level value of points on the surface
