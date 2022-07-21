from yacs.config import CfgNode as CN


cfg = CN()

cfg.VISUALIZER = CN()
cfg.VISUALIZER.name = 'Visualizer3D'
cfg.VISUALIZER.kwargs = CN()
cfg.VISUALIZER.kwargs.resolution_mc = 256
cfg.VISUALIZER.kwargs.resolution_render = 1920
cfg.VISUALIZER.kwargs.mc_value = 0.  # level value for marching cubes
cfg.VISUALIZER.kwargs.gradient_direction = 'descent'  # gradient direction towards object center
cfg.VISUALIZER.kwargs.uniform_grid = False
cfg.VISUALIZER.kwargs.connected = False
cfg.VISUALIZER.kwargs.cam_position = [0., 0., 2.]
