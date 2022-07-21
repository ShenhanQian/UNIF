from .visualizer import Visualizer2D, Visualizer3D


module_dict = {
    'Visualizer2D': Visualizer2D,
    'Visualizer3D': Visualizer3D,
}


def get_visualizer(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown visualizer: %s' % name)
    else:
        return obj
