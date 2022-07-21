from .surface3d import Surface3D
from .clothseq import ClothSeq
from .cape import CAPE
from .cape_bodyshape import CAPE_bodyshape


module_dict = {
    'Surface3D': Surface3D,
    'ClothSeq': ClothSeq,
    'CAPE': CAPE,
    'CAPE_bodyshape': CAPE_bodyshape,
}


def get_dataset(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown dataset: %s' % name)
    else:
        return obj
