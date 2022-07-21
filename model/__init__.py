from .inr import INR, UNIF


module_dict = {
    'INR': INR,
    'UNIF': UNIF,
}


def get_model(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown model: %s' % name)
    else:
        return obj
