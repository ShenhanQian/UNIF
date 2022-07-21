from .learner import Learner
from .shape_learner import ShapeLearner
from .shape_learner_unif import ShapeLearnerUNIF
from .shape_learner_smpl import ShapeLearnerSMPL


module_dict = {
    'Learner': Learner,
    'ShapeLearner': ShapeLearner,
    'ShapeLearnerUNIF': ShapeLearnerUNIF,
    'ShapeLearnerSMPL': ShapeLearnerSMPL,
}


def get_pipeline(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown pipeline: %s' % name)
    else:
        return obj
