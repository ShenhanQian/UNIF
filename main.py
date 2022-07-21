import argparse
import yacs

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from config.pipeline.default import get_cfg_defaults
from pipeline import get_pipeline


def setup_config():
    parser = argparse.ArgumentParser(description="Main program")
    parser.add_argument("--cfg", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
        help="Modify config options using the command-line")
    args = parser.parse_args()

    cfg = get_cfg_defaults()

    with open(args.cfg, 'r') as f:
        current_cfg = cfg.load_cfg(f)
    components = ['PIPELINE', 'MODEL', 'DATASET', 'VISUALIZER']
    for component in components:
        cfg.merge_from_file(current_cfg[component].parent_cfg)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.EXP.config = args.cfg.split('/')[-1].split('.')[0]
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    cfg = setup_config()

    leaner = get_pipeline(cfg.PIPELINE.name)(cfg)
    if cfg.EXP.test_only:
        leaner.test()
    else:
        leaner.train()
