import os
import logging

import numpy as np
from torch.utils.data.dataset import Dataset


class CAPE_bodyshape(Dataset):
    def __init__(
        self, 
        split,
        data_root,
        subject_name,
        ): 

        self.split = split

        assert split in ['train', 'val'], f'Unknown dataset split: {split}'
        msg_prefix = (f'[DATASET]({split}) ')
        logging.info(msg_prefix + f'Dataset root: {data_root}.')

        assert subject_name is not None, 'DATASET.kwargs.subject_name is required.'

        verts_path = os.path.join(data_root, 'minimal_body_shape', subject_name, f'%s_minimal.npy' % subject_name)
        verts = np.load(verts_path)

        gender_path = os.path.join(data_root, 'misc', 'subj_genders.pkl')
        gender = np.load(gender_path, allow_pickle=True)[subject_name]
        
        self.items = []

        item = {
            'verts': verts,
            'gender': gender,
        }
        self.items.append(item)

        logging.info(msg_prefix + f'subject: {subject_name}')
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.pipeline.default import get_cfg_defaults
    from dataset import get_dataset
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/dataset/cape_bodyshape.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    dataset = get_dataset(cfg.DATASET.name)(split='train', **cfg.DATASET.kwargs)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
 
    for batch in dataloader:
        pass
