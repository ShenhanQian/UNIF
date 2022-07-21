import os
import sys
import time
from multiprocessing import Manager, Pool
import pickle
import numpy as np
import trimesh as tr


def obj2ply(fname, src_dir, tgt_dir, flags, total):
    src_path = os.path.join(src_dir, fname)
    tgt_path = os.path.join(tgt_dir, fname.replace('.obj', '.ply'))
    frame_id = src_path.split('.')[0].split('/')[-1]
    print('frame_id: %s (%d / %d)' % (src_path, len(flags), total))
    # try:
    mesh = tr.load_mesh(src_path, 'obj')
    mesh.export(tgt_path)
    flags.append(1)
   
def run_pool(src_dir, tgt_dir, pool_size=16):
    mesh_list = [x for x in os.listdir(src_dir) if x.split('.')[-1] == 'obj']
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    
    with Manager() as manager:
        flags = manager.list()

        args_list = []
        for i, fname in enumerate(mesh_list):
            args_list.append([fname, src_dir, tgt_dir, flags, len(mesh_list)])
        
        tic = time.time()
        print('Begin...')
        pool = Pool(processes=pool_size)
        pool.starmap(obj2ply, args_list)
        pool.close()
        toc = time.time()
        print('Convertion done in {:.4f} seconds'.format(toc-tic))

def run(src_dir, tgt_dir):
    mesh_list = [x for x in os.listdir(src_dir) if x.split('.')[-1] == 'obj']
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)

    tic = time.time()
    print('Begin...')
    for i, fname in enumerate(mesh_list):
        obj2ply(fname, src_dir, tgt_dir, i, len(mesh_list))
    toc = time.time()
    print('Convertion done in {:.4f} seconds'.format(toc-tic))


if __name__ == '__main__':
    assert len(sys.argv) == 3

    _, src_dir, tgt_dir = sys.argv

    # run(src_dir, tgt_dir)
    run_pool(src_dir, tgt_dir, 16)