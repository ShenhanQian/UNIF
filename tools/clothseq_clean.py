import os
import sys
import numpy as np
import time
import trimesh as tr
from multiprocessing import Pool


def clean_and_save(input_dir, output_dir, fname):
    path_in = os.path.join(input_dir, fname)
    mesh_in = tr.load_mesh(path_in, 'obj')

    path_out = os.path.join(output_dir, fname)
    mesh_out = tr.Trimesh(mesh_in.vertices/1000, mesh_in.faces)
    mesh_out.export(path_out)
    print('Saved:', path_out)

def clean_files_pool(input_dir, output_dir, pool_size=16):
    obj_list = os.listdir(input_dir)

    args_list = []
    for i, fname in enumerate(obj_list):
        suffix = fname.split('.')[-1]
        if suffix not in ['obj']:
            continue
        args_list.append([input_dir, output_dir, fname])
    
    tic = time.time()
    print('Begin...')
    pool = Pool(processes=pool_size)
    pool.starmap(clean_and_save, args_list)
    pool.close()
    toc = time.time()
    print('Processing done in {:.4f} seconds'.format(toc-tic))
    tic = time.time()

def clean_files(input_dir, output_dir):
    obj_list = os.listdir(input_dir)

    for i, fname in enumerate(obj_list):
        suffix = fname.split('.')[-1]
        if suffix not in ['obj']:
            continue

        clean_and_save(input_dir, output_dir, fname)


if __name__ == '__main__':
    assert len(sys.argv) == 3

    _, input_dir, output_dir = sys.argv

    # clean_files(input_dir, output_dir)  # single process
    clean_files_pool(input_dir, output_dir)  # multi-process