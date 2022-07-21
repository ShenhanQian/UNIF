import os
from pathlib import Path
import logging
import time
from collections import OrderedDict
from datetime import datetime
import numpy as np
import cv2 as cv
import matplotlib as mpl
mpl.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.eta import ETA
from utils.types import AccumuDict
from model import get_model
from utils import get_visualizer
from dataset import get_dataset


class Learner(object):
    """A generic object for neural network experiments."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_pipeline = self.cfg.PIPELINE
        self.cfg_model = self.cfg.MODEL
        self.cfg_exp = self.cfg.EXP
        self.cfg_optim = self.cfg.OPTIMIZER
        self.cfg_sch = self.cfg.SCHEDULER
        self.cfg_dataset = self.cfg.DATASET
        self.cfg_vis = self.cfg.VISUALIZER

        self.optimizers = {}
        self.schedulers = {}
        self.vis = get_visualizer(self.cfg_vis.name)(**self.cfg_vis.kwargs)
    
    def train(self):
        mode = 'TRAIN'
        epoch, global_step = self._setup_exp(mode)
        # self.scaler = torch.cuda.amp.GradScaler()

        while epoch < self.cfg_exp.TRAIN.num_epochs:
            epoch += 1
            torch.cuda.synchronize()
            tic_epoch = time.time()
            loss_accum_dict = AccumuDict()
            self.model.train()
            for step, batch in enumerate(self.train_dataloader, 1):
                torch.cuda.synchronize()
                tic_step = time.time()
                global_step += 1

                # with torch.cuda.amp.autocast():
                result_dict, loss_dict = self.forward(batch)
                loss_accum_dict.accumulate(loss_dict)

                self._optimize_step(loss_dict)

                if step % self.cfg_exp.interval_log_step == 0:
                    self._write_logs_step(mode, loss_dict, global_step, epoch, step, tic_step, self.cfg_exp.TRAIN.num_epochs, len(self.train_dataloader))
                if step % self.cfg_exp.interval_result_step == 0:
                    self._save_results(mode, result_dict, epoch, step, self.cfg_exp.TRAIN.num_epochs, len(self.train_dataloader))
                    
            if epoch % self.cfg_exp.interval_log_epoch == 0:
                self._write_logs_epoch(mode, loss_dict, global_step, epoch, step, tic_epoch, self.cfg_exp.TRAIN.num_epochs, len(self.train_dataloader))
            if epoch % self.cfg_exp.interval_result_epoch == 0:
                self._save_results(mode, result_dict, epoch, step, self.cfg_exp.TRAIN.num_epochs, len(self.train_dataloader))

            if epoch % self.cfg_exp.interval_ckpt_epoch == 0:
                self._save_ckpt(global_step, epoch)
                if self.cfg_exp.validate:
                    self.validate(epoch)
            
            for k, sch in self.schedulers.items():
                sch.step()

        self.tb_writer.close()
    
    def _optimize_step(self, loss_dict):
        self.optimizers['base'].zero_grad()
        
        # default behavior
        loss_dict['loss'].backward()
        self.optimizers['base'].step()

        # # With Gradient Scaler
        # self.scaler.scale(loss_dict['loss']).backward()
        # self.scaler.step(self.optimizers['base'])
        # self.scaler.update()

    def validate(self, epoch):
        mode = 'VAL'

        torch.cuda.synchronize()
        tic_epoch = time.time()
        loss_accum_dict = AccumuDict()
        self.model.eval()
        for step, batch in enumerate(self.test_dataloader, 1):
            torch.cuda.synchronize()
            tic_step = time.time()

            result_dict, loss_dict = self.forward(batch, is_testing=True)
            loss_dict = self._detach_tensor_dict(loss_dict)
            loss_accum_dict.accumulate(loss_dict)

            if step % self.cfg_exp.interval_log_step == 0:
                self._write_logs_step(mode, loss_dict, None, epoch, step, tic_step, self.cfg_exp.TRAIN.num_epochs, len(self.test_dataloader))
            if step % self.cfg_exp.interval_result_step == 0 or self.cfg_exp.TEST.save_all_results:
                self._save_results(mode, result_dict, epoch, step, self.cfg_exp.TRAIN.num_epochs, len(self.test_dataloader))
        
        if step % self.cfg_exp.interval_result_step != 0 and not self.cfg_exp.TEST.save_all_results:
            self._save_results(mode, result_dict, epoch, step, self.cfg_exp.TRAIN.num_epochs, len(self.test_dataloader))
        self._write_logs_epoch(mode, loss_accum_dict.mean(), None, epoch, step, tic_epoch, self.cfg_exp.TRAIN.num_epochs, len(self.test_dataloader))
    
    def test(self):
        mode = 'TEST'
        self._setup_exp(mode)
        
        epoch = 1
        torch.cuda.synchronize()
        tic_epoch = time.time()
        loss_accum_dict = AccumuDict()
        self.model.eval()
        for step, batch in enumerate(self.test_dataloader, 1):
            torch.cuda.synchronize()
            tic_step = time.time()

            result_dict, loss_dict = self.forward(batch, is_testing=True)
            loss_dict = self._detach_tensor_dict(loss_dict)
            loss_accum_dict.accumulate(loss_dict)

            if step % self.cfg_exp.interval_log_step == 0:
                self._write_logs_step(mode, loss_dict, None, epoch, step, tic_step, 1, len(self.test_dataloader))
            if step % self.cfg_exp.interval_result_step == 0 or self.cfg_exp.TEST.save_all_results:
                self._save_results(mode, result_dict, epoch, step, 1, len(self.test_dataloader))
        
        if step % self.cfg_exp.interval_result_step != 0 and not self.cfg_exp.TEST.save_all_results:
            self._save_results(mode, result_dict, epoch, step, 1, len(self.test_dataloader))
        self._write_logs_epoch(mode, loss_accum_dict.mean(), None, epoch, step, tic_epoch, 1, len(self.test_dataloader))
        
        self.tb_writer.close()

    def _detach_tensor_dict(self, tensor_dict):
        return dict(map(lambda x: (x[0], x[1].detach()), tensor_dict.items()))
    
    def _shrink_tensor_dict_(self, tensor_dict, shrink_size):
        """ in-place shrink """
        for k in tensor_dict:
            if isinstance(tensor_dict[k], torch.Tensor):
                tensor_dict[k] = tensor_dict[k][:shrink_size]
            elif isinstance(tensor_dict[k], dict):
                tensor_dict[k] = self._shrink_tensor_dict_(tensor_dict[k], shrink_size)
            else:
                raise NotImplementedError
        return tensor_dict
    
    def _slice_tensor_dict(self, tensor_dict, index):
        tensor_dict = tensor_dict.copy()  # avoid modify the original tensor_dict
        for k in tensor_dict:
            if isinstance(tensor_dict[k], torch.Tensor):
                tensor_dict[k] = tensor_dict[k][index]
            elif isinstance(tensor_dict[k], dict):
                tensor_dict[k] = self._slice_tensor_dict(tensor_dict[k], index)
            else:
                raise NotImplementedError
        return tensor_dict
    
    def _write_logs_save(self, mode, epoch, step, tic_save, num_epochs, steps_per_epoch):
        g_mem = torch.cuda.memory_reserved()/1024/1024
        toc_save = time.time() - tic_save
        msg = '[%5s] epoch: %d/%d  step: %d/%d  g_mem: %d MB  time_save: %.3f s | Saved all results.' % (
            mode, epoch, num_epochs, step, steps_per_epoch, g_mem, toc_save)
        logging.info(msg)
    
    def _write_logs_step(self, mode, loss_dict, global_step, epoch, step, tic_step, num_epochs, steps_per_epoch):
        msg = ''
        # lr
        if mode == 'TRAIN':
            for k, opt in self.optimizers.items():
                lr = opt.param_groups[0]['lr']
                msg += 'lr_%s: %.1e  ' % (k, lr)

        # loss
        for k, v in loss_dict.items():
            loss = v.detach().item()
            msg += '%s: %.5f  ' % (k, v)
            if mode == 'TRAIN':
                self.tb_writer.add_scalar(f'{mode}/{k}', loss, global_step)

        # mem and time
        g_mem = torch.cuda.memory_reserved()/1024/1024
        torch.cuda.synchronize()
        toc_step = (time.time() - tic_step)

        msg_prefix = '[%5s] epoch: %d/%d  step: %d/%d  g_mem: %d MB  time_step:  %.3f s | ' % (
                mode, epoch, num_epochs, step, steps_per_epoch, g_mem, toc_step)
        if mode == 'TRAIN':
            eta = self.eta(step=step, epoch=epoch, toc_step=toc_step)
            msg_prefix += 'ETA: %.2f h | ' % (eta/3600)
        logging.info(msg_prefix + msg)
    
    def _write_logs_epoch(self, mode, loss_dict_epoch, global_step, epoch, step, tic_epoch, num_epochs, steps_per_epoch):
        msg = ''
        # lr
        if mode == 'TRAIN':
            for k, opt in self.optimizers.items():
                lr = opt.param_groups[0]['lr']
                msg += 'lr_%s: %.1e  ' % (k, lr)
                self.tb_writer.add_scalar(f'{mode}/lr_{k}', lr, epoch)

        # loss
        for k, v in loss_dict_epoch.items():
            loss = v.detach().item()
            msg += '%s: %.5f  ' % (k, v)
            if mode == 'TRAIN':
                self.tb_writer.add_scalar(f'{mode}/{k}', loss, global_step)
            else:
                self.tb_writer.add_scalar(f'{mode}/{k}', loss, epoch)

        # mem and time
        g_mem = torch.cuda.memory_reserved()/1024/1024
        torch.cuda.synchronize()
        toc_epoch = time.time() - tic_epoch

        msg_prefix = '[%5s] epoch: %d/%d  step: %d/%d  g_mem: %d MB  time_epoch: %.3f s | ' % (
            mode, epoch, num_epochs, step, steps_per_epoch, g_mem, toc_epoch)
        if mode == 'TRAIN':
            eta = self.eta(epoch=epoch, toc_epoch=toc_epoch)
            self.tb_writer.add_scalar(f'{mode}/ETA', eta/3600, epoch)
            msg_prefix += 'ETA: %.2f h | ' % (eta/3600)
        logging.info(msg_prefix + msg)
    
    def _save_ckpt(self, global_step, epoch):
        ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'ckpt-ep%02d.pth' % epoch)

        ckpt_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'model': self.model.state_dict(),
        }
        for k, opt in self.optimizers.items():
            ckpt_dict[f'optimizer_{k}'] = opt.state_dict()
        torch.save(ckpt_dict, ckpt_path)

        msg = '[TRAIN] epoch: %d/%d  checkpoint saved: %s' % (
            epoch, self.cfg_exp.TRAIN.num_epochs, ckpt_path)
        logging.info(msg)
    
    def _save_results(self, mode, result_dict, epoch, step, num_epochs, steps_per_epoch):
        torch.cuda.synchronize()
        tic_save = time.time()

        # if self.cfg_exp.save_img:
        #     self._save_img_batch(result_dict['out'], mode, epoch, step, name='out')
        
        # if self.cfg_exp.save_fig:
        #     self._save_fig_batch(result_dict, mode, epoch, step)
        
        toc_save = time.time() - tic_save
        msg = '[%5s] epoch: %d/%d  step: %d/%d  time_save: %.3f s | Saved results.' % (
            mode, epoch, num_epochs, step, steps_per_epoch, toc_save)
        logging.info(msg)
    
    def _get_save_path(self, mode, epoch, batch_idx, dir_name, file_name=None):
        out_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'ep%04d-%s' % (epoch, mode))

        base_idx = self.cfg_exp.TRAIN.batch_size * (batch_idx - 1)
        name = f'-{file_name}' if file_name is not None else ''

        return  out_path, name, base_idx
    
    def _save_img_batch_grid(self, img_batch_list, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'img', name)

        B = img_batch_list[0].shape[0]
        for idx in range(B):
            img_name = '%06d' % (base_idx+idx)
            if name is not None:
                img_name = f'{img_name}_{name}'
            img_path = os.path.join(base_path, 'ep%02d-%s-%s.jpg' % (epoch, mode, img_name))

            img_grid = torch.cat([(x[idx].clamp(0, 1) * 255) for x in img_batch_list], 2).detach().byte().permute([1,2,0]).cpu().numpy()
            assert img_grid.shape[-1] == 3
            img_grid = cv.cvtColor(img_grid, cv.COLOR_RGB2BGR)
            cv.imwrite(img_path, img_grid)
    
    def _save_img_batch(self, img_batch, mode, epoch, batch_idx, name=None):
        base_path, name, base_idx = self._get_save_path(mode, epoch, batch_idx, 'img', name)

        B = img_batch.shape[0]
        for idx in range(B):
            img_name = '%06d' % (base_idx+idx)
            if name is not None:
                img_name = f'{img_name}_{name}'
            img_path = os.path.join(base_path, 'ep%02d-%s-%s.jpg' % (epoch, mode, img_name))

            img = (img_batch[idx].clamp(0, 1) * 255).byte().permute([1,2,0]).detach().cpu().numpy()
            assert img.shape[-1] == 3
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imwrite(img_path, img)
    
    def _save_fig_batch(self, result_dict, mode, epoch, batch_idx, name=None):
        pass
    
    def _filter_param(self, named_params, blacklist=[], whitelist=[], prefix=''):
        if not isinstance(named_params, OrderedDict):
            named_params = OrderedDict(named_params)

        update_params = OrderedDict()
        blacklist_names = []
        whitelist_names = []
        if len(blacklist) > 0:
            for k, v in named_params.items():
                for b in blacklist:
                    if b in k:
                        blacklist_names.append(k)
                    else:
                        whitelist_names.append(k)
                        update_params[k] = v
        elif len(whitelist) > 0:
            for k, v in named_params.items():
                for w in whitelist:
                    if w in k:
                        whitelist_names.append(k)
                        update_params[k] = v
                    else:
                        blacklist_names.append(k)
        else:
            for k, v in named_params.items():
                whitelist_names.append(k)
                update_params[k] = v
        
        self._formated_print_parameters(whitelist_names, prefix=prefix+' Whitelist parameters:')
        self._formated_print_parameters(blacklist_names, prefix=prefix+' Blacklist parameters:')
        return update_params
    
    def _map_param(self, named_params, map_list=(), prefix=''):
        if not isinstance(named_params, OrderedDict):
            named_params = OrderedDict(named_params)

        if len(map_list) > 0:
            mapped_names = []
            update_params = OrderedDict()

            for k, v in named_params.items():

                k_new = k
                for name_old, name_new in map_list:
                    if name_old in k:
                        k_new = k_new.replace(name_old, name_new)
                if k_new != k:
                    assert k_new not in named_params
                    update_params[k_new] = named_params[k]
                    mapped_names.append('%s -> %s' % (k, k_new))
                else:
                    update_params[k] = named_params[k]
            self._formated_print_parameters(mapped_names, prefix=prefix+' Mapped parameters:', num_cols=1)
            return update_params
        else:
            return named_params

    def _formated_print_parameters(self, params, prefix='', num_cols=4):
        msg = '[PARAM]%s\n' % (prefix)

        params_cols = [params[i:i + num_cols] for i in range(0, len(params), num_cols)]
        params_lines = zip(params_cols)
        for params in params_lines:
            msg += '  '
            for p in params[0]:
                msg += ('%-40s  ' % p)
            msg += '\n'
        logging.info(msg)
    
    def _compose_ckpt(self, ckpt_path, ckpt_map=None):
        if ckpt_map is None:
            assert isinstance(ckpt_path, str), 'Only support a single checkpoint path when EXP.checkpoint_map is not provided.'
            assert os.path.exists(ckpt_path), 'File not exists: %s' % ckpt_path
            assert ckpt_path.split('.')[-1]=='pth', 'File type not supported: %s' % ckpt_path

            ckpt = torch.load(ckpt_path)
        else:
            if isinstance(ckpt_path, str):
                assert isinstance(ckpt_map, tuple)
                
                ckpt = torch.load(ckpt_path)
                ckpt['model'] = OrderedDict(
                    map(lambda x: (x[0].replace(*ckpt_map), x[1]), 
                        ckpt['model'].items()))
            elif isinstance(ckpt_path, list):
                assert isinstance(ckpt_map, list)
                assert len(ckpt_path) == len(ckpt_map), 'Number of EXP.CHECKPATH_PATH (%d) should match with EXP.checkpoint_map (%d).' % (len(ckpt_path), len(ckpt_map))

                ckpt = {'model': OrderedDict()}
                for i, p in enumerate(ckpt_path):
                    tmp_ckpt = torch.load(p)
                    tmp_ckpt['model'] = OrderedDict(
                        map(lambda x: (x[0].replace(*ckpt_map[i]), x[1]), 
                            tmp_ckpt['model'].items()))
                    ckpt['model'].update(tmp_ckpt['model'])
            else:
                raise TypeError("Expected EXP.checkpoint of type {<class 'str'>, <class 'list'>}, get type: %s" % type(ckpt_path))
        return ckpt
    
    def _setup_exp(self, mode):
        if mode == 'TRAIN':
            # resume training
            if self.cfg_exp.resume_from is not None:
                self._setup_output(mode, Path(self.cfg_exp.resume_from).resolve().parent.parent)

                ckpt = self.cfg_exp.resume_from
                logging.info('Resuming from checkpoint: %s' % ckpt)
                ckpt = self._compose_ckpt(self.cfg_exp.resume_from)

                epoch = ckpt['epoch']
                global_step = ckpt['global_step']
                self._setup_dataset(mode)
                self._setup_model(ckpt['model'])
                self._setup_optimizer(epoch, ckpt)
            # train from a pretrained a model
            elif self.cfg_exp.checkpoint is not None:
                self._setup_output(mode)

                ckpt = self.cfg_exp.checkpoint
                logging.info('Loading from pretrained checkpoint: %s' % ckpt)
                ckpt = self._compose_ckpt(self.cfg_exp.checkpoint, self.cfg_exp.checkpoint_map)

                epoch = 0
                global_step = 0
                self._setup_dataset(mode)
                self._setup_model(ckpt['model'])
                self._setup_optimizer(epoch)
            # train from scratch
            else:
                self._setup_output(mode)
                logging.info('Training from scratch.')

                epoch = 0
                global_step = 0
                self._setup_dataset(mode)
                self._setup_model()
                self._setup_optimizer(epoch)
            
            return epoch, global_step

        elif mode == 'TEST':
            if self.cfg_exp.checkpoint is not None:
                self._setup_output(mode)

                ckpt = self.cfg_exp.checkpoint
                logging.info('Loading from checkpoint: %s' % ckpt)
                ckpt = self._compose_ckpt(self.cfg_exp.checkpoint, self.cfg_exp.checkpoint_map)

                self._setup_dataset(mode)
                self._setup_model(ckpt['model'])
            else:
                raise Exception('Checkpoint file is not provided.')

        else:
            raise NotImplementedError

    def _setup_output(self, mode, resume_dir=None):
        # output_dir
        if resume_dir is not None:
            assert mode == 'TRAIN'

            self.output_dir = resume_dir
        else:
            dt = str(datetime.now())[2:-7].replace(':', '').replace('-', '').replace(' ', '-')
            tag = '%s-%s_%s' % (dt, mode, self.cfg_exp.tag)

            self.output_dir = os.path.join(self.cfg_exp.base_dir, self.cfg_exp.config, tag)
            os.makedirs(self.output_dir, exist_ok=True)

        # logger
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-0.4s] %(message)s")


        log_path = '%s/log.txt' % self.output_dir
        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        logging.info('log path: %s' % log_path)
        logging.info('\n====== Configurations ======\n' + str(self.cfg) + '\n============\n')

        # tensorboard writer
        self.tb_writer = SummaryWriter(log_dir=self.output_dir)
    
    def _setup_dataset(self, mode):
        if mode == 'TRAIN':
            self.train_dataset = get_dataset(self.cfg_dataset.name)(split='train', **self.cfg_dataset.kwargs)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg_exp.TRAIN.batch_size,
                shuffle=True,
                num_workers=self.cfg_exp.num_workers,
                persistent_workers=self.cfg_exp.persistent_workers if self.cfg_exp.num_workers > 0 else False)
            if self.cfg_exp.validate:
                self.test_dataset = get_dataset(self.cfg_dataset.name)(split='val', **self.cfg_dataset.kwargs)
                self.test_dataloader = DataLoader(
                    self.test_dataset, 
                    batch_size=self.cfg_exp.TEST.batch_size, 
                    shuffle=self.cfg_exp.TEST.shuffle, 
                    num_workers=self.cfg_exp.num_workers)
                self.eta = ETA(len(self.train_dataloader) + len(self.test_dataloader), self.cfg_exp.TRAIN.num_epochs)
            else:
                self.eta = ETA(len(self.train_dataloader), self.cfg_exp.TRAIN.num_epochs)

        elif mode == 'TEST':
            self.test_dataset = get_dataset(self.cfg_dataset.name)(split='val', **self.cfg_dataset.kwargs)
            self.test_dataloader = DataLoader(
                self.test_dataset, 
                batch_size=self.cfg_exp.TEST.batch_size,
                shuffle=self.cfg_exp.TEST.shuffle,
                num_workers=self.cfg_exp.num_workers)
            self.eta = ETA(len(self.test_dataloader), 1)
        else:
            raise NotImplementedError()
    
    def _setup_model(self, state_dict=None):
        self._init_model()
        self.model = self.model.cuda()

        if state_dict is not None:
            state_dict = self._map_param(
                state_dict, 
                map_list=self.cfg_model.pretrain_map_list,
                prefix='(Pretraining)',
                )
            state_dict = self._filter_param(
                state_dict, 
                blacklist=self.cfg_model.pretrain_blacklist,
                whitelist=self.cfg_model.pretrain_whitelist,
                prefix='(Pretraining)',
                )
            # strict loading is enabled by default. Set MODEL.strict_load to False to cancel it.
            self.model.load_state_dict(state_dict, self.cfg_model.strict_load)
    
    def _init_model(self):
        self.model = get_model(self.cfg_model.name)(**self.cfg_model.kwargs)

    def forward(self, batch, is_testing=False):
        # prepare target
        tgt = torch.ones([8, 3, 256, 256]).cuda()

        # prepare input
        scalar = 2

        # model forward
        out = self.model * scalar

        result_dict = {
            'out': out,
        }

        # loss computation
        loss_dict = {}

        criterion = nn.MSELoss()
        loss_dict['loss'] = criterion(out, tgt)

        # metrics computation
        if is_testing:
            with torch.no_grad():
                err = torch.abs(out - tgt).mean()
                loss_dict['err'] = err

        return result_dict, loss_dict

    def _setup_optimizer(self, epoch, ckpt=None):
        if self.cfg_optim.type.lower() == 'adam':
            optim = torch.optim.Adam
        elif self.cfg_optim.type.lower() == 'adamw':
            optim = torch.optim.AdamW
        else:
            raise NotImplementedError('Unknown optimizer: %s' % self.cfg_optim.type.lower())
        
        # init optimizers
        self.optimizers['base'] = optim(params=self._setup_params(), lr=self.cfg_optim.lr)

        # load state_dict if a checkpoint is provided (intended for resumed training)
        if ckpt is not None:
            for k, opt in self.optimizers.items():
                ckpt_key = f'optimizer_{k}'
                opt.load_state_dict(ckpt[ckpt_key])

        if self.cfg_sch.type is None:
            pass
        elif self.cfg_sch.type == 'MultiStepLR':
            warm_up_steps = int(self.cfg_sch.interval * 0.1)
            milsstones = [self.cfg_sch.interval * (i+1) for i in range(self.cfg_sch.num_steps)]

            if self.cfg_sch.warmup_type is None:
                lr_lambda = lambda step: self.cfg_sch.gamma**len([m for m in milsstones if m <= step])
            elif self.cfg_sch.warmup_type == 'linear':
                lr_lambda = lambda step: (step+1) / warm_up_steps if step < warm_up_steps else self.cfg_sch.gamma**len([m for m in milsstones if m <= step])
            elif self.cfg_sch.warmup_type == 'exp':
                lr_lambda= lambda step: 1 - np.exp(-(step+1) / warm_up_steps * 10) if step < warm_up_steps else self.cfg_sch.gamma**len([m for m in milsstones if m <= step])
            else:
                raise NotImplementedError

            for k, opt in self.optimizers.items():
                self.schedulers[k] = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda, last_epoch=epoch-1)
        else:
            raise NotImplementedError
    
    def _setup_params(self):
        params = [
            v for k, v in self._filter_param(
                self.model.named_parameters(), 
                blacklist=self.cfg_optim.update_blacklist,
                whitelist=self.cfg_optim.update_whitelist,
                prefix='(Optimization)',
                ).items()
            ]
        return params
