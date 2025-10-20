import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.loss import CharbonnierLoss, VGGLoss

logger = logging.getLogger('base')


class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        # --- tentukan device: paksa CPU kalau CUDA tidak tersedia / gpu_ids kosong ---
        use_cuda = bool(opt.get('gpu_ids')) and torch.cuda.is_available()
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')

        if opt.get('dist'):
            self.rank = torch.distributed.get_rank() if use_cuda else -1
        else:
            self.rank = -1  # non dist training

        train_opt = opt.get('train', {})

        # --- define network dan pindahkan ke device ---
        self.netG = networks.define_G(opt).to(self.device)

        # Bungkus paralel HANYA jika memang pakai CUDA
        if use_cuda and opt.get('dist'):
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        elif use_cuda and len(opt.get('gpu_ids', [])) > 1:
            self.netG = DataParallel(self.netG)
        # CPU: jangan dibungkus apa-apa

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt.get('pixel_criterion', 'l1')
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError(f'Loss type [{loss_type}] is not recognized.')
            self.l_pix_w = train_opt.get('pixel_weight', 1.0)

            self.cri_pix_ill  = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)

            self.cri_vgg = VGGLoss()

            #### optimizers
            wd_G = train_opt.get('weight_decay_G', 0) or 0
            if train_opt.get('ft_tsa_only'):
                normal_params, tsa_fusion_params = [], []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        (tsa_fusion_params if 'tsa_fusion' in k else normal_params).append(v)
                    elif self.rank <= 0:
                        logger.warning('Params [%s] will not optimize.', k)
                optim_params = [
                    {'params': normal_params, 'lr': train_opt.get('lr_G', 1e-4)},
                    {'params': tsa_fusion_params, 'lr': train_opt.get('lr_G', 1e-4)},
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    elif self.rank <= 0:
                        logger.warning('Params [%s] will not optimize.', k)

            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=train_opt.get('lr_G', 1e-4),
                weight_decay=wd_G,
                betas=(train_opt.get('beta1', 0.9), train_opt.get('beta2', 0.999)),
            )
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            scheme = train_opt.get('lr_scheme', 'MultiStepLR')
            if scheme == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt.get('lr_steps', []),
                            restarts=train_opt.get('restarts', []),
                            weights=train_opt.get('restart_weights', []),
                            gamma=train_opt.get('lr_gamma', 0.5),
                            clear_state=train_opt.get('clear_state', False),
                        )
                    )
            elif scheme == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt.get('T_period', [250000]),
                            eta_min=train_opt.get('eta_min', 1e-7),
                            restarts=train_opt.get('restarts', []),
                            weights=train_opt.get('restart_weights', []),
                        )
                    )
            else:
                raise NotImplementedError(f'LR scheme [{scheme}] not implemented.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf    = data['nf'].to(self.device)
        if need_GT and 'GT' in data:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        if self.optimizers and self.optimizers[0].param_groups:
            self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt.get('train', {}).get('ft_tsa_only') and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        dark  = self.var_L
        dark  = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 1e-4)

        b, _, h, w = mask.shape
        mask_max = torch.max(mask.view(b, -1), dim=1)[0].view(b, 1, 1, 1).repeat(1, 1, h, w)
        mask = torch.clamp(mask / (mask_max + 1e-4), min=0.0, max=1.0).float()

        self.fake_H, self.fake_Amp, self.fake_H_s1, self.snr = self.netG(self.var_L)

        image_fft  = torch.fft.fft2(self.real_H, norm='backward')
        self.real_Amp = torch.abs(image_fft)
        self.real_Pha = torch.angle(image_fft)

        out_fft   = torch.fft.fft2(self.fake_H, norm='backward')
        self.fake_Pha = torch.angle(out_fft)

        l_pix  = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_amp  = self.l_pix_w * self.cri_pix_ill(self.fake_Amp, self.real_Amp) * 0.01
        l_vgg  = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.1
        l_final = l_pix + l_amp + l_vgg
        l_final.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)
        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_amp'] = l_amp.item()
        self.log_dict['l_vgg'] = l_vgg.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, _, self.fake_H_s1, self.snr = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ']     = self.var_L.detach()[0].float().cpu()
        out_dict['rlt']    = self.fake_H.detach()[0].float().cpu()
        out_dict['rlt_s1'] = self.fake_H_s1.detach()[0].float().cpu()
        out_dict['rlt2']   = self.nf.detach()[0].float().cpu()
        if need_GT and hasattr(self, 'real_H'):
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        # bersihkan variabel besar
        if hasattr(self, 'real_H'): del self.real_H
        if hasattr(self, 'nf'):     del self.nf
        if hasattr(self, 'var_L'):  del self.var_L
        if hasattr(self, 'fake_H'): del self.fake_H

        # kosongkan cache CUDA hanya jika tersedia
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{}] ...'.format(load_path_G))
            # ambil modul asli jika dibungkus DataParallel
            target = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG

            # muat checkpoint ke device yang benar (CPU/GPU)
            ckpt = torch.load(load_path_G, map_location=self.device)

            # jika checkpoint berupa dict (sering ada 'state_dict' atau sejenisnya)
            if isinstance(ckpt, dict):
                for key in ['state_dict', 'params', 'model', 'net', 'netG', 'generator']:
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break

            # hilangkan prefix 'module.' kalau ada
            if isinstance(ckpt, dict):
                new_ckpt = OrderedDict()
                for k, v in ckpt.items():
                    new_ckpt[k.replace('module.', '')] = v
                ckpt = new_ckpt

            strict = bool(self.opt.get('path', {}).get('strict_load', True))
            target.load_state_dict(ckpt, strict=strict)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
