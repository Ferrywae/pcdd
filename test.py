import os.path as osp
import logging
import argparse

import cv2
import numpy as np
import torch
import lpips

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ------------------------------------------------------------
# NOTE: Script ini diset untuk CPU by default.
# Jika nanti pakai GPU, pastikan opt['gpu_ids'] tidak kosong
# dan torch.cuda.is_available() == True.
# ------------------------------------------------------------

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options/test/LOLv2_real.yml',
                    help='Path to options YAML file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    # Tentukan device dari YAML + ketersediaan CUDA
    use_cuda = bool(opt.get('gpu_ids')) and torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # LPIPS di-setup sesuai device
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    save_imgs = True
    model = create_model(opt)  # pastikan model internal juga ikut device dari opt
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    output_folder_s1 = osp.join(save_folder, 'images/output_s1')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(output_folder_s1)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt, psnr_rlt_avg, psnr_total_avg = {}, {}, 0.0
        ssim_rlt, ssim_rlt_avg, ssim_total_avg = {}, {}, 0.0
        lpips_rlt, lpips_rlt_avg, lpips_total_avg = {}, {}, 0.0

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']

            if psnr_rlt.get(folder) is None:  psnr_rlt[folder] = []
            if ssim_rlt.get(folder) is None:  ssim_rlt[folder] = []
            if lpips_rlt.get(folder) is None: lpips_rlt[folder] = []

            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()

            # uint8 BGR (sesuai util.tensor2img)
            rlt_img   = util.tensor2img(visuals['rlt'])
            rlt_s1_img= util.tensor2img(visuals['rlt_s1'])
            gt_img    = util.tensor2img(visuals['GT'])
            input_img = util.tensor2img(visuals['LQ'])

            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    cv2.imwrite(osp.join(output_folder,     f'{tag}.png'), rlt_img)
                    cv2.imwrite(osp.join(GT_folder,         f'{tag}.png'), gt_img)
                    cv2.imwrite(osp.join(output_folder_s1,  f'{tag}.png'), rlt_s1_img)
                    cv2.imwrite(osp.join(input_folder,      f'{tag}.png'), input_img)
                except Exception as e:
                    print('Save error:', e)

            # --- Metrics ---
            # PSNR: referensi = GT terlebih dahulu, data_range=255 utk uint8
            psnr = peak_signal_noise_ratio(gt_img, rlt_img, data_range=255)
            psnr_rlt[folder].append(psnr)

            # SSIM: gunakan API baru skimage (channel_axis=-1)
            ssim = structural_similarity(gt_img, rlt_img, channel_axis=-1)
            ssim_rlt[folder].append(ssim)

            # LPIPS: ekspektasi input [-1, 1], CHW, float32
            img_t = torch.from_numpy(rlt_img.astype(np.float32) / 255.0)
            gt_t  = torch.from_numpy(gt_img.astype(np.float32) / 255.0)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            gt_t  = gt_t.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            lpips_alex = loss_fn_alex(img_t, gt_t).detach().cpu().item()
            lpips_rlt[folder].append(lpips_alex)

            pbar.update('Test {} - {}'.format(folder, idx_d))

        # Rata-rata per-folder dan total
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]
        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]
        for k, v in lpips_rlt.items():
            lpips_rlt_avg[k] = sum(v) / len(v)
            lpips_total_avg += lpips_rlt_avg[k]

        psnr_total_avg  /= max(1, len(psnr_rlt))
        ssim_total_avg  /= max(1, len(ssim_rlt))
        lpips_total_avg /= max(1, len(lpips_rlt))

        log_s = '# Validation # PSNR: {:.6f}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.6f}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.6f}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.6f}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # LPIPS: {:.6f}:'.format(lpips_total_avg)
        for k, v in lpips_rlt_avg.items():
            log_s += ' {}: {:.6f}'.format(k, v)
        logger.info(log_s)

        # Rata-rata global semua frame
        psnr_all = sum(sum(v) for v in psnr_rlt.values()) / max(1, sum(len(v) for v in psnr_rlt.values()))
        ssim_all = sum(sum(v) for v in ssim_rlt.values()) / max(1, sum(len(v) for v in ssim_rlt.values()))
        lpips_all = sum(sum(v) for v in lpips_rlt.values()) / max(1, sum(len(v) for v in lpips_rlt.values()))

        print('PSNR_all:', psnr_all)
        print('SSIM_all:', ssim_all)
        print('LPIPS_all:', lpips_all)


if __name__ == '__main__':
    main()
