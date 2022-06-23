import os.path
import sys
import math
import argparse
import time
import random
from datetime import datetime
import numpy as np
from collections import OrderedDict
import logging

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import lpips_models

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(logdir='./tb_logger/' + opt['name'] + '/' + datetime.now().strftime('%y%m%d-%H%M%S'))

    # random seed
    seed = opt['train']['manual_seed']
    #seed = None
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)
    lpips_mode = lpips_models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True,version='0.1')

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                lp = 0.0
                idx = 0

                img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                util.mkdir(img_dir)
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('# Validation # <epoch:{:3d}, iter:{:8,d}>'.format(epoch, current_step))

                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    visuals['SR'] = visuals['SR'][0:visuals['HR'].size(0),0:visuals['HR'].size(1),0:visuals['HR'].size(2)]
                    score = lpips_mode.forward(visuals['HR']*2-1,visuals['SR']*2-1)
                    lp_value = score.item()
                    lp += lp_value
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['HR'])  # uint8
                    h,w,c = gt_img.shape
                    sr_img = sr_img[0:h,0:w,0:c]
                    #dr_img = util.tensor2img(visuals['DR'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, 'SR{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)
                    """
                    save_img_path = os.path.join(img_dir, 'DR{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(dr_img, save_img_path)
                    """
                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    psnr_value = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_psnr += psnr_value
                    ssim_value = util.ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += ssim_value

                    logger_val.info('<image:{}> PSNR: {:.4e}, SSIM: {:.4e}, LPIPS: {:.4e}'.format(
                        img_name, psnr_value, ssim_value, lp_value))

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_lpips = lp / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}, LPIPS: {:.4e}'.format(avg_psnr, avg_ssim, avg_lpips))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('# Validation # <epoch:{:3d}, iter:{:8,d}> PSNR: {:.4e}, SSIM: {:.4e}, LPIPS: {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_lpips))
                #logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                #logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    #epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

                delete_step = current_step - opt['logger']['save_checkpoint_freq'] * opt['logger']['max_save_num']
                delete_G_path = os.path.join(opt['path']['models'], f"{delete_step}_G.pth")
                if os.path.exists(delete_G_path):
                    os.remove(delete_G_path)
                delete_D_path = os.path.join(opt['path']['models'], f"{delete_step}_D.pth")
                if os.path.exists(delete_D_path):
                    os.remove(delete_D_path)
                delete_D2_path = os.path.join(opt['path']['models'], f"{delete_step}_D2.pth")
                if os.path.exists(delete_D2_path):
                    os.remove(delete_D2_path)
                delete_state_path = os.path.join(opt['path']['training_state'], f"{delete_step}.state")
                if os.path.exists(delete_state_path):
                    os.remove(delete_state_path)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
