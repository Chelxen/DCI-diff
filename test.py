import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import options.options as option
from utils import util
from utils.intensity import to_raw_intensity, validate_intensity_config
from data import create_dataloader
from data.LACT_dataset import XCATDataset
import torchvision.transforms as T
import lpips
import model as Model
import core.logger as Logger
import core.metrics as Metrics
from torchvision import transforms
from dataconsistency.CTClass import set_CTClass
from measure import compute_measure,compute_MSE,compute_PSNR,compute_SSIM,compute_RMSE

transform = transforms.Lambda(lambda t: (t * 2) - 1)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to option YMAL file.',
                            default='./config/GE.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')


    parser.add_argument('-c', '--config', type=str, default='config/GE_test.json',
                        help='JSON file for configuration')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model weights (.pth); overrides path.resume_state in the JSON config')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--use_dc', action='store_true', default=False,
                        help='Enable data consistency during sampling')
    parser.add_argument('--angle', type=int, choices=[90, 120], default=90,
                        help='Angle for data selection: 90 or 120 degrees (default: 90)')
    parser.add_argument('--patient-id', type=str, default=None,
                        help='Select one GE patient ID without editing the dataset config')
    parser.add_argument('--param_fn', type=str, default=None,
                        help='Path to the parameter file for data consistency (auto-set based on --angle if not specified)')
    parser.add_argument('--continous', action='store_true')
    parser.add_argument('--dc_method', type=str, default='cgrad',
                        choices=['cgrad', 'apgm'],
                        help='Data consistency solver: cgrad / apgm')
    parser.add_argument('--dc_max_iter', type=int, default=20,
                        help='Maximum iterations for data consistency solver')
    

    args = parser.parse_args()
    
    opt = Logger.parse(args)
    if args.checkpoint is not None:
        opt['path']['resume_state'] = args.checkpoint
    if not opt['path'].get('resume_state') or not os.path.isfile(opt['path']['resume_state']):
        parser.error('Specify an existing model checkpoint with --checkpoint or path.resume_state in the JSON config')
    opt = Logger.dict_to_nonedict(opt)
    opt_dataset = option.parse(args.dataset, is_train=False)
    if args.patient_id is not None:
        opt_dataset['datasets']['val']['patient_ids'] = [args.patient_id]
    raw_data = validate_intensity_config(opt, opt_dataset)

    if args.param_fn is None:
        args.param_fn = f'./geo_config/{opt_dataset["dataset"]}_fan_{args.angle}.txt'
    
    # Update test_low_dir based on angle
    if 'val' in opt_dataset['datasets']:
        opt_dataset['datasets']['val']['test_low_dir'] = (
            f'limited_npy_{args.angle}' if opt_dataset['dataset'] == 'GE' else str(args.angle)
        )


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    opt['phase'] = 'test'
    opt['uncertainty_train'] = False
    data_range = float(opt.get('data_range', 1.0))
    display_range = (0.0, data_range) if raw_data else (-1.0, 1.0)

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        logger.info(f'Using angle: {args.angle} degrees')
        logger.info(f'Parameter file: {args.param_fn}')
        if 'val' in opt_dataset['datasets']:
            logger.info(f'Test data directory: {opt_dataset["datasets"]["val"]["test_low_dir"]}')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### seed
    seed = opt['seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt_dataset['dataset'] == 'GE':
        dataset_cls = XCATDataset
    # elif opt_dataset['dataset'] == 'LOLv2':
    #     dataset_cls = LOLv2_Dataset
    elif opt_dataset['dataset'] == 'XCAT':
        dataset_cls = XCATDataset
    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt_dataset['datasets'].items():
        if phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt_dataset)
            val_loader = create_dataloader(val_set, dataset_opt, opt_dataset, None)
            logger.info('Test data: %s (%d pairs), patient_ids=%s',
                        val_set.root, len(val_set), dataset_opt.get('patient_ids'))

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')


    loss_fn_vgg = lpips.LPIPS(net='alex') 
    result_path = '{}'.format(opt['path']['results'])
    result_path_gt = result_path+'/gt/'
    result_path_out = result_path+'/output/'
    result_path_input = result_path+'/input/'
    os.makedirs(result_path_gt, exist_ok=True)
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    # Initialize CTClass for data consistency
    ct_class = None
    
    if args.use_dc:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        param_fn = args.param_fn
        try:
            ct_class = set_CTClass(
                device=device,
                use_cuda=torch.cuda.is_available(),
                batch_size=1,
                use_static=False,
                use_fov=True,
                param_fn=param_fn,
                rank=0,
                logger=logger
            )
            ct_class.eval()  # Set to evaluation mode
            logger.info(f'CTClass initialized with param file: {param_fn}')
            logger.info('Data consistency will be applied at each sampling step')
        except Exception as e:
            logger.warning(f'Failed to initialize CTClass: {e}. Data consistency will be disabled.')
            ct_class = None
    else:
        logger.info('Data consistency is disabled (use --use_dc to enable)')
    
    logger_val = logging.getLogger('val')  # validation logger

    # Store all metric values for calculating mean and std
    psnr_values = []
    ssim_values = []
    rmse_values = []
    lpips_values = []
    idx = 0
    
    for val_data in val_loader:

        idx += 1
        diffusion.feed_data(val_data)

        lq_path = val_data.get("LQ_path", None)
        if isinstance(lq_path, (list, tuple)):
            f_name = lq_path[0]
        else:
            f_name = str(lq_path) if lq_path is not None else str(idx)
        
        # Generate sinogram from GT for data consistency
        sino_obs = None
        if ct_class is not None:
            # GT shape: (N, C, H, W), in the configured intensity space.
            gt = diffusion.data['GT']
            # Ensure GT is single channel for CT projection
            if gt.shape[1] > 1:
                gt = gt[:, 0:1, :, :]  # Take first channel if multi-channel
            # Raw GE intensities pass directly to the CT projector.
            gt_dc = to_raw_intensity(gt, data_range, raw_data)
            # Forward project GT to get sinogram
            with torch.no_grad():
                sino_obs = ct_class.fmult(gt_dc)  # Shape: (N, Views, Rows, Cols)
            logger_val.info(f'Sample {idx}: Generated sinogram from GT, shape: {sino_obs.shape}')
        
        # save continuous results
        continous = args.continous

        # Test with data consistency
        diffusion.test(
            continous=continous,
            ct_class=ct_class,
            sino_obs=sino_obs,
            apply_dc=(ct_class is not None),
            dc_method=args.dc_method,
            dc_max_iter=args.dc_max_iter,
            dc_lam=1e5,
            dc_lr=5e-6,
            dc_data_range=data_range,
        )

        visuals = diffusion.get_current_visuals()
        
        # Image tensors remain in the configured training intensity space.
        normal_img = visuals['HQ']   #(1,res,res)
        gt_img = visuals['GT']
        ll_img = visuals['LQ']

        #png
        normal_img_png = Metrics.tensor2img(visuals['HQ'], min_max=display_range)
        gt_img_png = Metrics.tensor2img(visuals['GT'], min_max=display_range)
        ll_img_png = Metrics.tensor2img(visuals['LQ'], min_max=display_range)

        if not continous:
            util.save_img(
                gt_img_png, '{}/{}_gt.png'.format(result_path_gt, f_name))
            util.save_npy(gt_img, '{}/{}_gt.npy'.format(result_path_gt, f_name), raw_data=raw_data, data_range=data_range)
            util.save_img(
                ll_img_png, '{}/{}_in.png'.format(result_path_input, f_name))
            util.save_npy(ll_img, '{}/{}_in.npy'.format(result_path_input, f_name), raw_data=raw_data, data_range=data_range)
            # util.save_img(
            #     normal_img, '{}/{}_normal_noadjust.png'.format(result_path, idx))
        else:
            result_path_continuous = result_path+'/continuous/'
            os.makedirs(result_path_continuous, exist_ok=True)
            # util.save_img(
            #     gt_img_png, '{}/{}_gt.png'.format(result_path, idx))
            # util.save_img(
            #     normal_img_png, '{}/{}_{}_normal_process.png'.format(result_path, idx))
            for i in range(visuals['HQ'].shape[0]):
                if i == 0:   # idx = 0 is the LA image
                    continue
                util.save_img(
                    Metrics.tensor2img(visuals['HQ'][i], min_max=display_range),
                    '{}/{}_{}_normal.png'.format(result_path_continuous, f_name, i)
                )
            # util.save_img(
            #     Metrics.tensor2img(visuals['HQ'][-1]), '{}/{}_normal.png'.format(result_path_continuous, idx))
            # normal_img_png = Metrics.tensor2img(visuals['HQ'][-1])

        _, pred = compute_measure(ll_img, gt_img, normal_img, data_range=data_range, raw_data=raw_data)

        psnr, ssim, rmse = pred[0], pred[1], pred[2]

        util.save_img(normal_img_png, '{}/{}_normal.png'.format(result_path_out, f_name))
        util.save_npy(normal_img, '{}/{}_normal.npy'.format(result_path_out, f_name), raw_data=raw_data, data_range=data_range)

        # lpips - handle both grayscale (2D) and RGB (3D) images
        if normal_img_png.ndim == 2:
            # Single channel image, expand to 3 channels for LPIPS
            img_hq = np.stack([normal_img_png/255] * 3, axis=0)  # (3, H, W)
            img_gt = np.stack([gt_img_png/255] * 3, axis=0)  # (3, H, W)
        else:
            # Multi-channel image (3D: H, W, C)
            img_hq = np.transpose(normal_img_png/255, (2, 0, 1))  # (C, H, W)
            # img_gt = np.transpose(gt_img_png/255, (2, 0, 1))  # (C, H, W)
            img_gt = np.stack([gt_img_png/255] * 3, axis=0)  # (3, H, W)

        
        img_hq = transform(torch.from_numpy(img_hq).unsqueeze(0))
        img_gt = transform(torch.from_numpy(img_gt).unsqueeze(0))
        lpips_ = loss_fn_vgg(img_hq.to(torch.float32), img_gt.to(torch.float32))
        
        # lpips_ = loss_fn_vgg(visuals['HQ'], visuals['GT'])

        lpips_value = lpips_.detach().numpy()[0][0][0][0]
        logger_val.info('### {} pred-PSNR: {:.4f} pred-SSIM: {:.4f} pred-RMSE: {:.4f} LPIPS: {:.4f}'.format(idx, psnr, ssim, rmse, lpips_value))
        
        # Store metric values
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        rmse_values.append(rmse)
        lpips_values.append(lpips_value)
        
        # Break after processing 5 samples
        # if idx >= 5:
        #     logger_val.info(f'Processed {idx} samples, breaking...')
        #     break

    # Calculate mean and std
    mean_psnr = np.mean(psnr_values) if psnr_values else 0.0
    std_psnr = np.std(psnr_values) if psnr_values else 0.0
    mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
    std_ssim = np.std(ssim_values) if ssim_values else 0.0
    mean_rmse = np.mean(rmse_values) if rmse_values else 0.0
    std_rmse = np.std(rmse_values) if rmse_values else 0.0
    mean_lpips = np.mean(lpips_values) if lpips_values else 0.0
    std_lpips = np.std(lpips_values) if lpips_values else 0.0

    # log with mean ± std format
    logger_val.info('# Validation # PSNR: {:.4f} ± {:.4f} SSIM: {:.4f} ± {:.4f} RMSE: {:.4f} ± {:.4f} LPIPS: {:.4f} ± {:.4f}'.format(
        mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, mean_lpips, std_lpips))



if __name__ == '__main__':
    main()
