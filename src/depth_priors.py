import os, sys
sys.path.append('..')
import numpy as np
import torch
import cv2
# import pyexr
from utils.evaluation_utils import *

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pdb

from pytorch_msssim import MS_SSIM, SSIM
from models.depth_priors.mannequin_challenge_model import MannequinChallengeModel
from models.depth_priors.unet_multi_scale import UNet_multi_scale
from options import config_parser
from utils.io_utils import *
from utils.depth_priors_utils import *
import itertools
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_depth_model(args):
    """Instantiate depth model.
    """
    depth_model = MannequinChallengeModel()
    refine_model = UNet_multi_scale(2, 1).cuda()
    grad_vars = depth_model.parameters()
    refine_vars = refine_model.parameters()
    optimizer = torch.optim.Adam(itertools.chain(grad_vars, refine_vars), lr=args.depth_lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    ckpt_path = os.path.join(basedir, expname, 'depth_priors', 'checkpoints')

    # Load checkpoints
    ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        
        new_lrate = args.depth_lrate * (decay_rate ** (start / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        depth_model.model.netG.load_state_dict(ckpt['netG_state_dict'])
        refine_model.load_state_dict(ckpt['refine_model_state_dict'])

    return depth_model, refine_model, start, optimizer



def train(args):
    print('Depths prior training begins !')
    image_list = load_img_list(args.datadir)
    depth_model, refine_model, global_step_depth, optimizer = create_depth_model(args)

    # Summary writers
    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')
    writer = SummaryWriter(os.path.join(save_dir, 'summary'))

    segmentation_train = load_seg(image_list, os.path.join(args.datadir, 'segmentation'),
                       args.depth_H, args.depth_W)

    images = load_rgbs(image_list, os.path.join(args.datadir, 'images'),
                       args.depth_H, args.depth_W)
    images_train = images.clone()

    depths, masks = load_colmap(image_list, args.datadir,   # (288, 384)
                                args.depth_H, args.depth_W)

    # 输出用以指导的稀疏深度图
    # d1 = depths
    # m1 = masks
    # import pyexr
    # sparse_depth_path = os.path.join(save_dir, "sparse_depth")
    # os.makedirs(sparse_depth_path, exist_ok=True)
    # for i in range(len(image_list)):
    #     frame_id = image_list[i].split('.')[0]
    #     pyexr.write(os.path.join(sparse_depth_path, '{}_depth.exr'.format(frame_id)), d1[i])
    #     depth_color1 = visualize_depth(d1[i], m1[i])  # (288, 384)
    #     cv2.imwrite(os.path.join(sparse_depth_path, '{}_depth.png'.format(frame_id)), depth_color1)
    #     cv2.imwrite(os.path.join(sparse_depth_path, '{}_mask.png'.format(frame_id)), m1[i] * 255)

    depths_train = torch.from_numpy(depths).to(device)   # torch.Size([35, 288, 384])
    depths_mask_train = torch.from_numpy(masks).to(device)   # torch.Size([35, 288, 384])
    N_rand_depth = args.depth_N_rand
    N_iters_depth = args.depth_N_iters

    i_batch = 0
    depth_model.train()
    start = global_step_depth + 1

    for i in trange(start, N_iters_depth):
        batch = images_train[i_batch:i_batch + N_rand_depth].to(device)
        depth_gt, mask_gt = depths_train[i_batch:i_batch + N_rand_depth], depths_mask_train[i_batch:i_batch + N_rand_depth]
        depth_pred = depth_model(batch)

        loss1 = compute_depth_loss(depth_pred, depth_gt, mask_gt)
        seg = segmentation_train[i_batch:i_batch + N_rand_depth].to(device)
        refine_input = torch.cat([depth_pred[:, None], seg], dim=1)
        refine_pred = refine_model(refine_input)
        loss2 = compute_depth_loss(refine_pred.squeeze(), depth_gt, mask_gt)
        loss = loss1 + 1.5 * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        new_lrate = args.depth_lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
        i_batch += N_rand_depth


        if i_batch >= images_train.shape[0]:

            print("Shuffle depth data after an epoch!")
            rand_idx = torch.randperm(images_train.shape[0])
            images_train = images_train[rand_idx]
            depths_train = depths_train[rand_idx]
            depths_mask_train = depths_mask_train[rand_idx]
            i_batch = 0

        if i % args.depth_i_weights == 0:
            path = os.path.join(save_dir, 'checkpoints', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'netG_state_dict': depth_model.model.netG.state_dict(),
                'refine_model_state_dict': refine_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.depth_i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")

        global_step_depth += 1
    print('depths prior training done!')

    with torch.no_grad():
        depth_model.eval()
        refine_model.eval()
        for i, image_name in enumerate(image_list):
            frame_id = image_name.split('.')[0]
            batch = images[i:i + 1].to(device)
            depth_pred = depth_model.forward(batch).cpu().numpy()   # (288, 384)
            depth_color = visualize_depth(depth_pred)               # (288, 384, 3)
            cv2.imwrite(os.path.join(save_dir, 'results', '{}_depth.png'.format(frame_id)), depth_color)
            # pyexr.write(os.path.join(save_dir, 'results', '{}_depth.exr'.format(frame_id)), depth_pred)
            np.save(os.path.join(save_dir, 'results', '{}_depth.npy'.format(frame_id)), depth_pred)

    print('results have been saved in {}'.format(os.path.join(save_dir, 'results')))

    image_list = load_img_list(args.datadir, load_test=False)
    prior_path = os.path.join(args.basedir, args.expname, 'depth_priors', 'results')
    prior_depths = load_depths(image_list, prior_path)
    gt_depths, _ = load_gt_depths(image_list, args.datadir)
    print("prior depth evaluation:")
    depth_evaluation(gt_depths, prior_depths, savedir=prior_path)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    print(args.datadir)
    train(args)
