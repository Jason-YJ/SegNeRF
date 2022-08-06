import os, sys
sys.path.append('..')
import numpy as np
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import cv2
import pdb


from models.nerf.run_nerf_helpers import *
from options import config_parser
from .load_llff import load_llff_data
from utils.io_utils import *
from utils.nerf_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])       # 65536, 3
    embedded = embed_fn(inputs_flat)   # positional encoding   65536, 63

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)   # 1024, 64, 3
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # 65536, 3
        embedded_dirs = embeddirs_fn(input_dirs_flat)   # positional encoding   torch.Size([65536, 27])
        embedded = torch.cat([embedded, embedded_dirs], -1)   # torch.Size([65536, 90])

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, depth_priors=None, depth_confidences=None, priorback=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], depth_priors=depth_priors[i:i+chunk], depth_confidences=depth_confidences[i:i+chunk], priorback=priorback, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, depth_priors=None, depth_confidences=None, priorback=False,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o_ori = rays_o.clone()
        rays_d_ori = rays_d.clone()
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    depth_priors = torch.reshape(depth_priors, [-1]).float()
    depth_confidences = torch.reshape(depth_confidences, [-1]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, depth_priors, depth_confidences, priorback=priorback, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if ndc:
        print("ndc!!!")
        all_ret['depth_map'] = -1/rays_d_ori[:, 2]*(1 / (1 - all_ret['depth_map']) + rays_o_ori[:, 2])
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map', 'label_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, image_list, sc,
                depth_priors=None, depth_confidences=None, savedir=None, render_factor=0, priorback=False):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    depths = []
    labels = []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, depth, label, extras = render(H, W, focal, depth_priors=depth_priors[i], depth_confidences=depth_confidences[i], chunk=chunk, c2w=c2w[:3,:4], priorback=priorback, **render_kwargs)

        label = torch.argmax(label, dim=2)
        rgbs.append(rgb.cpu().numpy())
        labels.append(label.cpu().numpy())
        depths.append(depth)
        if i==0:
            print(rgb.shape, disp.shape)


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            frame_id = image_list[i].split('.')[0]

            filename = os.path.join(savedir, '{}.png'.format(frame_id))
            imageio.imwrite(filename, rgb8)

            filename = os.path.join(savedir, '{}_label.png'.format(frame_id))
            cv2.imwrite(filename, labels[-1])

            filename = os.path.join(savedir, '{}_depth.npy'.format(frame_id))
            np.save(filename, depth.cpu().numpy() / sc)

            # import pyexr
            # exr_name = os.path.join(savedir, '{}_depth.exr'.format(frame_id))
            # pyexr.write(exr_name, depth.cpu().numpy() / sc)
            disp_visual = visualize_depth(depth.cpu().numpy())
            filename = os.path.join(savedir, '{}_depth.png'.format(frame_id))
            cv2.imwrite(filename, disp_visual)



    rgbs = np.stack(rgbs, 0)
    depths = torch.stack(depths, 0)
    labels = np.stack(labels, 0)

    return rgbs, depths.cpu().numpy(), labels



def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [3]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        #start = 0
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (start / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    else:
        ckpt_path = os.path.join(basedir, expname, 'nerf', 'checkpoints')
        ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'near_bound': args.near,
        'far_bound': args.far,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    label = raw[..., 4:]  # [N_rays, N_samples, N_class]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]  # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, N_samples, 3] -> [N_rays, 3]
    label_map = torch.sum(weights[..., None] * label, -2)  # [N_rays, N_samples, N_class] -> [N_rays, N_class]
    depth_map = torch.sum(weights * z_vals, -1)   # [N_rays, N_samples] -> [N_rays]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, rgb, label_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                depth_priors,
                depth_confidences,
                retraw=False,
                lindisp=False,
                perturb=0.,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                near_bound=None,
                far_bound=None,
                priorback=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    #print("depth_priors.shape: ", depth_priors.shape)
    #print("depth_priors: ", depth_priors)
    # depth_priors = depth_priors * (1 + torch.clamp(10 * depth_confidences, min=0, max=0.5))
    if priorback == True:
        near = (depth_priors * (1 - torch.clamp(depth_confidences, min=near_bound, max=near_bound))).unsqueeze(1)
        far = (depth_priors * (1 + torch.clamp(30 * depth_confidences, min=near_bound, max=0.5))).unsqueeze(1)
    else:
        near = (depth_priors * (1 - torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(1)
        far = (depth_priors * (1 + torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(1)
    #print("near: ", near)
    #print("far: ", far)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    #print("z_vals.shape: ", z_vals.shape)
    #print("z_vals: ", z_vals)
    z_vals = z_vals.expand([N_rays, N_samples])
    #print("z_vals.shape: ", z_vals.shape)
    #print("z_vals: ", z_vals)
    
    
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            print("pytest")
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, rgb, label_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, "weights" : weights, "label_map": label_map}
    if retraw:
        ret['raw'] = raw
    
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def train(args):
    print('Nerf begins !')
    # Load data  
    images, poses, bds, render_poses, i_train, i_test, sc = load_llff_data(args.datadir, args.factor,   # img是train+test
                                                              recenter=True, bd_factor=.75,
                                                              spherify=args.spherify, N_views=args.N_views)

    if len(i_test) > 0:
        load_test = True
    else:
        load_test = False
        
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    print("H, W : ", H, W)
    image_list = load_img_list(args.datadir, load_test=load_test)
    colmap_depths, colmap_masks = load_colmap(image_list, args.datadir, H, W)
   
    image_list_train = load_img_list(args.datadir, load_test=False)
    depth_priors = load_depths(image_list_train,
                        os.path.join(args.basedir, args.expname, 'depth_priors', 'results'), 
                        H, W)

    depth_priors = align_scales(depth_priors, colmap_depths, colmap_masks,   # train35+test5
                                poses, sc, i_train, i_test)
    

    image_list = load_img_list(args.datadir, load_test=True)
    segmentation = load_seg_opencv(image_list, os.path.join(args.datadir, 'segmentation'),  # train35+test5
                       H, W)
    
    poses_tensor = torch.from_numpy(poses).to(device)
    K = torch.FloatTensor([[focal, 0, -W / 2.0, 0],
                           [0, -focal, -H / 2.0, 0],
                           [0,  0,  -1, 0],
                           [0,  0,  0, 1]]).to(device)
    if poses_tensor.shape[1] == 3:
        bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
        bottom = bottom.repeat(poses_tensor.shape[0], 1, 1).to(poses_tensor.device)
        T = torch.cat([poses_tensor, bottom], 1)
    else:
        T = poses_tensor.clone()

    depth_confidences = cal_depth_confidences(torch.from_numpy(depth_priors).to(device),   # train35+test5
                                              T, K, i_train, args.topk)

    if args.remove == True:
        print("Removing the 5th poor prior")
        priors_sum = np.sum(depth_confidences[:35], axis=(1, 2))
        remove = np.argpartition(priors_sum, 30)[-5:]
        i_train = np.delete(i_train, remove)
        with open(os.path.join(args.datadir, 'train.txt'), 'r') as f:
            lines = f.readlines()
            ori_train_list = [line.strip() for line in lines]
        ori_train_list = np.array(ori_train_list)
        train_nerf_list = ori_train_list[i_train]
        with open(os.path.join(args.datadir, 'train_nerf.txt'), 'w') as f:
            for i in train_nerf_list:
                f.write(str(i))
                f.write('\n')

    # print("saving confidence map....")
    # import pyexr
    # confsavedir = os.path.join(args.basedir, args.expname, 'nerf', "confidence_map")
    # os.makedirs(confsavedir, exist_ok=True)
    # for i in range(len(image_list)):
    #     frame_id = image_list[i].split('.')[0]
    #     pyexr.write(os.path.join(confsavedir, '{}_conf.exr'.format(frame_id)), depth_confidences[i])
    #     depth_color1 = visualize_depth(depth_confidences[i])
    #     cv2.imwrite(os.path.join(confsavedir, '{}_conf.png'.format(frame_id)), depth_color1)

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)


    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    save_path = os.path.join(args.basedir, args.expname, 'nerf')

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                testsavedir = os.path.join(save_path, 'results', 
                                           'renderonly_{}_{:06d}'.format('test', start))
                render_poses = poses_tensor[i_test]
                depth_priors = depth_priors[i_test]
                depth_confidences = depth_confidences[i_test]
                image_list = image_list[i_test]
            else:
                testsavedir = os.path.join(save_path, 'results', 
                                           'renderonly_{}_{:06d}'.format('train', start))
                render_poses = poses_tensor[i_train]
                depth_priors = depth_priors[i_train]
                depth_confidences = depth_confidences[i_train]
                image_list = image_list[i_train]

            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths, labels = render_path(render_poses, hwf, args.chunk, render_kwargs_test, sc=sc,
                                          depth_priors=torch.from_numpy(depth_priors).to(device),
                                          depth_confidences=torch.from_numpy(depth_confidences).to(device), 
                                          savedir=testsavedir, render_factor=args.render_factor, priorback=args.priorback,
                                          image_list=image_list)
            print('Done rendering', testsavedir)
    
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0)  # [40, ro+rd, H, W, 3]此时仍按是train+test
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [40, ro+rd+rgb, H, W, 3]   40张图的ro就是pose所以相同，rd共有40 * 484 * 648种
    depths_pri_and_seg = np.stack([depth_priors,  depth_confidences.astype(np.float32),
                               segmentation], -1)  # [40, H, W, 3]
    rays_rgb = np.concatenate([rays_rgb, depths_pri_and_seg[:, None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])  # [40, H, W, ro+rd+rgb(d)+prior, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only [35-5, H, W, ro+rd+rgb(d)+prior, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,4,3]) 

    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    i_batch = 0

    # Move training data to GPU
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    print(rays_rgb.shape)


    N_iters = args.N_iters
    
    print(args.chunk)
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)

    # Summary writers
    writer = SummaryWriter(os.path.join(args.basedir, args.expname, 'nerf', 'summary'))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand]  # [1024, 4, 3]
        batch = torch.transpose(batch, 0, 1)  # [4, 1024, 3]
        batch_rays, target_s = batch[:2], batch[2]
        target_prior = batch[3]
        depth_prior = target_prior[:, 0]
        depth_confidence = target_prior[:, 1]
        seg_target = target_prior[:, 2].long()
        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, label, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                               depth_priors=depth_prior,
                                               depth_confidences=depth_confidence,
                                               retraw=True, priorback=args.priorback, **render_kwargs_train)

        ce = nn.CrossEntropyLoss(ignore_index=255)
        seg_loss = ce(label, seg_target)
        img_loss = img2mse(rgb, target_s)
        loss = img_loss + args.lossrate * seg_loss

        psnr = mse2psnr(img_loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(save_path, 'checkpoints', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} seg_loss：{seg_loss.item()} img_loss: {img_loss.item()} PSNR: {psnr.item()} lossrate: {args.lossrate}")
            writer.add_scalar("seg_loss", seg_loss.item(), i)
            writer.add_scalar("Loss", loss.item(), i)
            writer.add_scalar("PSNR", psnr.item(), i)
        
        global_step += 1
    
    with torch.no_grad():
        testsavedir = os.path.join(save_path, 'results')
        render_poses = poses_tensor

        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths, label = render_path(render_poses, hwf, args.chunk, render_kwargs_test, sc=sc,
                                          depth_priors=torch.from_numpy(depth_priors).to(device),
                                          depth_confidences=torch.from_numpy(depth_confidences).to(device), 
                                          savedir=testsavedir, render_factor=args.render_factor, 
                                          image_list=image_list, priorback=args.priorback)
        print('Done rendering', testsavedir)
    

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)
