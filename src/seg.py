import os, sys
# sys.path.append('..')
# import numpy as np
# import torch
# import cv2
# import pyexr
# from utils.evaluation_utils import *
#
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm, trange
# import pdb
#
# from pytorch_msssim import MS_SSIM, SSIM
# from models.depth_priors.mannequin_challenge_model import MannequinChallengeModel
# from models.depth_priors.unet_multi_scale import UNet_multi_scale
# from options import config_parser
# from utils.io_utils import *
# from utils.depth_priors_utils import *
# import itertools
# import torch.nn as nn
#
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'

def seg(args):
    input_file = os.path.join(args.datadir, "images")
    save_folder = os.path.join(args.datadir, "segmentation")
    os.makedirs(save_folder, exist_ok=True)
    config_fpath = "mseg-semantic/mseg_semantic/config/test/default_config_360_ms.yaml"
    os.system("python -u mseg-semantic/mseg_semantic/tool/universal_demo.py --config=%s --file_save 2 model_name mseg-3m model_path mseg-3m.pth input_file %s save_folder %s" %
              (config_fpath, input_file, save_folder))
