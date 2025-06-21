import os
from asyncio import base_tasks
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import argparse
from torch import Tensor, index_select, nn
import random
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import torchvision.transforms as transforms
#import rff
from torchvision import datasets, transforms
import torchvision.utils as vutils
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from utils.quantizer import quantize
from utils.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from utils.upsampling import Upsampling
from utils.quantizemodel import quantize_model
from enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#idea: GAN, encoded_latent

manual_seed=1
def seed_everything(seed=1029):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PATHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(1)

    
def loss_to_psnr(loss, max=1):
  return 10*np.log10(max**2/np.asarray(loss))

def compute_model_rate(model):
    rate_mlp = 0.0
    rate_per_module = model.get_network_rate()
    for _, module_rate in rate_per_module.items():
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate
    return rate_mlp
def get_mgrid(w_sidelen,h_sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = torch.linspace(-1, 1, steps=w_sidelen)  # Linspace for width
    y = torch.linspace(-1, 1, steps=h_sidelen)  # Linspace for height
    tensors = (x, y) if dim == 2 else (x, ) * dim  # Extend for higher dimensions if needed
    # Create the grid and stack along the last dimension
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1).permute(1,0,2)  # Use 'ij' indexing for (w, h) shape
    mgrid = mgrid.reshape(1, -1, dim)  # Reshape to [1, w*h, dim] for a 2D image
    return mgrid
def eval_model(args,model,binary_mask,dataloader,img_index):
    # plot_gradient consumes way more memory
    criterion = nn.MSELoss().cuda()

    for batch_idx, (img_in,_) in enumerate(dataloader, 0):
        batch_size,_,height,width=img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size,-1, 3).cuda()
        coords = get_mgrid(width,height, 2).repeat(batch_size,1,1).cuda()
        print("********************Evalutation with quantization")
        print("********************Starting quantizing models")
        model_q=quantize_model(model,binary_mask,coords,pixels,args)
        model=model_q

        torch.cuda.empty_cache()
        #eval:
        model.eval()
        model_output,rate,binary_mask = model(coords,binary_mask)   
        model_output=model_output.view(batch_size,-1,3)
        bits_rate_eval=rate.sum()/(width*height)
        loss_mse=criterion(model_output,pixels)
        psnr_eval=loss_to_psnr(loss_mse.item())
        out_network_rate=compute_model_rate(model)/(width*height)

        print("********************Evaluation the Image %d-th, BEST PSNR: %0.6f, Print rate %0.6f, Network rate %0.6f. *************************" % (img_index, psnr_eval,bits_rate_eval.item(),out_network_rate))
        img_out=model_output.view(batch_size,height,width,3).permute(0,3,1,2)
        #vutils.save_image(img_out,'./eval_'+str(img_index)+'.png',nrow=1)
        torch.cuda.empty_cache()

    return psnr_eval,bits_rate_eval.item(),out_network_rate.item()

def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = (2.*np.pi*x) @ B.T
    embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
    return embedding
