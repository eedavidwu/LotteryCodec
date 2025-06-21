import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

from asyncio import base_tasks
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import argparse
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
from torch import Tensor, index_select, nn
from models.model import Masked_INR,ModConv
from utils.quantizer import quantize
from utils.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from utils.upsampling import Upsampling
from utils.eval_model import eval_model,compute_model_rate

from utils.quantizemodel import quantize_model
from enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict


# manual_seed=1
# def seed_everything(seed=1029):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     os.environ['PATHONHASHSEED'] = str(seed)
#     #torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

# seed_everything(1)

# print('seed',manual_seed)

def get_mgrid(w_sidelen,h_sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = torch.linspace(-1, 1, steps=w_sidelen)  # Linspace for width
    y = torch.linspace(-1, 1, steps=h_sidelen)  # Linspace for height
    tensors = (x, y) if dim == 2 else (x, ) * dim  # Extend for higher dimensions if needed
    # Create the grid and stack along the last dimension
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1).permute(1,0,2)   # Use 'ij' indexing for (w, h) shape
    mgrid = mgrid.reshape(1, -1, dim)  # Reshape to [1, w*h, dim] for a 2D image
    return mgrid


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    return  0

def loss_to_psnr(loss, max=1):
  return 10*np.log10(max**2/np.asarray(loss))

def train(model,dataloader, total_steps, total_steps_2,steps_til_summary,img_index,saved_path):
    # plot_gradient consumes way more memory
    #vis_colum=3
    best_psnr=0
    criterion = nn.MSELoss().cuda()
    base_params = [p for name, p in model.named_parameters() if p.requires_grad and "scores" not in name]
    score_params = [p for name, p in model.named_parameters() if p.requires_grad and "scores" in name]
    optim = torch.optim.Adam([{'params': base_params, 'lr': args.lr},{'params': score_params, 'lr': 0.1}])
    #optim = Lookahead(optim, alpha=0.5,k=6)

    scheduler = CosineAnnealingLR(optim, T_max=total_steps)
    optimizer_stage_2 =torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scheduler_stage_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage_2, mode='min', factor=0.8, patience=20, verbose=True)

    for batch_idx, (img_in,_) in enumerate(dataloader, 0):
        model.train()
        batch_size,_,height,width=img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size,-1, 3).cuda()
        #vutils.save_image(img_in,'./saved/gt_'+str(img_index)+'.png',nrow=vis_colum)
        coords = get_mgrid(width//args.scale,height//args.scale, 2).repeat(batch_size,1,1).cuda()
        losses = []
        losses_2 = []
        best_psnr=0
        ####linear decay the noise para and tempertature
        initial_noise_param = 2.0
        final_noise_param = 1.0
        initial_temperature = 0.3
        final_temperature = 0.1
        print('start temperature:', initial_temperature,'noise parameter:',initial_noise_param)
        print('end temperature:', final_temperature,'noise parameter:',final_noise_param)

        print("********************Start from stage I")
        #model.soft_round_temperature=0.3
        #model.noise_parameter=2
        #print('*******************Switch into stage I beginning phase')
        best_rd=1000

        for step in range(total_steps+1):
            ###stage 1:
            model.train()
            # Linear decay of noise_parameter and soft_round_temperature
            model.noise_parameter = initial_noise_param - (step / total_steps) * (initial_noise_param - final_noise_param)
            model.soft_round_temperature = initial_temperature - (step / total_steps) * (initial_temperature - final_temperature)
            #print('temperature:', model.soft_round_temperature,'noise parameter:',model.noise_parameter, 'at', step)
            model_output,rate,_ = model(coords)   
            model_output=model_output.view(batch_size,-1,3)
            bits_rate=rate.sum()/(width*height)
            loss_mse=criterion(model_output,pixels)
            loss=args.lambda_rate*bits_rate+loss_mse
            #losses.append(loss.item())
            if step % 5000 ==0:
                psnr_this_iter=loss_to_psnr(loss_mse.item())
                print("Print step %d, PSNR: %0.6f, Total loss %0.6f" % (step, psnr_this_iter,loss),'with its rate', bits_rate.item())
            if not step % steps_til_summary or (step==total_steps-1):
                psnr_this_iter=loss_to_psnr(loss_mse.item())
                #psnr_this_iter=loss_to_psnr(loss.item())
                #print("Step %d, PSNR: %0.6f, Total loss %0.6f" % (step, psnr_this_iter,loss))
                if (loss<best_rd) and (step>0):
                    best_psnr=psnr_this_iter
                    best_rd=loss
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'binary mask': None
                    }
                    #img_out=model_output.view(batch_size,height,width,3).permute(0,3,1,2)
                    print("Step %d, BEST RD PSNR: %0.6f, Total loss %0.6f" % (step, psnr_this_iter,loss),'with its rate', bits_rate.item())
            optim.zero_grad()
            loss.backward()
            # print('weight:', model.net[0].weight[0:10,0])
            # print('score', model.net[0].scores[0,0:10])
            #nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10,norm_type=2.0, error_if_nonfinite=False)
            #nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10, norm_type=2.0, error_if_nonfinite=False)
            optim.step()
            scheduler.step()
            #torch.cuda.empty_cache()
        #torch.cuda.empty_cache()
        torch.save(checkpoint, saved_path)
        checkpoints=torch.load(saved_path)
        model.load_state_dict(checkpoints['model_state_dict'])
   
        
        ###stage_2
        print("********************going into stage II")
        best_psnr_2=0
        best_rd_2=1000

        for step in range(total_steps_2):
            model.train()
            model.quantizer_type="softround_alone"
            model.quantizer_noise_type="none"
            model.soft_round_temperature=1e-4
            #model.noise_parameter=1 ##1 means uniform....
            model_output,rate,_ = model(coords)   
            model_output=model_output.view(batch_size,-1,3)
            bits_rate=rate.sum()/(width*height)
            loss_mse=criterion(model_output,pixels)
            loss_2=args.lambda_rate*bits_rate+loss_mse
            losses_2.append(loss_2.item())
            if not step % steps_til_summary or (step==total_steps_2-1):
                psnr_this_iter=loss_to_psnr(loss_mse.item())
                if (loss_2<best_rd_2) and (step>0):
                        best_psnr_2=psnr_this_iter
                        best_rd_2=loss_2
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'binary mask': None
                        }
                        print('Print rate', bits_rate)
                        print('latent_bits', rate.sum().item())
                        print("Step %d, BEST PSNR: %0.6f, Total loss %0.6f" % (step, psnr_this_iter,loss_2))
            optimizer_stage_2.zero_grad()
            loss_2.backward()
            optimizer_stage_2.step()
            scheduler_stage_2.step(loss_2)
            #torch.cuda.empty_cache()
            current_lr = optimizer_stage_2.param_groups[0]['lr']
            # Check if the learning rate has fallen below the threshold
            if current_lr < 1e-8:
                print(f"Current learning rate: {current_lr}")
                print(f"Stopping training early: Learning rate has dropped below lr_threshold")
                break 

        torch.cuda.empty_cache()
        torch.save(checkpoint, saved_path)
        checkpoints=torch.load(saved_path)
        print('Saved model at',saved_path)

        model.load_state_dict(checkpoints['model_state_dict'])
        #eval:
        model.eval()
        model_output,rate,binary_mask = model(coords)   
        model_output=model_output.view(batch_size,-1,3)
        bits_rate_eval=rate.sum()/(width*height)
        loss_mse=criterion(model_output,pixels)
        psnr_eval=loss_to_psnr(loss_mse.item())
        print("********************Evaluation the Image %d-th, after Step %d, BEST PSNR: %0.6f, Print rate %0.6f. *************************" % (img_index,step, psnr_eval,bits_rate_eval.item()))
        torch.cuda.empty_cache()
    return psnr_eval,bits_rate_eval.item()

global args
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=1, help='Batch-size')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate')
parser.add_argument('--data', type=str, default='../data', help='Location to store data')
parser.add_argument('--sparsity', type=float, default=0.3, help='prune rate')
parser.add_argument('--upsampling_kernel_size', type=int, default=8, help='2, 4 or 8')
parser.add_argument('--static_upsampling_kernel', default=False, help='Use this flag to **not** learn the upsampling kernel')
parser.add_argument('--latent_factor', type=int, default=1, help='Full resolution -> 1, other W,H/factor')
parser.add_argument('--mod_base', type=int, default=7, help='Number of base')

parser.add_argument('--highest_flag', type=int, default=1, help='Full resolution -> 1, other W,H/factor')
parser.add_argument('--context_arm', type=int, default=24, help='8,16,24,32')
parser.add_argument('--dim_arm_mod', type=int, default=24, help='arm dimension')
parser.add_argument('--flag_A', type=int, default=0, help='0: 0-41; 1: 0-20; 2: 20-41')
parser.add_argument('--mod_hid_layer', type=int, default=0, help='3x3 mod layer')
parser.add_argument('--hidden_features', type=int, default=64, help='hidden')
parser.add_argument('--hidden_layer', type=int, default=2, help='layer')
parser.add_argument('--scale', type=int, default=1, help='Predict every scale*1 pixel')
parser.add_argument('--lambda_rate', type=float, default=1e-3, metavar='LR',help='weight')

args = parser.parse_args()

all_psnr=[]
all_rate=[]
mask_rate=[]
eval_all_psnr=[]
eval_all_y_rate=[]
eval_all_mlp_rate=[]

for it in range(0,2):
    val_folder='./dataset/CLIC2020/img_'+str(it)
    transform_val = transforms.Compose([
            transforms.ToTensor() ])
    val_dataset = datasets.ImageFolder(val_folder,transform_val)
    dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=2, pin_memory=True)
    img_in, _ = next(iter(dataloader))
    args.patch_h=img_in.shape[2]
    args.patch_w=img_in.shape[3]

    print(args)
    folder_path='./saved/context_'+str(args.context_arm)+'_arm_mod_'+str(args.dim_arm_mod)
    make_path(folder_path)
    folder_path_=folder_path+'/sparity_'+str(args.sparsity)
    make_path(folder_path_)
    saved_path=folder_path_+'/inr_mod_'+str(args.dim_arm_mod)+'_'+str(args.mod_hid_layer)+'_pw_'+str(args.lambda_rate)+'_img_'+str(it)+'.pth'
    total_steps =100000
    total_steps_2=10000
    steps_til_summary = 10
    print('top %:',args.sparsity)
    mask_model = Masked_INR(args,sparsity=args.sparsity,in_features=2, out_features=3*args.scale*args.scale, hidden_features=args.hidden_features, hidden_layers=args.hidden_layer)
    entropy_each_item= (-args.sparsity*np.log2(args.sparsity)-(1-args.sparsity)*np.log2(1-args.sparsity))

    print('bit each para',entropy_each_item)
    all_w=0
    for i in range(len(mask_model.net)):
        w_in,w_out=mask_model.net[i].weight.shape
        all_w=all_w+w_in*w_out
    num_para=all_w
    para_num=num_para*entropy_each_item
    bpp_mask_each=para_num/args.patch_h/args.patch_w
    print('bits for all paramter:',para_num,'with bpp',bpp_mask_each)

    num_channel=args.patch_h*args.patch_w//args.scale//args.scale
    print(mask_model)
    mask_model.cuda()
    print('train the',it,'-th image')
    out_psnr,out_rate=train(mask_model, dataloader, total_steps, total_steps_2,steps_til_summary,it,saved_path)
    all_psnr.append(out_psnr)
    all_rate.append(out_rate)
    mask_rate.append(bpp_mask_each)
    print('Trained the image with PSNR:',out_psnr,' latent bits',out_rate)
    print(all_psnr)
    print(all_rate)
    print(mask_rate)
    ###eval:
    ###load model:
    checkpoints=torch.load(saved_path)
    mask_model.load_state_dict(checkpoints['model_state_dict'])
    binary_mask=checkpoints['binary mask']
    print('load the model:',saved_path, 'for the ',it,'-th image')
    mask_model.cuda()
    mask_model.eval()
    eval_out_psnr,eval_y_rate,eval_network_rate=eval_model(args,mask_model, binary_mask,dataloader,it)
    eval_all_psnr.append(eval_out_psnr)
    eval_all_y_rate.append(eval_y_rate)
    eval_all_mlp_rate.append(eval_network_rate)
    eval_all_rate_y_mlp_latent=[y + mlp+np.array(mask_rate)  for y, mlp in zip(eval_all_y_rate, eval_all_mlp_rate)]
    print('Evaluate the image: PSNR:',eval_out_psnr,'All bits:',eval_all_rate_y_mlp_latent[-1],' latent bits',eval_y_rate,' network bits',eval_network_rate)
    print(eval_all_psnr)
    print(eval_all_rate_y_mlp_latent)
    print('Current eval Ave PSNR:',np.mean(eval_all_psnr),'Ave Bits',np.mean(eval_all_rate_y_mlp_latent),'Ave Bits for mask',np.mean(mask_rate))

print('.......Copmlete all dataset training......')
print('Ave Training PSNR:',np.mean(all_psnr),'Ave Training Bits',np.mean(all_rate))
print('Training PSNR:', all_psnr)
print('Training rate:', all_rate)

print('Evalutation: Ave Eval PSNR:',np.mean(eval_all_psnr),'Ave Eval all bits:',np.mean(eval_all_rate_y_mlp_latent),'Ave Eval Latent Bits',np.mean(eval_all_y_rate),'Ave Eval Network',np.mean(eval_all_mlp_rate),'Ave Mask',np.mean(mask_rate))
print('Eval All PSNR:',eval_all_psnr)
print('Eval All rate',eval_all_rate_y_mlp_latent)
print('Eval Latent rate',eval_all_y_rate)
print('Eval MLP rate',eval_all_mlp_rate)
print('Eval Mask rate',mask_rate)