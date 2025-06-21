import os
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

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


       
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class GetSubnet_batch(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        
        batch_size,w1,w2=scores.shape
        score_reshape=scores.view(batch_size,-1)
        _, indices = torch.sort(score_reshape, dim=1, descending=True)
        #_, idx = score_reshape.sort(-1)
        j = int((1 - k) * score_reshape.size(1))

        binary_mask = torch.zeros_like(score_reshape)
        binary_mask.scatter_(1, indices[:, :j], 1)
        binary_mask = binary_mask.view(batch_size, w1, w2)
        return binary_mask

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class NonAffineBatchNorm(nn.BatchNorm1d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class SupermaskLinear(nn.Linear):
    def __init__(self,args_all,ffn_flg,res_flag,sparsity, last_flag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.unsqueeze(0).repeat(args_all.batch_size,1,1).size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.res=res_flag
        self.scores.requires_grad = True
        self.sparsity=sparsity
        self.last_flag=last_flag
        #self.modulate_flag=modulate_flag
        #ffn_flg=1
        ###for normal weight:
        if ffn_flg==0:
            self.custom_kaiming_normal_(self.weight,fan_mode="fan_in",scale=0.1)
        ###for FFN weight:
        if ffn_flg==1:
            self.phi_num = 32
            self.alpha=0.05
            self.high_freq_num =64
            self.low_freq_num = 64
            self.bases=self.init_bases()
            self.lamb=self.init_lamb()
            self.weight=nn.Parameter(torch.matmul(self.lamb,self.bases),requires_grad=False)
            print('FFN: phi',self.phi_num,' high freq:,',self.high_freq_num, ' low freq:,',self.low_freq_num)
        self.weight.requires_grad = False


    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)

        return bases

    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator,np.sqrt(6/m)/dominator)
        self.lamb=nn.Parameter(self.lamb,requires_grad=False)
        return self.lamb
    
    def custom_kaiming_uniform_(self,tensor, fan_mode='fan_in', scale=6):
        # Calculate fan based on the selected mode
        fan = torch.nn.init._calculate_correct_fan(tensor, fan_mode)
        # Custom bound based on the scale parameter
        bound = math.sqrt(scale / fan)
        # Apply uniform initialization within the custom bounds
        return torch.nn.init.uniform_(tensor, -bound, bound)
    
    def custom_kaiming_normal_(self,tensor, fan_mode='fan_in', scale=2):
        fan = torch.nn.init._calculate_correct_fan(tensor, fan_mode)
        std = math.sqrt(scale / fan)
        print(std)
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    def forward(self, x,mod,kept_num,mask=None):
        batch_size=x.shape[0]
        if mod!=None:
            x=torch.cat((x,mod),dim=-1)
        if mask==None:
            sparity=1-kept_num/(self.scores.shape[-1]*self.scores.shape[-2])
            subnet = GetSubnet_batch.apply(self.scores, sparity)
            #subnet = GetSubnet_batch.apply(self.scores, sparity)
        else:
            subnet=mask

        w=self.weight.unsqueeze(0).repeat(batch_size,1,1) * subnet
        if self.res==1:  
            #out=torch.sigmoid((torch.bmm(x, w.transpose(1, 2))))#+self.shits
            out=(torch.bmm(x, w.transpose(1, 2)))+x
        else:
            out=((torch.bmm(x, w.transpose(1, 2))))
        return out,subnet


class ModConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        mod_layer,
    ):
        super().__init__()
        self.residual = False
        self.hid_channels=hid_channels
        self.hid_layer=mod_layer
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=1)
        self.conv2_1=nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1)
        self.conv2_2=nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1)

        # -------- Instantiate empty parameters, set by the initialize function
    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.
        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})
    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        """Replace the current parameters of the module with param.
        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def forward(self, x):
        out_0=F.gelu(self.conv1_1(x) )
        out_1=F.gelu(self.conv1_2(out_0))
        #out_2=F.gelu(self.conv1_3((out_1)))
        out_2=F.gelu(self.conv2_1(out_1)+out_1)
        out_3=F.gelu(self.conv2_2(out_2)+out_2)
        return [torch.cat((out_3,out_2),dim=1),torch.cat((out_3,out_2,out_1),dim=1),torch.cat((out_3,out_2,out_1,out_0),dim=1)]

        
class Masked_INR(nn.Module):
    def __init__(self, args,sparsity,in_features,  out_features, hidden_features, hidden_layers):
        super().__init__()
        self.sparsity=sparsity
        self.net = []
        #patch_size:
        self.h=args.patch_h
        self.w=args.patch_w
        self.pe_flag=0
        self.upsampling_2d = Upsampling(
            args.upsampling_kernel_size, args.static_upsampling_kernel,args.highest_flag
        )
        self.dim_arm=args.dim_arm_mod
        self.n_hidden_layers_arm=2
        self.arm = Arm(args.context_arm,args.dim_arm_mod, self.n_hidden_layers_arm)

        self.quantizer_type="softround"
        self.quantizer_noise_type="kumaraswamy"
        self.soft_round_temperature=0.3
        self.noise_parameter=2.0 ##1 means uniform....
        max_mask_size = 9
        ####modulation
        self.modulation_base_number=args.mod_base
        #self.conv1_1_weight = nn.Conv2d(in_channels=9, out_channels=hidden_layers+2, kernel_size=1)
        self.conv_mod=ModConv(in_channels=self.modulation_base_number, hid_channels=args.dim_arm_mod,out_channels=hidden_layers+1,mod_layer=args.mod_hid_layer)
        self.fact_shape=[]
        if args.highest_flag==1:
            for i in range (self.modulation_base_number):
                self.fact_shape.append((self.h//(2**i),self.w//(2**i)))
        else:
            for i in range (self.modulation_base_number):
                self.fact_shape.append((self.h//(2**(i+1)),self.w//(2**(i+1))))
        self.fact_shape.reverse()
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.dim_arm}"
        )
        self.mask_size=9
        self.encoder_gains_sf=16
        print('Quantizer parameter: encoding gain ',self.encoder_gains_sf)

        self.all_pix_num=(args.patch_h//args.scale)*(args.patch_w//args.scale)
        print('total pixel:',self.all_pix_num)
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(args.context_arm),
            persistent=False,
        )
        self.net.append(SupermaskLinear(args,1,0,sparsity,0,2, 32,bias=False))
        self.net.append(SupermaskLinear(args,1,0,sparsity,0, 3+3+32, 24,bias=False))
        self.net.append(SupermaskLinear(args,1,0,sparsity,0,3+6+24, 16,bias=False))
        self.net.append(SupermaskLinear(args,1,0,sparsity,1,3+6+48+16, 3,bias=False))
        self.net = nn.Sequential(*self.net)
        self.latent_factor=args.latent_factor

        self.modulation_sf= nn.ParameterList()

        self.modules_to_send=['arm','conv_mod','upsampling_2d']

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.viewed_shape=[]
        for layer_idx in range(self.modulation_base_number):
            mod_shape=self.fact_shape[layer_idx]
            shits =  nn.Parameter(torch.zeros(args.batch_size,1,  mod_shape[0], mod_shape[1])).cuda()#.requires_grad=True
            self.modulation_sf.append(shits)
            print('Get Mod with shape',shits.shape,'at layer:',layer_idx+1)
            
    def quantize_all_latent(self,latent):
        q_shifts_all=[]
        #weighted_q_shift_all=[]
        q_shifts_all_for_conv=[]
        for id in range(len(latent)):
            #factor=int(math.sqrt(self.factor[id]))
            q_shifts_id = quantize(
                            latent[id] * self.encoder_gains_sf,
                            self.quantizer_noise_type if self.training else "none",
                            self.quantizer_type if self.training else "hardround",
                            self.soft_round_temperature,
                            self.noise_parameter,)
            q_shifts_all.append(q_shifts_id)
            q_shifts_all_for_conv.append(q_shifts_id)
        #q_upsample_conv=(self.upsampling_2d(q_shifts_all_for_conv)).view(1,len(self.factor),-1,1)
        q_upsample_conv=(self.upsampling_2d(q_shifts_all_for_conv))
        #q_upsample_all_stack=q_upsample_conv


        weight_shift_all=self.conv_mod(q_upsample_conv)
        #weighted_q_shift_all=weight_shift_all.unsqueeze(-1)

        return q_shifts_all_for_conv,weight_shift_all
        
    def estimate_rate(self, decoder_side_latent,arm_model):
        flat_context = torch.cat(
            [
                _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)
                for spatial_latent_i in decoder_side_latent
            ],
            dim=0,
        )
        flat_latent = torch.cat(
            [spatial_latent_i.view(-1) for spatial_latent_i in decoder_side_latent],
            dim=0
        )
        #output = self.net(coords,modulation)
        #if conv:
        flat_context_in=flat_context.unsqueeze(0).transpose(1, 2)
        ##else:
        ##flat_context_in=flat_context
        flat_mu, flat_scale, flat_log_scale__ = arm_model(flat_context_in)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,  # No value can cost more than 16 bits.
        )
        flat_rate = -torch.log2(proba)
        return flat_rate
    def get_network_rate(self):
        """Return the rate (in bits) associated to the parameters (weights and biases) of the different modules
        Returns:
            DescriptorCoolChic: The rate (in bits) associated with the weights
            and biases of each module
        """
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )
        return rate_per_module


    def compute_rate(self):
        all_score_list=[]
        for layer_id, layer in enumerate(self.net):
            all_score_list.append(layer.scores.view(-1))
        all_score=torch.cat(all_score_list,dim=0)
        num_top_20_percent = int(len(all_score) * (1-self.sparsity))
        topk_values, _ = torch.topk(all_score, num_top_20_percent)
        threshold = topk_values.min().item()
        out_num=[]
        for k in range(len(all_score_list)):
            out_num.append(torch.sum(all_score_list[k]>=threshold).item())
        return out_num

    
    def forward(self, coords,in_mask=None):
        saved_mask=[]
        if self.pe_flag==1:
            input_ = self.pe(coords)
        else:
            input_=coords
        q_shifts_all_viewed,weighted_q_shift_all=self.quantize_all_latent(self.modulation_sf)
        kept_num_list=self.compute_rate()
        for layer_id, layer in enumerate(self.net):
            #upsampled_shits=weighted_q_shift_all[layer_id]
            if in_mask==None:
                in_mask_it=None
            else:
                in_mask_it=in_mask[layer_id]

            if layer.last_flag==1:
                #current runing::direct shifts...
                mod=weighted_q_shift_all[layer_id-1].permute(0,2,3,1).view(1,self.h*self.w,-1)
                layer_out,binary_mask =layer(input_,mod,kept_num_list[layer_id],in_mask_it)
                input_ = (torch.tanh(layer_out)+1)/2
                #input_ = ((layer_out))
                #input_= upsampled_scale*input_+ upsampled_shits
                saved_mask.append(binary_mask)
            else:
                if layer_id!=0:
                    mod=weighted_q_shift_all[layer_id-1].permute(0,2,3,1).view(1,self.h*self.w,-1)
                else:
                    mod=None
                layer_out,binary_mask =layer(input_,mod,kept_num_list[layer_id],in_mask_it)
                input_ = F.gelu(layer_out)
                saved_mask.append(binary_mask)

        flat_rate= self.estimate_rate(q_shifts_all_viewed,self.arm)
        return input_,flat_rate,saved_mask   