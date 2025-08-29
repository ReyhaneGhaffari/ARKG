import os
import time
import torch
import copy
import glob
import random
import shutil
import itertools
import traceback
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from Functions import * 
from torch.nn import init
import torch.optim as optim
from itertools import chain
import datetime
from utils import weights_init
import torch.nn.functional as F
from setting import get_setting
import matplotlib.pyplot as plt
import matplotlib.image as imag
from setting import get_setting
from sklearn.manifold import TSNE
from utils import exp_lr_scheduler
from utils import get_learning_rate
from utils import Beta0_scheduler
from utils import Beta2_scheduler
import torchvision.utils  as vutils
import torch.backends.cudnn as cudnn
from pandas.core.common import flatten
from torch.utils.data import DataLoader
from dataload import prepare_dataloaders
from torch.nn.parameter import Parameter
from dataload import prepare_dataloaders
from torch.nn.functional import normalize
from sklearn.metrics import accuracy_score
import torch.distributions as distributions
from Functions import GradientReverseModule
from Functions import NLLNormal,weight_loglike
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader,TensorDataset
from dataload import get_transforms, get_dataset_classes, get_dataset_paths
train_loader, target_train_loader, target_test_loader = prepare_dataloaders(args)
from Network import Source_net, Target_net, Decoder_net, Class_net, Disc_net,Backbone_model


args = get_setting()
torch.cuda.empty_cache()
path = args.main_folder
result_path = os.path.join(path, 'results')
if not os.path.isdir(result_path):   
    os.mkdir(result_path)
         


def Reparam(mus=None, must=None, mut=None, sigmas=None, sigmat=None, sigmast=None):
    epsilon = torch.randn_like(mus)  
    zs  = mus + torch.exp(0.5 * sigmas) * epsilon
    zst = must + torch.exp(0.5 * sigmast) * epsilon
    zt  = mut + torch.exp(0.5 * sigmat) * epsilon
    return zs, zt, zst

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()



        self.Fsh_net = Backbone_model(backbone_name=args.backbone)
        self.Ens_net = Source_net(args)
        self.Ent_net = Target_net(args)
        self.Dec_net = Decoder_net(args)
        self.Cls_net = Class_net(args)
        self.Dis_net = Disc_net(args)

        self.Ens_net.apply(weights_init)
        self.Ent_net.apply(weights_init)
        self.Dec_net.apply(weights_init)
        self.Cls_net.apply(weights_init)
        self.Dis_net.apply(weights_init)

        self.Ens_net.to(args.device)
        self.Ent_net.to(args.device)
        self.Dec_net.to(args.device)
        self.Cls_net.to(args.device)
        self.Dis_net.to(args.device)
        self.Fsh_net.to(args.device)



        self.optimizerEns= optim.Adam(self.Ens_net.parameters(), lr=get_learning_rate('Ens', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)
        self.optimizerEnt= optim.Adam(self.Ent_net.parameters(), lr=get_learning_rate('Ent', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)
        self.optimizerDec= optim.Adam(self.Dec_net.parameters(), lr=get_learning_rate('Dec', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)
        self.optimizerCls= optim.Adam(self.Cls_net.parameters(), lr=get_learning_rate('Cls', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)
        self.optimizerDis= optim.Adam(self.Dis_net.parameters(), lr=get_learning_rate('Dis', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)
        self.optimizerFsh= optim.Adam(self.Fsh_net.parameters(), lr=get_learning_rate('Fsh', args), betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)

        self.recon_criterion = lambda pred, target: F.mse_loss(pred, target, reduction='mean').to(args.device)

        self.criterion_c = nn.CrossEntropyLoss().to(args.device)
        self.criterion_D = nn.BCELoss().to(args.device)
        self.start_point = args.start_point
        self.best_val   = 0       
        self.real_label = 0.
        self.fake_label = 1.
        

    
    def validate(self, epoch=args.num_epochs):
        total = 0
        correct = 0
        kn_correct = 0
        kn_total = 0
        unk_correct = 0
        unk_total = 0
        class_correct = [0 for _ in range(args.num_class)]  
        class_total =   [0 for _ in range(args.num_class)]  
        file_name = result_path
        Model_dir = os.path.join(file_name, 'Model')
        
        if not os.path.isdir(Model_dir):
            os.makedirs(Model_dir)
        
        with torch.no_grad():
            for i, data in enumerate(target_test_loader):
                img_test, gt_label = data
                if img_test is not None and img_test.size(0) == args.batch_size: 
                    img_test, gt_label = Variable(img_test.to(args.device)), Variable(gt_label.to(args.device))  
                    gt_onehot = F.one_hot(gt_label, num_classes=args.num_class).to(args.device)  
                    xt = self.Fsh_net(img_test)
                    mut, sigmat = self.Ent_net(xt, 0, None)
                    stdt = torch.exp(0.5 * sigmat)
                    vart_size = stdt.size()
                    epsilon = torch.randn(vart_size).to(args.device)  
                    zt = mut + epsilon * stdt
                    y_hat, _ = self.Cls_net.forward(zt, 1, is_source=False, reverse=False)
                    label = F.one_hot(torch.argmax(y_hat, dim=1), num_classes=args.num_class) 
                    max_values, max_indices = torch.max(y_hat, dim=1)
                    
                    #=========== UNK
                    indices   = torch.where(gt_label == args.num_class - 1)[0]  
                    indiceskn = torch.where(gt_label != args.num_class - 1)[0]  
                    valueg = gt_label[indices]
                    valuep = max_indices[indices]
                    unk_correct += (valuep == valueg).sum().item()
                    unk_total += valueg.size(0)
                    
                    #============ KN
                    valueg = gt_label[indiceskn]
                    valuekn = max_indices[indiceskn]
                    kn_correct += (valuekn == valueg).sum().item()
                    kn_total += valueg.size(0)
                    
                    correct += (max_indices == gt_label).sum().item()
                    total += gt_onehot.size(0)
        
        OS_star = kn_correct / (kn_total + 1e-9)
        Unk = unk_correct / (unk_total + 1e-9)
        HOS = 2 * OS_star * Unk / (OS_star + Unk + 1e-9)
        Acc = correct / total
        Time = datetime.datetime.now()
        print(f'{Time}|Epoch[{epoch}],====> OS:{Acc*100:.2f}%,====> OS*:{OS_star*100:.2f}%,====> UNK:{Unk*100:.2f}%,====> HOS:{HOS*100:.2f}%')
    
        return Acc

    def train(self, max_epoch=args.num_epochs):
        curr_iter = 0
        reallabel = torch.FloatTensor(args.batch_size).fill_(self.real_label).to(args.device)
        fakelabel = torch.FloatTensor(args.batch_size).fill_(self.fake_label).to(args.device)
        file_name = result_path
        Model_dir = os.path.join(file_name, 'Model')
    
        if not os.path.isdir(Model_dir):
            os.makedirs(Model_dir)
    
        for epoch in range(args.num_epochs):
            self.Fsh_net.train()
            self.Ens_net.train()
            self.Ent_net.train()
            self.Dec_net.train()
            self.Cls_net.train()
            self.Dis_net.train()
    
            source_cycle = itertools.cycle(train_loader)
            target_cycle = itertools.cycle(target_train_loader)
            num_batches_target = len(target_train_loader)
            num_batches_source = len(train_loader)
            max_numbatches = max(num_batches_target, num_batches_source)
            total_itr = args.num_epochs * max_numbatches
            curr_iter += 1
    
            if epoch > -1:
                self.optimizerEns = exp_lr_scheduler(self.optimizerEns, curr_iter, total_itr, alpha=args.alpha, beta=args.beta)
                self.optimizerEnt = exp_lr_scheduler(self.optimizerEnt, curr_iter, total_itr, alpha=args.alpha, beta=args.beta)
                self.optimizerCls = exp_lr_scheduler(self.optimizerCls, curr_iter, total_itr, alpha=args.alpha, beta=args.beta)
                self.optimizerDis = exp_lr_scheduler(self.optimizerDis, curr_iter, total_itr, alpha=args.alpha, beta=args.beta) 
                self.optimizerFsh = exp_lr_scheduler(self.optimizerDis, curr_iter, total_itr, alpha=args.alpha, beta=args.beta)
    
            for i in range(max_numbatches):
                torch.autograd.set_detect_anomaly(True)
                torch.cuda.empty_cache()
    
                (T_image, T_label) = next(target_cycle)
                (S_image, S_label) = next(source_cycle)
    
                T_image = (((T_image * std[0]) + mean[0]) - 0.5) * 2
                S_image = (((S_image * std[0]) + mean[0]) - 0.5) * 2
    
                if S_image.size(0) == args.batch_size and T_image.size(0) == args.batch_size:
                    print(i, "-th iteration of ", max_numbatches - 1)
    
                    try:
                        lds = F.one_hot(S_label, num_classes=args.num_class)
                        ldt = F.one_hot(T_label, num_classes=args.num_class)  
                        S_image, T_image = S_image.to(args.device), T_image.to(args.device)
                        lds, ldt = lds.to(args.device), ldt.to(args.device)
    
                        reallabel = Variable(reallabel)
                        fakelabel = Variable(fakelabel)
    
                        S_image = Variable(S_image)
                        T_image = Variable(T_image)
    
                        lds = Variable(lds)
                        ldt = Variable(ldt)
    
                        # Latent sampling
                        xs = self.Fsh_net(S_image)
                        xt = self.Fsh_net(T_image)
                        mus, sigmas = self.Ens_net(xs)
                        mut, sigmat = self.Ent_net(xt, 0, None)
                        must, sigmast = self.Ent_net(xs, 1, None)
    
                        zs, zt, zst = Reparam(mus, mut, must, sigmas, sigmat, sigmast)
    
                        rimgs = F.interpolate(S_image, size=(args.size_recimg, args.size_recimg), mode='bilinear')
                        rimgt = F.interpolate(T_image, size=(args.size_recimg, args.size_recimg), mode='bilinear')




                     
                     
                        #.............       high-confidence sample selection          ...............#
                        
                        soft_l, logits = self.Cls_net(must, curr_iter, is_source=False, reverse=False)
                        yt_hat, _ =  self.Cls_net(zt, curr_iter, is_source=False, reverse=False)
                        TC = threshold_selection(lds.float(), soft_l, rho  =args.rho)
                        kn_samples, kn_features, kn_labels, unk_samples, unk_features, unk_labels = HCS(zt, rimgt,yt_hat.to(args.device),TC.to(args.device),args)
                        real_labelt = 0.
                        fake_labelt = 1.
                        num_selected = kn_samples.shape[0]
                        
                        reallabelt = torch.FloatTensor(num_selected).fill_(real_labelt).to(args.device)
                        fakelabelt = torch.FloatTensor(num_selected).fill_(fake_labelt).to(args.device)

#...........................................1.source training ......................................................
                     
                        # Updating Disc_net

                        xhats, _,  _ = self.Dec_net.forward(zs,  lds,  is_source=True)
                        xhatst, _, _ = self.Dec_net.forward(zst, lds,  is_source=False)
                        d_labelr, featur_dr = self.Dis_net.forward(rimgs)
                        d_labelf, featur_df = self.Dis_net.forward(xhats)
                        d_labele, featur_de = self.Dis_net.forward(xhatst)
                        
                        realsD = d_labelr.view(-1)
                        fake1D = d_labelf.view(-1)
                        fake2D = d_labele.view(-1)
                        
                        lossDis_real = self.criterion_D(realsD, reallabel)
                        lossDis_fak1 = self.criterion_D(fake1D, fakelabel)
                        lossDis_fak2 = self.criterion_D(fake2D, fakelabel)
                     
                        if kn_samples.size(0) > 1:
                            d_labelt, featur_dt = self.Dis_net.forward(kn_samples)
                            fake3D = d_labelt.view(-1)
                            lossDis_fak3 = self.criterion_D(fake3D, fakelabelt)
                        
                            recon_known, _, _ = self.Dec_net.forward(kn_features, kn_labels, is_source=False)
                            d_labeltrecon, featur_dtrecon = self.Dis_net.forward(recon_known)
                            fake4D = d_labeltrecon.view(-1)
                            lossDis_fak4 = self.criterion_D(fake4D, fakelabelt)
                        else:
                            lossDis_fak3 = torch.tensor(0.0, device=args.device)
                            lossDis_fak4 = torch.tensor(0.0, device=args.device)

                         
                        Beta_0 = Beta0_scheduler(curr_iter,total_itr)
                        Beta_2 = Beta0_scheduler(curr_iter,total_itr)
                        Dis_loss_src = Beta_0*(lossDis_real +lossDis_fak1+lossDis_fak2+lossDis_fak3+lossDis_fak4) 
                        self.Dis_net.zero_grad()
                        Dis_loss_src.backward(retain_graph=True)
                        self.optimizerDis.step()

                        # Updating Dec_net

                        xhats, _, _ = self.Dec_net.forward (zs,    lds,   is_source=True)
                        xhatst,_,_  = self.Dec_net.forward (zst,   lds,   is_source=False)
                        
                        d_labelf, featur_df = self.Dis_net.forward(xhats)
                        d_labele, featur_de = self.Dis_net.forward(xhatst)
                        d_labelr, featur_dr = self.Dis_net.forward(rimgs)
                        epsilon_tensor = torch.tensor(-0.000001, device=args.device)
                     
                        fake1D = d_labelf.view(-1)
                        fake2D = d_labele.view(-1)
                        
                        lossDec_real1 = self.criterion_D(fake1D, reallabel)
                        lossDec_real2 = self.criterion_D(fake2D, reallabel)
                     
                        if kn_samples.size(0) > 1:
                            d_labelt, featur_dt = self.Dis_net.forward(kn_samples)
                            fake3D = d_labelt.view(-1)
                            lossDec_real3 = self.criterion_D(fake3D, reallabelt)
                        
                            recon_known, _, _ = self.Dec_net.forward(kn_features, kn_labels, is_source=False)
                            d_labeltrecon, featur_dtrecon = self.Dis_net.forward(recon_known)
                            fake4D = d_labeltrecon.view(-1)
                            lossDec_real4 = self.criterion_D(fake4D, reallabelt)
                            lossPrecep_t = (self.recon_criterion(featur_dt, featur_dtrecon)) ** 2
                        else:
                         
                            lossDec_real3 = torch.tensor(0.0, device=args.device)
                            lossDec_real4 = torch.tensor(0.0, device=args.device)
                            lossPrecep_t  = torch.tensor(0.0, device=args.device)

                        recon_loss = weight_loglike(xhats, rimgs,1,args)
                        self.Dec_net.zero_grad()
                        Dec_loss_src =recon_loss+ Beta_0*(lossDec_real1+lossDec_real2+lossDec_real3+lossDec_real4)
                        Dec_loss_src.backward(retain_graph=True)
                        self.optimizerDec.step()

                     
                        # Updating Cls_net, #  ~ OSBP Boundary 
                          
                        xs = self.Fsh_net(S_image)
                        xt = self.Fsh_net(T_image)
                        mus, sigmas    = self.Ens_net(xs)
                        mut, sigmat    = self.Ent_net(xt, 0,None)
                        must, sigmast  = self.Ent_net(xs, 1,None)
                        zs, zt, zst = Reparam(mus, mut, must, sigmas, sigmat, sigmast) 
                        
                        PC_t ,logits= self.Cls_net.forward(zt,curr_iter, is_source=False, reverse=False)
                        soft_labelt =PC_t
                        PC_t= PC_t[:, :-1]
                        PC_t= PC_t.detach()
                        ent = -torch.sum(PC_t * torch.log(PC_t + 1e-12), dim=1)
                        PC  = 1 - (ent / torch.log(torch.tensor(args.num_class-1, dtype=PC_t.dtype)))                          
                        exp_delta_= torch.zeros(args.batch_size,args.num_class - 1)
                        score_com_= torch.zeros(args.batch_size,args.num_class - 1)
                        
                        for i in range(args.num_class-1):
                          ldt_hard = F.one_hot(torch.tensor([i]), num_classes=args.num_class)
                          ldt_eval = ldt_hard.to(args.device).repeat(args.batch_size, 1)
                         
                        # ====>      pass target data  through unsupervised path          <====   #
                         
                          Rec_t_eval,_,_ = self.Dec_net.forward(zt, ldt_eval, is_source=False)
                          Dt_label,_ = self.Dis_net.forward(Rec_t_eval)
                          Prob_t= Dt_label.view(-1)#
                          score_com_[:,i]=Prob_t 
                          exp_delta_[:,i]=(Prob_t+1e-10) /(1 - Prob_t+1e-10)
                         
                        Negat_exp_delta=(1/exp_delta_)
                        score_com_ = score_com_.float()
                        min_vals   = torch.min(score_com_, dim=1).values
                        PD         = 1.0 - min_vals
                        Tre  =     0.5*((PD.to('cpu'))+(PC.to('cpu')))
                        Tre  =     torch.unsqueeze(Tre, dim=1).to(args.device)
                        
                       # Updating cls_net
 
                        yt_hat,_=self.Cls_net.forward(zt,curr_iter, is_source=False,reverse=True)
                        prob1 = torch.sum(yt_hat[:, :args.num_class - 1], 1).view(-1, 1)
                        prob2 = yt_hat[:, args.num_class - 1].contiguous().view(-1, 1)
                        prob  = torch.cat((1-prob2, prob2), 1)
                        thresh= torch.cat((Tre.detach(),1-Tre.detach() ), 1)
                        if not torch.all(prob >= 0) or not torch.all(prob <= 1):
                          prob = torch.clamp(prob, 0, 1)
                          print(f"Warning: Corrected prob values at epoch {epoch}")
                        if not torch.all(thresh >= 0) or not torch.all(thresh <= 1):
                          thresh = torch.clamp(thresh, 0, 1)
                          print(f"Warning: Corrected thresh values at epoch {epoch}")
                         
                        lossadvt = self.criterion_D(prob, thresh)
                     
                    
                        soft_l, logits = self.Cls_net.forward(must, curr_iter, is_source=True,  reverse=False)
                        Cls_loss_src   = self.criterion_c(logits, lds.float())+Beta_2*lossadvt
                        self.Cls_net.zero_grad()
                        Cls_loss_src.backward(retain_graph=True)
                        self.optimizerCls.step()
                     
                        # Updating Enc_net

                        soft_l,logits=self.Cls_net.forward(must,curr_iter, is_source=True,  reverse=False)
                        soft_p,logitt=self.Cls_net.forward(zt  ,args.max_steps, is_source=False, reverse=False)
                        kn_samples, kn_features, kn_labels, unk_samples, unk_features, unk_labels = HCS(zt, rimgt,soft_p.to(args.device),TC.to(args.device),args)
                        entropy = -torch.sum(soft_p[:, :-1] * torch.log(soft_p[:, :-1] + 1e-12), dim=1)
                        if unk_samples.size(0) > 1:
                            d_labeltunk, featur_dtunk = self.Dis_net.forward(unk_samples)
                            recon_unknown, _, _ = self.Dec_net.forward(unk_features, unk_labels, is_source=False)
                            d_labeltreconunkn, featur_dtreconunkn = self.Dis_net.forward(recon_unknown)
                            lossPrecep_tunkn =-(self.recon_criterion(featur_dtunk, featur_dtreconunkn)) ** 2
                        else:
                            lossPrecep_tunkn  = torch.tensor(0.0, device=args.device)                     
                        xhats,mu_prior,var_p = self.Dec_net.forward(zs, lds, is_source=True)
                        xhatst,_,_ = self.Dec_net.forward(zst, lds, is_source=False)
                        var_s =(torch.exp(0.5*sigmas)) ** 2
                        var_p =(torch.exp(0.5*sigmast)) ** 2
                        kld   =kl_divergence(mus,var_s,must,var_p)
                        kld   =kld.mean()
                     
                        d_labele, featur_de = self.Dis_net.forward(xhatst)
                        d_labelr, featur_dr = self.Dis_net.forward(rimgs)
                        fakeD = d_labele.view(-1) 

                        lossEns_st    = self.criterion_D(fakeD, reallabel)
                        lossEns_cl    = self.criterion_c(logits, lds.float())
                        EntropT_cl    = torch.mean(entropy)
                        lossPrecep_st = (self.recon_criterion(featur_dr, featur_de)) ** 2
                        lossPrecep_t  = lossPrecep_t.detach()
                        self.Fsh_net.zero_grad() 
                        self.Ens_net.zero_grad()
                        self.Ent_net.zero_grad()                     
                        Ens_loss_src  = lossEns_cl+args.Beta1*(kld +EntropT_cl+lossPrecep_st+lossPrecep_t+lossPrecep_tunkn)+Beta_0*lossEns_st
                        Ens_loss_src.backward(retain_graph=True)
                        self.optimizerEns.step()
                        self.optimizerEnt.step() 
                        self.optimizerFsh.step() 

#..................................................2.target training..........................................
                     
                         #  ~ OSBP Boundary  

                        if epoch>self.start_point:
                         xs = self.Fsh_net(S_image)
                         xt = self.Fsh_net(T_image)
                         mus, sigmas    = self.Ens_net(xs)
                         mut, sigmat    = self.Ent_net(xt, 0,None)
                         must, sigmast  = self.Ent_net(xs, 1,None)
                         zs, zt, zst = Reparam(mus, mut, must, sigmas, sigmat, sigmast) 
                         
                         PC_t ,logits= self.Cls_net.forward(zt,curr_iter, is_source=False, reverse=False)
                         soft_labelt =PC_t
                         PC_t= PC_t[:, :-1]
                         PC_t= PC_t.detach()
                         ent = -torch.sum(PC_t * torch.log(PC_t + 1e-12), dim=1)
                         PC  = 1 - (ent / torch.log(torch.tensor(args.num_class-1, dtype=PC_t.dtype)))                          
                         exp_delta_= torch.zeros(args.batch_size,args.num_class - 1)
                         score_com_= torch.zeros(args.batch_size,args.num_class - 1)
                         
                         for i in range(args.num_class-1):
                           ldt_hard = F.one_hot(torch.tensor([i]), num_classes=args.num_class)
                           ldt_eval = ldt_hard.to(args.device).repeat(args.batch_size, 1)
                          
                         # ====>      pass target data  through unsupervised path          <====   #
                          
                           Rec_t_eval,_,_ = self.Dec_net.forward(zt, ldt_eval, is_source=False)
                           Dt_label,_ = self.Dis_net.forward(Rec_t_eval)
                           Prob_t= Dt_label.view(-1)#
                           score_com_[:,i]=Prob_t 
                           exp_delta_[:,i]=(Prob_t+1e-10) /(1 - Prob_t+1e-10)
                          
                         Negat_exp_delta=(1/exp_delta_)
                         score_com_ = score_com_.float()
                         min_vals   = torch.min(score_com_, dim=1).values
                         PD         = 1.0 - min_vals
                         Tre  =     0.5*((PD.to('cpu'))+(PC.to('cpu')))
                         Tre  =     torch.unsqueeze(Tre, dim=1).to(args.device)
                         w,_  =     torch.max(Negat_exp_delta.to('cpu'),dim=1)
                         Expec1=    w.mean(dim=0)
                         Expec2=(w.detach()**2).mean(dim=0)
                         w_kl=w.detach()**2
                         w=w/(Expec1)
                         w_kl=    (w_kl/Expec2).to(args.device) 
                         w_loglik= w.detach().to(args.device)
                         
                        # Updating Dec_net
                         
                         xhat_t, _, _ = self.Dec_net.forward(zt, soft_labelt, is_source=False)
                         recon_loss_t = weight_loglike(xhat_t, rimgt,w_loglik[:,],args)
                         self.Dec_net.zero_grad()
                         Dec_loss_src =recon_loss_t
                         Dec_loss_src.backward(retain_graph=True)
                         self.optimizerDec.step()



                         yt_hat,_=self.Cls_net.forward(zt,curr_iter, is_source=False,reverse=True)
                         prob1 = torch.sum(yt_hat[:, :args.num_class - 1], 1).view(-1, 1)
                         prob2 = yt_hat[:, args.num_class - 1].contiguous().view(-1, 1)
                         prob  = torch.cat((1-prob2, prob2), 1)
                         thresh= torch.cat((Tre.detach(),1-Tre.detach() ), 1)
                         if not torch.all(prob >= 0) or not torch.all(prob <= 1):
                           prob = torch.clamp(prob, 0, 1)
                           print(f"Warning: Corrected prob values at epoch {epoch}")
                         if not torch.all(thresh >= 0) or not torch.all(thresh <= 1):
                           thresh = torch.clamp(thresh, 0, 1)
                           print(f"Warning: Corrected thresh values at epoch {epoch}")
                          
                         lossadvt = self.criterion_D(prob, thresh)
                         Cls_loss_trg= lossadvt

                      
                          # Updating En_net

                         var_t=(torch.exp(0.5*sigmat)) ** 2
                         var_p=(torch.exp(0.5*sigmast)) ** 2
                         kldt = kl_divergence(mut,var_t,must,var_p)*w_kl[:,]
                         kldt = kldt.mean()
                         yt_hat,_=self.Cls_net.forward(zt,curr_iter, is_source=False,reverse=False)
                         prob1 = torch.sum(yt_hat[:, :args.num_class - 1], 1).view(-1, 1)
                         prob2 = yt_hat[:, args.num_class - 1].contiguous().view(-1, 1)
                         prob  = torch.cat((1-prob2, prob2), 1)
                         thresh= torch.cat((Tre.detach(),1-Tre.detach() ), 1)
                         if not torch.all(prob >= 0) or not torch.all(prob <= 1):
                           prob = torch.clamp(prob, 0, 1)
                           print(f"Warning: Corrected prob values at epoch {epoch}")
                         if not torch.all(thresh >= 0) or not torch.all(thresh <= 1):
                           thresh = torch.clamp(thresh, 0, 1)
                           print(f"Warning: Corrected thresh values at epoch {epoch}")
                          
                         lossadvt = self.criterion_D(prob, thresh)
                         xhat_t, _, _ = self.Dec_net.forward(zt, yt_hat, is_source=False)
                         recon_loss_t = weight_loglike(xhat_t, rimgt,w_loglik[:,],args)
                         Ens_loss_trg= lossadvt
                         self.Ens_net.zero_grad()
                         self.Ent_net.zero_grad() 
                         self.Fsh_net.zero_grad() 
                         Ens_loss_trg  =args.Beta2*Ens_loss_trg+recon_loss_t+args.Beta1*kldt
                         Ens_loss_trg.backward(retain_graph=True)
                         self.optimizerFsh.step() 
                         self.optimizerEns.step()
                         self.optimizerEnt.step() 

                 
                    except Exception as e:
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                        continue
                     
            if epoch>0:
             self.validate(epoch + 1)   
             
        return Cls_loss_src                     


