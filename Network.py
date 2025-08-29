import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from setting import get_setting
from Functions import GradientReverseModule

import torch
import torch.nn as nn
import torchvision.models as models

args = get_setting()

class Backbone_model(nn.Module):
    def __init__(self, backbone_name=args.backbone, normalize=True):
        super(Backbone_model, self).__init__()
        self.normalize = normalize
        self.mean = False
        self.std = False
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == 'resnet50':
            model_resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(model_resnet.children())[:-2])

        elif self.backbone_name == 'resnet152':
            model_resnet = models.resnet152(pretrained=True)
            self.backbone = nn.Sequential(*list(model_resnet.children())[:-2])

        elif self.backbone_name == 'vgg19':
            model_vgg19 = models.vgg19(pretrained=True)
            self.backbone = model_vgg19.features

        elif self.backbone_name == 'efficientnet_b0':
            model_effnet = models.efficientnet_b0(pretrained=True)
            self.backbone = model_effnet.features

        elif self.backbone_name == 'densenet121':
            model_densenet = models.densenet121(pretrained=True)
            self.backbone = model_densenet.features

        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported.")

        if self.backbone_name == 'densenet121':
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_mean(self):
        if self.mean is False:
            self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.backbone(x)
        return x


def get_channel_count(model):
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        model = model.cuda()
        out = model(dummy_input)
        return out.shape[1]


def get_backbone_output_channel_count(backbone_name=args.backbone):
    model = Backbone_model(backbone_name)
    channel_count = get_channel_count(model)
    print(f"Number of output channels in the last conv layer for {backbone_name}: {channel_count}")
    return channel_count




#....................                      Source Encoder                          ................................#


class Source_net(nn.Module):
    def __init__(self, args):
        super(Source_net, self).__init__()

        self.dz, self.drop_p, self.num_class = args.latent_dimention, args.drop_p, args.num_class
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
        # Initialize the backbone based on args
        self.F_net = Backbone_model(backbone_name=args.backbone)
        self.inchannel = get_channel_count(self.F_net)
        
        self.dropout = nn.Dropout(p=self.drop_p)
        
        # Mu_c (Mean)
        self.mus = nn.Linear(self.inchannel, self.dz) 
        self.bn1 = nn.BatchNorm1d(self.dz, momentum=0.1)
        
        # sigma_s (Standard deviation)
        self.sigmas = nn.Linear(self.inchannel, self.dz)
        self.bn2 = nn.BatchNorm1d(self.dz, momentum=0.1)

    def mean_s(self, x, reverse=False):
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        if reverse:
            x = x.detach()
        x = self.bn1(self.mus(x)) 
        muc = self.relu(x)
        muc = self.dropout(muc)
        return muc

    def sigma_s(self, x, reverse=False):
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        if reverse:
            x = x.detach()
        x = self.bn2(self.sigmas(x))  
        sigmas = self.relu(x) + 1e-6  
        sigmas = self.dropout(sigmas)
        return sigmas

    def forward(self, xs, reverse=False):
        mus = self.mean_s(xs)
        sigma_s = self.sigma_s(xs)
        return mus, sigma_s

#....................                      Target Encoder                          ................................#


class Target_net(nn.Module):
    def __init__(self, args):
       super(Target_net, self).__init__()
       self.dz, self.drop_p, self.num_class = args.latent_dimention, args.drop_p, args.num_class
       self.GAP = nn.AdaptiveAvgPool2d((1, 1))
       self.relu = nn.LeakyReLU(negative_slope=0.2)
       self.F_net = Backbone_model()
       self.SharedMean = Source_net(args)
       self.inchannel = get_channel_count(self.F_net)
       self.dropout = nn.Dropout(p=self.drop_p) 
       self.landa = nn.Parameter(torch.tensor(1.0)) 

       self.out_features = args.out_features  

       self.sigmat = nn.Linear(self.inchannel, self.dz)
       self.bn3 = nn.BatchNorm1d(self.dz, momentum=0.1)  
       
       self.res = nn.Sequential(
           nn.Linear(in_features=self.dz, out_features=self.out_features),
           nn.BatchNorm1d(self.out_features, momentum=0.1),
           nn.LeakyReLU(negative_slope=0.01),
       
           nn.Linear(in_features=self.out_features, out_features=self.out_features),
           nn.BatchNorm1d(self.out_features, momentum=0.1),
           nn.LeakyReLU(negative_slope=0.01),
       
           nn.Linear(in_features=self.out_features, out_features=self.dz),
           nn.BatchNorm1d(self.dz, momentum=0.1),
           nn.Dropout(p=0.1),
       )

    def mean_t(self, x, ld, reverse=False):
        muc, _ = self.SharedMean(x)
        
        if ld is None:
            delta = self.res(muc)  
            delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
            mut = muc + delta  
        else:
            mut = muc
        return mut

    def sigma_t(self, x, reverse=False):
        x = self.GAP(x)  
        x = x.view(x.size(0), -1)  
        if reverse:
            x = x.detach()  
        
        x = self.bn3(self.sigmat(x))  
        sigmat = self.relu(x) + 1e-6  
        sigma_t = self.dropout(sigmat)  
        return sigma_t

    def forward(self, xt, sign, ld, reverse=False):
        if sign == 1:
            mut = self.mean_t(xt, ld)  
            sigma_t = self.sigma_t(xt)  
        else:
            mut = self.landa * self.mean_t(xt, ld)  
            sigma_t = self.landa * self.sigma_t(xt)  

        return mut, sigma_t



#....................                     Shared Decoder                         ................................#


class Decoder_net(nn.Module):
    def __init__(self, args):
        super(Decoder_net, self).__init__()

        self.dec0, self.dec1, self.dec2, self.dec3, self.dec4 = args.dec0, args.dec1, args.dec2, args.dec3, args.dec4
        self.dz, self.drop_p, self.num_class = args.latent_dimention, args.drop_p, args.num_class
        self.relu = nn.LeakyReLU()
        self.softlabel = nn.Sequential(
            nn.Linear(self.num_class, self.dz),
            nn.BatchNorm1d(self.dz, momentum=0.01),
            nn.Sigmoid()
        )
        self.in_channels = self.dz + self.dz
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic')

        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.dec4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.dec4, momentum=0.01),
            self.relu
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(self.dec4, self.dec3, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.dec3, momentum=0.01),
            self.relu
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(self.dec3, self.dec2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec2, momentum=0.01),
            self.relu
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(self.dec2, self.dec1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec1, momentum=0.01),
            self.relu
        )
        self.convTrans4_3 = nn.Sequential(
            nn.ConvTranspose2d(self.dec1, self.dec0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec0, momentum=0.01),
            self.relu
        )
        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(self.dec0, 3, 3, 2, 1, bias=False),
        )

    def forward(self, z, ld, is_source):
        ld = ld.float()
        ld = self.softlabel(ld)
        mu_prior = ld
        if is_source==False:    
         var_p = torch.randn_like(mu_prior) * 0.0005
         l_prior = mu_prior + var_p
         
        else:
         var_p=0
         l_prior = mu_prior
         
        z_shared = torch.cat((z, l_prior), dim=1)
        z_shared = z_shared.unsqueeze(2).unsqueeze(3)
        x = z_shared.view(-1, self.in_channels, 1, 1)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        x = self.convTrans4_3(x)
        x = self.convTrans5(x)
        rec_image = F.interpolate(x, size=(args.size_recimg, args.size_recimg), mode='bicubic')
        final_output = rec_image
        return final_output, mu_prior, var_p
     
     
#....................                     Classifier Network                         ................................#




class Class_net(nn.Module):
    def __init__(self, args):
        super(Class_net, self).__init__()
        self.dz = args.latent_dimention
        self.num_class = args.num_class
        self.max_step = args.max_steps

        self.logit = nn.Sequential(
            nn.Linear(self.dz, args.fc_units),
            nn.ReLU(),
            nn.Linear(args.fc_units, self.num_class)
        )
        self.predict = nn.Softmax(dim=1)
        self.T = args.T
        self.grl_module = GradientReverseModule(args)

    def forward(self, z, step, is_source=False, reverse=False):
        if reverse:
            z = self.grl_module(z, step)
        logits = self.logit(z)
        if is_source:
            probabilities = self.predict(logits)
        else:
            probabilities = self.predict(logits / self.T)
         
        return probabilities, logits



#....................                     Discriminator Network                         ................................#


class Disc_net(nn.Module):
    def __init__(self, args):
        super(Disc_net, self).__init__()

        input_Disc = args.input_Disc
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.feature = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, input_Disc, 3, 1, 1)),
            nn.BatchNorm2d(input_Disc),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.utils.spectral_norm(nn.Conv2d(input_Disc, input_Disc * 2, 3, 1, 1)),
            nn.BatchNorm2d(input_Disc * 2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.utils.spectral_norm(nn.Conv2d(input_Disc * 2, input_Disc * 4, 3, 1, 1)),
            nn.BatchNorm2d(input_Disc * 4),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.utils.spectral_norm(nn.Conv2d(input_Disc * 4, input_Disc * 2, 3, 1, 1)),
            nn.BatchNorm2d(input_Disc * 2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, 4)
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fcD1 = nn.Linear(input_Disc * 2, input_Disc * 4)
        self.bn1 = nn.BatchNorm1d(input_Disc * 4)
        self.fcD3 = nn.Linear(input_Disc * 4, 1)
        self.sigmoid = nn.Sigmoid()

        self.fcD_feat = nn.Linear(input_Disc * 4, input_Disc * 2)

    def forward(self, input):
        x = self.feature(input)
        x = self.global_avgpool(x).view(x.size(0), -1)
        x = self.leaky_relu(self.bn1(self.fcD1(x)))
        dom_logit = self.fcD3(x)
        domain = self.sigmoid(dom_logit)
        f_D = self.fcD_feat(x)
        return domain, f_D







