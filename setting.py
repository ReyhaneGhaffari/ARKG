import argparse
import torch
#......Reyhane Ghaffari et.al
#......2025
def get_setting():
 
    parser = argparse.ArgumentParser(description='Settings for ARKG method')
    parser.add_argument('--dataset', type=str, default='domainnet',choices=[ 'office', 'officehome','visda','domainnet'],help='Dataset name')
    parser.add_argument('--main_folder', type=str,default='C:\\Users\\USER\\befor_start_1404\\Recon_code\\Recon',help='Main folder path for dataset')
    parser.add_argument('--num_class', type=int, default=41, help='Number of classes')
    parser.add_argument('--source_domain', type=str, default='clipart',help='Source domain')
    parser.add_argument('--target_domain', type=str, default='real',help='Target domain')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--allowed_hosts', type=str, default='DESKTOP-65IEAN0',help='Comma separated list of authorized hostnames')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size for resizing')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='default: cuda if  available, otherwise cpu')
    parser.add_argument('--backbone',type=str, default='densenet121', choices=['resnet50',  'vgg19', 'efficientnet_b0', 'densenet121'], help='Backbone ')
    parser.add_argument('--latent_dimention', type=int, default=256,help='Latent space dimension for the model')
    parser.add_argument('--drop_p', type=float, default=0.1 ,help='Dropout probability')
    parser.add_argument('--out_features', type=int, default=128, help='Residual layer dimension')
    parser.add_argument('--dec0', type=int, default=64,  help='Number of channels for dec0 layer')
    parser.add_argument('--dec1', type=int, default=128, help='Number of channels for dec1 layer')
    parser.add_argument('--dec2', type=int, default=256, help='Number of channels for dec2 layer')
    parser.add_argument('--dec3', type=int, default=512, help='Number of channels for dec3 layer')
    parser.add_argument('--dec4', type=int, default=1024,help='Number of channels for dec4 layer')
    parser.add_argument('--size_recimg', type=int, default=128, help='Size of the reconstructed image')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps for gradient reversal')
    parser.add_argument('--fc_units', type=int, default=512, help='Number of units in the fully connected layer')
    parser.add_argument('--T', type=float,  default=1, help='Calibration parameter')
    parser.add_argument('--input_Disc', type=int, default=64, help='Input dimension for discriminator network')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Total number of training epochs')
    parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum learning rate for LR scheduler')
    parser.add_argument('--alpha',  type=float, default=10.0, help='Alpha for LR scheduler')
    parser.add_argument('--beta',   type=float, default=0.75, help='Beta for LR scheduler')
    parser.add_argument('--start_point', type=int, default=150, help='Starting point for weighted training')
    parser.add_argument('--Beta2', type=float, default=0.01,   help='Beta2 value for dynamic adversarial boundary')
    parser.add_argument('--Beta1', type=float, default=0.1, help='Beta1 value for KL and prceptual loss')
    parser.add_argument('--rho', type=float, default=3.0, help='Threshold parameter for high-confidence sample selection')
    parser.add_argument('--recon_loss_type', type=str, default='mse', choices=['mse', 'nll'], help='Type of reconstruction loss: mse or nll (default: mse)')
    parser.add_argument('--lr_Ens', type=float, default=0.00001)
    parser.add_argument('--lr_Ent', type=float, default=0.00001)
    parser.add_argument('--lr_Dec', type=float, default=0.001)
    parser.add_argument('--lr_Cls', type=float, default=0.0001)
    parser.add_argument('--lr_Dis', type=float, default=0.001)
    parser.add_argument('--lr_Fsh', type=float, default=0.00001)
 
    args, unknown = parser.parse_known_args() 

    return args


 




#parser.add_argument('--main_folder', type=str, default='/path/to/data', help='Main folder path for the dataset')