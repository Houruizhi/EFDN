import argparse

parser = argparse.ArgumentParser(description="denoise")

# model params
parser.add_argument("--model_name", type=str, default='efdn', help='model name')
parser.add_argument("--num_iter", type=int, default=4, help='iteration number')
parser.add_argument("--in_channels", type=int, default=3, help='Input channel')
parser.add_argument("--out_channels", type=int, default=3, help='Output channel')
parser.add_argument("--num_features", type=int, default=64, help='Features number')
parser.add_argument("--norm_type", type=str, default='bn', help='Normalization')
parser.add_argument("--act_type", type=str, default='relu', help='Activation function')
parser.add_argument("--embedding", type=str, default='embedded_gaussian', help="nonlocal method",
    choices=['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation'])
# settings for efdn
parser.add_argument("--down_first", type=bool, default=1, help='down or up first')
parser.add_argument("--back_projection", type=int, default=1, help='if back projection 0 or 1')
parser.add_argument("--groups", type=int, default=1, help='if back projection 0 or 1')

# settings for mefdn
parser.add_argument("--filter_guided", type=int, default=1, help='if filter guided 0 or 1')

# setting for the lvdn
parser.add_argument("--lv_size", type=int, default=9, help='Local variance window size')
parser.add_argument("--operator_scale", type=int, default=4, help='operator scale')

# mobile
parser.add_argument("--expand_scale", type=int, default=2, help='expand the features')

# dataloader settings
parser.add_argument("--train_dir", type=str, default='/home/rzhou/ssd_cache/train_color', help="Training data root")
parser.add_argument("--val_dir", type=str, default='/home/rzhou/ssd_cache/', help="Validating data root")
parser.add_argument("--outf", type=str, default='../experiment/cefdn_n4_s50', help="Training log dir")
parser.add_argument("--seed", type=int, default=1234, help="initial random seed")
parser.add_argument("--repeat", type=int, default=64, help="Repeat the training data")
parser.add_argument("--noise_level", type=float, default=50, help='Noise level')
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--patch_size", type=int, default=60, help="Training data patch size")
parser.add_argument("--num_worker", type=int, default=2, help="Workers number to load data")

# training settings
parser.add_argument("--save_middle_resluts", type=bool, default=False, help="If saving the middle images")
parser.add_argument("--if_parallel", type=bool, default=False, help="If training in parallel")
parser.add_argument("--if_val", type=bool, default=False, help="If validating")
parser.add_argument("--gpu_ids", type=str, default='2', help="Visible gpu ids")
parser.add_argument("--load_pretrain", type=bool, default=False, help="If load the pretrained model")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--lr_decay", type=float, default=0.1, help="Learning rate decay")
parser.add_argument("--gclip", type=float, default=0.5, help='clip the gradient')
parser.add_argument("--optimizer", type=str, default="adam", help='optimizer')
parser.add_argument("--loss_function", type=str, default="mse", help='loss function type')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
