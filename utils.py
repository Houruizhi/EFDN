import os
import math
import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
def normalize(data):
    return data/255.

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_files(root, ext = ['jpg','bmp','png']):
    files = []
    for file_ in os.listdir(root):
        file_path = os.path.join(root, file_)
        if os.path.isdir(file_path):
            files += get_files(file_path)
        else:
            if file_path.split('.')[-1] in ext:
                files.append(file_path)
    return files

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def chp_process(chp):
    new_state_dict = OrderedDict()
    for k, v in chp.items():
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def load_model(model, path):
    if os.path.exists(path):
        checkpoint = chp_process(torch.load(path))
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f'{path} not exists')
    return model

def tensor_to_image(img):
    """
    input: (n,c,w,h)
    output: (n,w,h,c), c=3 or (n,w,h), c=1
    """ 
    n,c,w,h = img.shape
    if c == 1:
        img = img.squeeze()
    elif c == 3:
        img = img.permute(0,2,3,1)
    if img.__class__ == torch.Tensor:
        if img.device != torch.device('cpu'):
            img = img.cpu()
        img = img.numpy().astype(np.float32)
    return img

def batch_PSNR(Img, Iclean, data_range):
    Img = tensor_to_image(Img)
    Iclean = tensor_to_image(Iclean)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += compare_psnr(Iclean[i, ...], Img[i, ...])
    return psnr/Img.shape[0]

def batch_SSIM(Img, Iclean):
    Img = tensor_to_image(Img)
    Iclean = tensor_to_image(Iclean)

    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i,...], Img[i,...], multichannel = True if c==3 else False)
    return (SSIM/Img.shape[0])

def gradient_clip(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-2, 2)
    return optimizer
    
def clear_result_dir(path):
    if os.path.exists(path):
        for i in os.listdir(path):
            os.remove(os.path.join(path,i))
    else:
        mkdir(path)

def make_loader(args):
    from dataset import Dataset
    from torch.utils.data import DataLoader
    dataset_train = Dataset(args, data_root = args.train_dir, train = True)
    loader_train = DataLoader(dataset = dataset_train, num_workers = args.num_worker,\
        batch_size = args.batch_size, shuffle = True)
    if args.if_val:
        dataset_val = Dataset(args, data_root=args.val_dir, train=False)
        loader_val = DataLoader(dataset=dataset_val, num_workers=args.num_worker,\
            batch_size=1, shuffle=False)
    else:
	    loader_val = None
    return {'train':loader_train, 'val':loader_val}

def initial_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

