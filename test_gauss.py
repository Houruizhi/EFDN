import cv2
import csv
import os
import argparse
import numpy as np
import time
import torch
from models import make_model
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils import mkdir, load_model, get_files, clear_result_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="test")
# model params
parser.add_argument("--model_name", type=str, default='efdn', help='model name')
parser.add_argument("--num_iter", type=int, default=4, help='iteration number')
parser.add_argument("--groups", type=int, default=1, help='group number')
parser.add_argument("--back_projection", type=int, default=1, help='if sampling')
parser.add_argument("--down_first", type=int, default=1, help='1 for down-and-up; 0 for up-and-down')

parser.add_argument("--in_channels", type=int, default=3, help='input channel')
parser.add_argument("--out_channels", type=int, default=3, help='output channel')
parser.add_argument("--num_features", type=int, default=64, help='features number')
parser.add_argument("--norm_type", type=str, default='bn', help='normalization')
parser.add_argument("--act_type", type=str, default='relu', help='activation function')

parser.add_argument("--save_pic", type=bool, default=True, help='if save the results')
parser.add_argument("--save_path", type=str, default='./test_results/efdn', help='path to save the results')
parser.add_argument("--weights_path", type=str, default=f"./weights/cefdn/efdn_n4_s25/net.pth", help='path of weight')
parser.add_argument("--test_data", type=str, default='/home/rzhou/DataSets/nature_img/TestSet/LIVE1', help='the dir of testing images')
parser.add_argument("--noise_level", type=float, default=25, help='noise level used on test set')
args = parser.parse_args()
def main():
    mkdir(args.save_path)
    with open(os.path.join(args.save_path, 'results.csv'),'a') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'psnr', 'ssim', 'time'])
    
    model = make_model(args)
    
    args.save_path = os.path.join(args.save_path, f'_{args.test_data.split("/")[-1]}')
    clear_result_dir(args.save_path)
    print(f'Loading model from{args.weights_path}')
    model = load_model(model, args.weights_path)
    model = model.cuda()
    model.eval()
    files_source = get_files(args.test_data)
    files_source.sort()
    psnr_test = 0
    ssim_test = 0
    time_test = []
    csv_path = os.path.join(args.save_path, 'results.csv')
    torch.manual_seed('1234')
    np.random.seed(1234)
    for i, file in enumerate(files_source, 0): 
        Img = cv2.imread(file, 1)
        w, h, c = Img.shape
        if w % 2 != 0:
            Img = Img[:w-1,...]
            w -= 1
        if h % 2 != 0:
            Img = Img[:,:h-1,...]
            h -= 1

        if (args.in_channels == 1) and (c==3):
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            Img = Img[None,:,:]

        if args.in_channels == 3:
            Img = Img.transpose(2, 0, 1)

        ISource = np.array(Img).astype('double')/255.
        ISource = torch.Tensor(ISource).unsqueeze(0)
        INoisy = ISource + torch.FloatTensor(ISource.size()).normal_(mean=0, std=args.noise_level/255.)

        ISource, INoisy = ISource.cuda(), INoisy.cuda()
        time1 = time.time()
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        time2 = time.time() - time1
        Out = Out.squeeze().cpu().numpy()
        ISource = ISource.squeeze().cpu().numpy()
        if args.in_channels == 3:
            Out = Out.transpose(1, 2, 0)
            ISource = ISource.transpose(1, 2, 0)
    
        psnr = compare_psnr(Out, ISource)
        ssim = compare_ssim(Out, ISource, multichannel = (c == 3)) 
        psnr_test += psnr
        ssim_test += ssim
        time_test.append(time2)
        print("%s Shape: %s, PSNR %f SSIM %.4f TIME %.4f" % (file, f'{ISource.shape}',psnr, ssim, time2))
        if args.save_pic:
            name = file.split('.')[-2].split('/')[-1]
            cv2.imwrite(f'{args.save_path}/{name}_{psnr}_{ssim}.png', Out * 255)
        with open(csv_path,'a') as f:
        	writer = csv.writer(f)
        	writer.writerow([i, psnr, ssim, time2])
    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print(f"results on {args.test_data}, PSNR: {psnr_test}, SSIM: {ssim_test}\n, TIME:{np.mean(time_test)}")
    with open(csv_path,'a') as f:
    	writer = csv.writer(f)
    	writer.writerow(['avr', psnr_test, ssim_test, np.mean(time_test)])

if __name__ == "__main__":
    main()
