import cv2
import csv
import os
import sys
import argparse
import numpy as np
import time
import torch
from models import make_model
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils import mkdir, load_model, get_files, clear_result_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser(description="test")
# model params
# model params
parser.add_argument("--model_name", type=str, default='efdn', help='model name')
parser.add_argument("--num_iter", type=int, default=4, help='iteration number')
parser.add_argument("--groups", type=int, default=1, help='iteration number')
parser.add_argument("--back_projection", type=int, default=1, help='if back projection 0 or 1')
parser.add_argument("--down_first", type=int, default=1, help='if back projection 0 or 1')

parser.add_argument("--in_channels", type=int, default=1, help='Input channel')
parser.add_argument("--out_channels", type=int, default=1, help='Output channel')
parser.add_argument("--num_features", type=int, default=64, help='Features number')
parser.add_argument("--norm_type", type=str, default='bn', help='Normalization')
parser.add_argument("--act_type", type=str, default='relu', help='Activation function')

parser.add_argument("--save_pic", type=bool, default=True, help='If save the results')
parser.add_argument("--save_path", type=str, default='./test_results/efdn', help='path to save the results')
parser.add_argument("--weights_path", type=str, default=f"./weights/efdn_jpeg/efdn_n4", help='path of log files')
parser.add_argument("--test_data", type=str, default='/home/rzhou/DataSets/nature_img/TestSet/classic5', help='test on Set12 , Urban100 or Set68')
parser.add_argument("--noise_level", type=float, default=25, help='noise level used on test set')
args = parser.parse_args()

def main():
    mkdir(args.save_path)
    with open(os.path.join(args.save_path, 'results.csv'),'a') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'psnr', 'ssim', 'time'])
    
    model = make_model(args)
    args.weights_path = os.path.join(args.weights_path, 'net.pth')

    print(f'Loading model from{args.weights_path}')
    model = load_model(model, args.weights_path)
    # model.load_state_dict(torch.load(args.weights_path))
    model = model.cuda()
    model.eval()
    files_source = get_files(args.test_data)
    files_source.sort()
    
    for Q in range(10,50,10):
        jpeg_path = args.test_data+'_jpeg'+'/compress_factor'+str(int(Q)) 
        print(jpeg_path)
        psnr_test = 0
        ssim_test = 0
        time_test = []
        save_path = os.path.join(args.save_path, 'compress_factor'+str(int(Q)))
        clear_result_dir(save_path)
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, 'results.csv')
        for i, file in enumerate(files_source, 0): 
            noise = cv2.imread(os.path.join(jpeg_path,file.split('/')[-1].split('.')[0]+'.jpg'),1)
            Img = cv2.imread(file,1)

            if len(Img.shape) == 3:
                w, h, c = Img.shape
            else:
                w, h = Img.shape
                c = 1
            if w % 2 != 0:
                noise = noise[:w-1,...]
                Img = Img[:w-1,...]
                w -= 1
            if h % 2 != 0:
                noise = noise[:,:h-1,...]
                Img = Img[:,:h-1,...]
                h -= 1
            if (args.in_channels == 1) and (c==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
                Img = Img[:,:,None]
                noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
                noise = cv2.cvtColor(noise, cv2.COLOR_RGB2GRAY)
                noise = noise[:,:,None]

            Img = Img.transpose(2, 0, 1)
            noise = noise.transpose(2, 0, 1)

            Img = np.array(Img).astype('double')/255.
            INoisy = np.array(noise).astype('double')/255.

            INoisy = np.array(INoisy).astype('double')
            INoisy = torch.Tensor(INoisy).unsqueeze(0)
            INoisy = INoisy.cuda()

            time1 = time.time()
            with torch.no_grad(): # this can save much memory
                Out = torch.clamp(model(INoisy), 0., 1.)
            time2 = time.time() - time1
            Out = Out.squeeze().cpu().numpy()
            Img = Img.squeeze()
            if args.in_channels == 3:
                Out = Out.transpose(1,2,0)

            psnr = compare_psnr(Out, Img)
            ssim = compare_ssim(Out, Img, multichannel = (c == 3)) 
            psnr_test += psnr
            ssim_test += ssim
            time_test.append(time2)
            print("%s PSNR %f SSIM %.4f TIME %.4f" % (file, psnr, ssim, time2))
            if args.save_pic:
                name = file.split('.')[-2].split('/')[-1]
                cv2.imwrite(f'{save_path}/{name}_{psnr}_{ssim}.png', Out * 255)
            with open(csv_path,'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, Q, psnr, ssim, time2])
        psnr_test /= len(files_source)
        ssim_test /= len(files_source)
        print(f"results on {args.test_data}, PSNR: {psnr_test}, SSIM: {ssim_test}\n, TIME:{np.mean(time_test)}")
        with open(csv_path,'a') as f:
            writer = csv.writer(f)
            writer.writerow(['avr', psnr_test, ssim_test, np.mean(time_test)])

if __name__ == "__main__":
    main()
