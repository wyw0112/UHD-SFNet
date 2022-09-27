import os
import torch
import network,network_doublegrid,network_doubleNAF,network_wochannel,network_wo1d, network_pin, network_kong
import dataset
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from kornia.filters import laplacian
import math
from PIL import Image
import time
from thop import profile
from thop import clever_format
import copy
def psnr1(img1, img2):
    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)
def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = network.Pyramid()
    #input = torch.randn(1, 3, 224, 224)
    #Mac, params = profile(model, inputs=(input, )) # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
    #Mac, params = clever_format([Mac, params], "%.3f")
    model = model.cuda()
    best_psnr1 = 0.0
    best_epoch1 = -1
    best_psnr2 = 0.0
    best_epoch2 = -1
    #model.load_state_dict(torch.load('model' + '/our_deblur_wochannel_300.pth'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


    mse = nn.L1Loss().cuda()
    content_folder1 = '/home/wenwen/dataset/underwater_dataset/train/input/'
    information_folder = '/home/wenwen/dataset/underwater_dataset/train/gt/'
    test_content1 = '/home/wenwen/dataset/underwater_dataset/UIEBD_test/input/'
    test_information1= '/home/wenwen/dataset/underwater_dataset/UIEBD_test/gt/'
    test_content2 = '/home/wenwen/dataset/underwater_dataset/cycle_test/input/'
    test_information2= '/home/wenwen/dataset/underwater_dataset/cycle_test/gt/'
    train_loader = dataset.style_loader(content_folder1, information_folder, args.size, args.batch_size)
    test_loader1 = dataset.style_loader_test(test_content1, test_information1, args.size, 1)
    test_loader2 = dataset.style_loader_test(test_content2, test_information2, args.size, 1)
    num_batch = len(train_loader)
    t_num_batch1 = len(test_loader1)
    t_num_batch2 = len(test_loader2)
    for epoch in range(0, args.epoch):
        for phase in ['train', 'val']:
        #for phase in ['train']:
            if phase == 'train':
                model.train(True)
                loop = tqdm(enumerate(train_loader), total=num_batch)
                for idx, batch in loop:
                    content = batch[0].float().cuda()
                    information = batch[1].float().cuda()

                    optimizer.zero_grad()

                    output = model(content)

                    total_loss = mse(output, information)

                    total_loss.backward()

                    optimizer.step()
                    loop.set_description(f'Epoch [{epoch+1}/{args.epoch}]')
                    loop.set_postfix(loss = total_loss.item())
                    #if np.mod(total_iter + 1, 1) == 0:
                    #    print('3_Epoch:{}/{} Iter:{} total loss: {}'.format(epoch, args.epoch - 1, total_iter,
                        #                                                  total_loss.item()))

                    #if not os.path.exists(args.save_dir + '/image'):
                    #    os.mkdir(args.save_dir + '/image')
                if (epoch + 1) % 20 == 0:
                    out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
                    #save_image(out_image, args.save_dir + '/image/iter{}.jpg'.format(epoch + 1))
                    torch.save(model.state_dict(), 'model' + '/test_our_deblur_{}.pth'.format(epoch+1))
            else:
                PSNR = 0
                error = 0
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for idx, batch in tqdm(enumerate(test_loader1), total=t_num_batch1):
                        # 获取输入
                        content = batch[0].float().cuda()
                        information = batch[1].float().cuda()
                        output = model(content)
                        try:
                            PSNR += psnr1(information*255, output*255)
                        except Exception as e:
                            error+=1
                            print("IndexError Details : " + str(e))
                            continue

                PSNR = PSNR/(len(test_loader1.dataset)-error)
                if PSNR>best_psnr1:
                    best_psnr1 = PSNR
                    best_epoch1 = epoch
                    #torch.save(model.state_dict(), 'model' + '/our_deblur_UIEBD_{}.pth'.format("best"))
                #print('Best UIEBD val psnr: {:4f},Best epoch:{}'.format(best_psnr1, best_epoch1))
                #print('UIEBD val psnr: {:4f}'.format(PSNR))
                print('UIEBD val psnr: {:4f}'.format(PSNR))
                PSNR = 0
                error = 0
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for idx, batch in tqdm(enumerate(test_loader2), total=t_num_batch2):
                        # 获取输入
                        content = batch[0].float().cuda()
                        information = batch[1].float().cuda()
                        output = model(content)
                        try:
                            PSNR += psnr1(information*255, output*255)
                        except Exception as e:
                            error+=1
                            print("IndexError Details : " + str(e))
                            continue
                PSNR = PSNR/(len(test_loader2.dataset)-error)
                if PSNR>best_psnr2:
                    best_psnr2 = PSNR
                    best_epoch2 = epoch
                    #torch.save(model.state_dict(), 'model' + '/our_deblur_CYCLE_{}.pth'.format("best"))
                #print('Best CYCLE val psnr: {:4f},Best epoch:{}'.format(best_psnr2, best_epoch2))
                print('CYCLE val psnr: {:4f}'.format(PSNR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train(args)