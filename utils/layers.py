import torch
import  torch.nn as nn
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
import math
from .basic import imgload
from tqdm import tqdm
import matplotlib.pyplot as plt
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):

        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Photometor():
    def __init__(self,pool,batch_size,step):
        self.pool = pool
        self.batch_size = batch_size
        self.step = step
        self.ssim = SSIM().cuda()
        #self.ssim.cuda()


        pass
    def __call__(self, paths):

        epoch = math.ceil(len(paths) / self.batch_size)
        error_maps = []
        for i in tqdm(range(0, epoch)):

            if i == epoch - 1:  # last batch,
                residue = len(paths) % self.batch_size
                if residue == 0:
                    front_files = paths[i * self.batch_size:-self.step]
                    rear_files = paths[i * self.batch_size + self.step:]
                elif self.step > residue:
                    break
                elif self.step <= residue:
                    residue_comp = residue - self.step

                    front_files = paths[i * self.batch_size:i * self.batch_size + residue_comp]
                    rear_files = paths[i * self.batch_size + self.step:]
            else:
                rear_files = paths[i * self.batch_size + self.step:i * self.batch_size + self.step + self.batch_size]
                front_files = paths[i * self.batch_size:i * self.batch_size + self.batch_size]

            # with Pool(processes=3) as p:
            front_imgs = self.pool.map(imgload, rear_files)
            rear_imgs = self.pool.map(imgload, front_files)

            front_batch = torch.cat(front_imgs, dim=0)
            rear_batch = torch.cat(rear_imgs, dim=0)

            if front_batch.shape != rear_batch.shape:
                bx, _, _, _ = front_batch.shape
                by, _, _, _ = rear_batch.shape
                b = min(bx, by)
                front_batch = front_batch[:b, ]
                rear_batch = rear_batch[:b, ]

            error_ssim = self.ssim(front_batch, rear_batch).mean(1, True)

            abs_diff = torch.abs(front_batch - rear_batch)
            l1_error = abs_diff.mean(1, True)  # [b,1,h,w]

            error = 0.85 * error_ssim + 0.15 * l1_error
            #error_maps.append(error.cpu().numpy())
            error_maps.append(error)

            # if mode =='mean':
            #     ret_list += loss.mean([1, 2, 3]).cpu().numpy().tolist()
            # elif mode == 'raw'
        error_maps = torch.cat(error_maps,dim=0)
        return error_maps

def PhotometricErr(paths,pool,batch_size,step):

    ssim = SSIM().cuda()


    epoch = math.ceil(len(paths)/batch_size)
    error_maps = []
    for i in tqdm(range(0,epoch)):

        if i==epoch-1:#last batch,
            residue = len(paths )% batch_size
            if residue==0:
                front_files = paths[i * batch_size:-step]
                rear_files = paths[i * batch_size + step:]
            elif step > residue:
                break
            elif step <= residue :
                residue_comp = residue-step

                front_files = paths[i*batch_size:i*batch_size + residue_comp]
                rear_files=paths[i*batch_size+step:]
        else:
            rear_files = paths[i*batch_size+step:i*batch_size+step+batch_size]
            front_files =  paths[i*batch_size:i*batch_size+batch_size]

        #with Pool(processes=3) as p:
        front_imgs=pool.map(imgload,rear_files)
        rear_imgs=pool.map(imgload,front_files)

        front_batch = torch.cat(front_imgs,dim=0)
        rear_batch= torch.cat(rear_imgs,dim=0)


        if front_batch.shape != rear_batch.shape:
            bx, _, _, _ = front_batch.shape
            by, _, _, _ = rear_batch.shape
            b = min(bx, by)
            front_batch = front_batch[:b, ]
            rear_batch = rear_batch[:b, ]



        loss_ssim = ssim(front_batch, rear_batch).mean(1, True)

        abs_diff = torch.abs(front_batch - rear_batch)
        l1_loss = abs_diff.mean(1, True)  # [b,1,h,w]

        error = 0.85 * loss_ssim + 0.15 * l1_loss
        error_maps.append(error.cpu().numpy())
        # if mode =='mean':
        #     ret_list += loss.mean([1, 2, 3]).cpu().numpy().tolist()
        # elif mode == 'raw'
    return error_maps






