#visdrone 含有大量不同分辨率, 相机静止序列, 夜晚序列, 这些会对深度估计造成较大的负面影响.
#我们对图像进行预处理, 使得尽可能

import torch
import  torch.nn as nn
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
from tqdm import  tqdm
import math


def writelines(list,path):
    lenth = len(list)
    with open(path,'w') as f:
        for i in range(lenth):
            if i == lenth-1:
                f.writelines(str(list[i]))
            else:
                f.writelines(str(list[i])+'\n')

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines



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



def PhotometricErr(paths,pool,batch_size,step):

    ret_list =[]
    ssim = SSIM().cuda()


    epoch = math.ceil(len(paths)/batch_size)
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

        loss = 0.85 * loss_ssim + 0.15 * l1_loss
        ret_list += loss.mean([1, 2, 3]).cpu().numpy().tolist()





    return ret_list

def imgload(path):#1,3,h,w
    def pil_loader(path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L').resize((88,48))
    img = pil_loader(path)
    return transforms.ToTensor()(img).unsqueeze(0).cuda()




def list_remove(arr,idxs,frame_interval):
    '''
    根据idxs的 true false 进行筛除
    :param arr:
    :param idxs:
    :return:
    '''

    step = max(frame_interval)
    idxs[:step] = 0
    idxs[-step:] = 0


    for i in range(len(idxs)):
        if idxs[i] == 0:
            arr[i] = ''

    while '' in arr:
        arr.remove('')

    return arr


