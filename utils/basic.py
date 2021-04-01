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



def imgload(path):#1,3,h,w
    def pil_loader(path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L').resize((388,256))
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


