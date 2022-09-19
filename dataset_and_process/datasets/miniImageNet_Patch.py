from torchvision import transforms
import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as functional
from .miniImageNet import miniImageNet

import random

class RandomResizedCrop_revise(RandomResizedCrop):
    """
    Modified from torchvision, return positions of cropping boxes
    """
    def __init__(self, size):
        super().__init__(size = size)

    def centercrop(self, img):
        w,h = img.size
        w, h = int(w*random.random()/1.5), int(h*random.random()/1.5)

        transform = transforms.Compose([
            transforms.CenterCrop((w,h)),
        ])
        return transform(img), (1,1,1,1)


    def forward(self, img):
        # return self.centercrop(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i,j,h,w)

class miniImageNet_Patch(miniImageNet):
    r"""The standard  dataset for miniImageNet. ::
         
        root
        |
        |
        |---train
        |    |--n01532829
        |    |   |--n0153282900000005.jpg
        |    |   |--n0153282900000006.jpg
        |    |              .
        |    |              .
        |    |--n01558993
        |        .
        |        .
        |---val
        |---test  
    Args:
        root: Root directory path.
        mode: train or val or test
    """
    def __init__(self, root: str, mode: str, image_sz = 84) -> None:
        super().__init__(root, mode)
        self.crop_func = RandomResizedCrop_revise(image_sz)
        self.crop_num =  5
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        patch_list = []
        positions= []
        patch_list.append(self.transform(image))
        positions.append([-1.,-1.,-1.,-1.])
        for num_patch in range(self.crop_num):
            image_, position = self.crop_func(image)
            positions.append(position)                  # i, j, h, w
            patch_list.append(self.transform(image_))
        patch_list=torch.stack(patch_list,dim=0)        # all picture + patch * 30
        return patch_list[0], patch_list[1:], label




def return_class():
    return miniImageNet_Patch

if __name__ == '__main__':
    a = miniImageNet_Patch("../../datasets/miniImageNet_Ori_FG/oriData", "train")
    a = a.__getitem__(1)
    print()
