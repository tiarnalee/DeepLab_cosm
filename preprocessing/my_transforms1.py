import torch
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter
import elasticdeform


##########################################################################
#COSMETRICS 

class PermuteIm(object):
    def __call__(self, sample):
            img = sample['image']
            mask = sample['label']
                
            img=img.permute(2,0,1).float()
            mask=mask.permute(2,0,1).float()/255
            return {'image': img, 'label': mask}
    
class RandomMirroring(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        p1=[random.random(), random.random()]
        if p1[0] < 0.5:
            img = T.functional.hflip(img).float()
            mask = T.functional.hflip(mask)
        if p1[1] < 0.5:
            img = T.functional.vflip(img).float()
            mask = T.functional.vflip(mask)

        return {'image': img, 'label': mask}

class RandomRotate(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.3:
            rotate_degree = random.uniform(-180,180)
            img = T.functional.rotate(img, rotate_degree).float()
            mask = T.functional.rotate(mask, rotate_degree)

        return {'image': img,'label': mask}
        
class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if np.random.randn() < 0.3:
            img= T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5))(img).float()
        return {'image': img,'label': mask}
        

class AutoContrast(object):
  def __call__(self, sample):
    img = sample['image']
    mask = sample['label']
    if np.random.randn() < 0.3:
      img = T.RandomAutocontrast(p=1)(img)

    return {'image': img,'label': mask}

class Erasing(object):
  def __call__(self, sample):
    img = sample['image']
    mask = sample['label']
    if np.random.randn() < 0.2: #0.15
      img = T.RandomErasing(p=1, scale=(0.2, 0.2))(img)

    return {'image': img,'label': mask}

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        transform = T.RandomChoice([
            T.RandomResizedCrop(size=(self.crop_size, self.crop_size))
            # T.Resize(size=(self.crop_size, self.crop_size))
        ])

        img = sample['image']
        mask = sample['label']
        
        img=transform(img).float()
        mask=transform(mask)

        return {'image': img,  'label': mask}


class Normalise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std=std
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img=T.Normalize(mean=self.mean, std=self.std)(img) #98, 80

        return {'image': img, 'label': mask}

