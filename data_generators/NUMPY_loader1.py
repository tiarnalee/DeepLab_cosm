from __future__ import print_function, division
import os
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms 
import sys
sys.path.append('/home/tle19/Documents/deeplab_cosm/preprocessing/')

from my_transforms1 import *
import random
import torch
import cv2
import torchvision.transforms as T

class np_loader(Dataset):
    """
    DeepFashion dataset
    """
#    NUM_CLASSES = 14

    def __init__(self,
                 config,
                 split='train',
                 ):
        super().__init__()
        self._base_dir = config['dataset']['base_path']
        self._image_dir = os.path.join(self._base_dir, 'train', 'image')
        self._cat_dir = os.path.join(self._base_dir, 'labels')
        self.config = config
        self.split = split

        with open(os.path.join(self._base_dir, 'train_val_test.json')) as f:
            self.full_dataset = json.load(f)

        self.images = []
        self.categories = []
        self.num_classes = self.config['network']['num_classes']
        self.crop_size=self.config['image']['crop_size']
        self.mean=self.config['dataset']['mean']
        self.std=self.config['dataset']['std']

        self.shuffle_dataset()

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def shuffle_dataset(self):
        #reset lists
        self.images.clear()
        self.categories.clear()

        dataset = self.full_dataset[self.split]

        if self.split == 'train' and self.config['training']['train_on_subset']['enabled']:
            fraction = self.config['training']['train_on_subset']['dataset_fraction']

            sample = int(len(dataset) * fraction)
            dataset = random.sample(dataset, sample)

        for item in dataset:
            self.images.append(os.path.join(self._image_dir, item['image']))
            self.categories.append(os.path.join(self._cat_dir, item['annotation']))

        #be sure that total dataset size is divisible by 2
        if len(self.images) % 2 != 0:
            self.images.append(os.path.join(self._image_dir, item['image']))
            self.categories.append(os.path.join(self._cat_dir, item['annotation']))

        assert (len(self.images) == len(self.categories))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target}

        #for split in self.split:
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = torch.from_numpy(cv2.imread(self.images[index]))
        _target = torch.from_numpy(np.load(self.categories[index], allow_pickle=True))

        return _img, _target

    def transform_tr(self, sample):
        crop_size=self.config['image']['crop_size']
        mean=self.config['dataset']['mean']
        std=self.config['dataset']['std']
        composed_transforms = transforms.Compose([
            PermuteIm(),
            #Erasing(),
            #AutoContrast(),
            RandomMirroring(),
            RandomRotate(), 
            #RandomGaussianBlur(),
            CenterCrop(crop_size),
            Normalise(mean, std), 
            ])
        
        return composed_transforms(sample)

    def transform_val(self, sample):
        crop_size=self.config['image']['crop_size']
        mean=self.config['dataset']['mean']
        std=self.config['dataset']['std']
        composed_transforms = transforms.Compose([
            PermuteIm(),
            CenterCrop(crop_size),
            Normalise(mean, std),
            ])
        
        return composed_transforms(sample)


    @staticmethod
    def preprocess(sample, crop_size, mean, std):
        sample = {'image': sample['image'], 'label': sample['label']}
        composed_transforms = transforms.Compose([
            PermuteIm(),
            CenterCrop(crop_size),
            Normalise(mean, std),
            ])
                
        return composed_transforms(sample)
    

    def __str__(self):
        return 'UK Biobank(split=' + str(self.split) + ')'
