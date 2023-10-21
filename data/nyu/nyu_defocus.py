import os
import cv2
import json
import scipy
import torch
import numpy as np
import random
import json
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#selected_dirs: what rgb directories are being selected : a list of indices of sorted dir names
class nyudefocus(Dataset):
    def __init__(self, conf,mode):
        self.conf=conf
        self.mode=mode
        self.blur_scale=25e-3
        meta_file=open(os.path.join(conf.datasets.nyu_defocus.base_path,'meta.json'))
        self.meta_data=json.load(meta_file)
        self.depth_path=os.path.join(conf.datasets.nyu_defocus.base_path,
                                     conf.datasets.nyu_defocus.depth_dir)
        #read scene names
        scene_path=os.path.join(conf.datasets.nyu_defocus.base_path, 'scenes.mat')
        self.scenes=scipy.io.loadmat(scene_path)['scenes']

        #read splits
        splits_path=os.path.join(conf.datasets.nyu_defocus.base_path, 'splits.mat')
        splits=scipy.io.loadmat(splits_path)

        if mode=='train':
            self.file_idx=list(splits['trainNdxs'][:,0])
        else:
            self.file_idx=list(splits['testNdxs'][:,0])

        crop=conf.datasets.nyu_defocus.crop=conf.datasets.nyu_defocus.crop
        basic_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(crop[0],crop[1]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.train_transform=A.Compose(transforms=basic_transform)
        self.test_transform=transforms.ToTensor()
        
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (mode, len(self.file_idx)))
        

    def __len__(self):
        return len(self.file_idx)
        
    def get_blur(self,s1,s2,f):
        blur=np.abs(s2-s1)/s2/(s1-f)*(f**2)/(self.blur_scale**2)
        return blur
    
    def augment_training_data(self,image,depth):
        H, W, C = image.shape

        additional_targets = {'depth': 'mask'}
        
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)

        image = augmented['image']
        depth = augmented['depth']

        totensor=transforms.ToTensor()
        image = totensor(image)
        depth = self.to_tensor(depth).squeeze()
        return image,depth

    def augment_test_data(self, image, depth):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()
        return image,depth
    
    def get_camparam(self,dir_name):
        splitvals=dir_name.split('_')
        f_m=float(splitvals[1])*1e-3
        fdist=float(splitvals[3])
        pix_len=self.meta_data['px']
        original_f_px=self.meta_data['original_f_px']
        f_px=f_m/pix_len
        scale=f_px/original_f_px
        return {'f_m':f_m,'f_px':f_px,'scale':scale,'fdist':fdist}

    def __getitem__(self, idx):
        num=self.file_idx[idx]
        #select an item from rgb_dir_list
        rgb_dir=random.choice(self.conf.datasets.nyu_defocus.rgb_dirs)
        camparam=self.get_camparam(rgb_dir)

        rgbpath=os.path.join(self.conf.datasets.nyu_defocus.base_path,
                             rgb_dir,
                             (str(num)+".png"))
        gt_path=os.path.join(self.depth_path,(str(num)+".png"))
        scene_name=self.scenes[num-1][0][0][:-5]

        image = cv2.imread(rgbpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = depth / 1000.0  # convert to meters
        #scale depth image to the scale of image
        h,w,_=image.shape
        depth_re=cv2.resize(depth,(w,h))

        if self.mode=='train':
            transformed=self.train_transform(image=image,mask=depth)
            image,depth = transformed['image'],transformed['mask']
        elif self.mode=='test':
            transformed=self.test_transform(image=image,mask=depth)
            image,depth = transformed['image'],transformed['mask']

        blur=self.get_blur(camparam['fdist'],depth,camparam['f_m'])

        return {'image': image, 'depth': depth, 'blur':blur,'camparam':camparam}

