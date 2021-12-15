import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading,queue
from easydict import EasyDict as edict

import util
from util import log,debug

class Dataset(torch.utils.data.Dataset):

    def __init__(self,opt,split="train"):
        super().__init__()
        self.opt = opt
        self.split = split
        self.augment = split=="train" and opt.data.augment
        # define image sizes
        if opt.data.center_crop is not None:
            self.crop_H = int(self.raw_H*opt.data.center_crop)
            self.crop_W = int(self.raw_W*opt.data.center_crop)
        else: self.crop_H,self.crop_W = self.raw_H,self.raw_W
        if not opt.H or not opt.W:
            opt.H,opt.W = self.crop_H,self.crop_W

    def setup_loader(self,opt,shuffle=False,drop_last=False):
        loader = torch.utils.data.DataLoader(self,
            batch_size=opt.batch_size or 1,
            num_workers=opt.data.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False, # spews warnings in PyTorch 1.9 but should be True in general
        )
        print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self,opt):
        raise NotImplementedError

    def preload_worker(self,data_list,load_func,q,lock,idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(self.opt,idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self,opt,load_func,data_str="images"):
        data_list = [None]*len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(opt.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

    def __getitem__(self,idx):
        raise NotImplementedError

    def get_image(self,opt,idx):
        raise NotImplementedError

    def generate_augmentation(self,opt):
        brightness = opt.data.augment.brightness or 0.
        contrast = opt.data.augment.contrast or 0.
        saturation = opt.data.augment.saturation or 0.
        hue = opt.data.augment.hue or 0.
        color_jitter = torchvision.transforms.ColorJitter.get_params(
            brightness=(1-brightness,1+brightness),
            contrast=(1-contrast,1+contrast),
            saturation=(1-saturation,1+saturation),
            hue=(-hue,hue),
        )
        aug = edict(
            color_jitter=color_jitter,
            flip=np.random.randn()>0 if opt.data.augment.hflip else False,
            rot_angle=(np.random.rand()*2-1)*opt.data.augment.rotate if opt.data.augment.rotate else 0,
        )
        return aug

    def preprocess_image(self,opt,image,aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            image = image.rotate(aug.rot_angle,resample=PIL.Image.BICUBIC)
        # center crop
        if opt.data.center_crop is not None:
            self.crop_H = int(self.raw_H*opt.data.center_crop)
            self.crop_W = int(self.raw_W*opt.data.center_crop)
            image = torchvision_F.center_crop(image,(self.crop_H,self.crop_W))
        else: self.crop_H,self.crop_W = self.raw_H,self.raw_W
        # resize
        if opt.data.image_size[0] is not None:
            image = image.resize((opt.W,opt.H))
        image = torchvision_F.to_tensor(image)
        return image

    def preprocess_camera(self,opt,intr,pose,aug=None):
        intr,pose = intr.clone(),pose.clone()
        # center crop
        intr[0,2] -= (self.raw_W-self.crop_W)/2
        intr[1,2] -= (self.raw_H-self.crop_H)/2
        # resize
        intr[0] *= opt.W/self.crop_W
        intr[1] *= opt.H/self.crop_H
        return intr,pose

    def apply_color_jitter(self,opt,image,color_jitter):
        mode = image.mode
        if mode!="L":
            chan = image.split()
            rgb = PIL.Image.merge("RGB",chan[:3])
            rgb = color_jitter(rgb)
            rgb_chan = rgb.split()
            image = PIL.Image.merge(mode,rgb_chan+chan[3:])
        return image

    def __len__(self):
        return len(self.list)
