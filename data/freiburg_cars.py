import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
#import base
import camera
from util import log,debug
import json

class Dataset(base.Dataset):

    def __init__(self,opt, split="train", subset=None):
        self.raw_H,self.raw_W = 90, 160
        super().__init__(opt,split)
        self.root = opt.data.root or "data/freiburg_cars"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}/images_6".format(self.path)
        image_fnames = sorted(os.listdir(self.path_image))
        poses_raw,bounds = self.parse_cameras_and_bounds(opt)
        self.list = list(zip(image_fnames,poses_raw,bounds))
        #print('root', 'path', self.root, self.path, len(self.list))
        
        # only use train split
        train_test_split_path = os.path.join(self.path, 'train_test_splits.json') 
        train_idxs, test_idxs = self.get_train_test_idxs(train_test_split_path)
        self.list = list(np.array(self.list)[train_idxs]) if split=='train' else list(np.array(self.list)[test_idxs])
        
        # manually split train/val subsets
        # validation - the term may be slightly different from the original one
        # num_val_split = int(len(self)*opt.data.val_ratio)
        # self.list = self.list if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]
        # preload dataset
        print(opt.data.preload)
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")
        
#             if split =='train':
#                 self.images = list(np.array(self.images)[train_idxs])
#                 self.cameras = list(np.array(self.cameras, dtype=object)[train_idxs])
#             else:
#                 print(np.array(self.images).shape, test_idxs.shape)
#                 self.images = list(np.array(self.images)[test_idxs])
#                 self.cameras = list(np.array(self.cameras, dtype=object)[test_idxs])
#             #print(len(self.images), len(self.cameras), self.images[0], self.cameras[0])
            
    
    def get_train_test_idxs(self, train_test_split_path):
        with open(train_test_split_path, 'r') as f:
            splits = json.load(f)
            total = splits['train'] + splits['test']
            total = np.sort(total)
            trains, tests = [], []
            for num, t in enumerate(total):
                if t in splits['train']:
                    trains.append(num)
                else:
                    tests.append(num)
            train_idxs = np.arange(len(trains)+len(tests))[trains]
            test_idxs = train_idxs[::30]
            train_idxs = np.array([j for j in range(len(train_idxs)) if j % 30 != 0])
            
            #test_idxs = np.arange(len(trains)+len(tests))[tests]
            return train_idxs, test_idxs
            

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def parse_cameras_and_bounds(self,opt):
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname),dtype=torch.float32)
        # parse cameras (intrinsics and poses)
        cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
        poses_raw = cam_data[...,:4] # [N,3,4]
        poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
        raw_H,raw_W,self.focal = cam_data[0,:,-1]
        #assert(self.raw_H==raw_H and self.raw_W==raw_W)
        self.focal = self.focal * self.raw_H / raw_H
        # parse depth bounds
        bounds = data[:,-2:] # [N,2]
        scale = 1./(bounds.min()*0.75) # not sure how this was determined
        poses_raw[...,3] *= scale
        bounds *= scale
        # roughly center camera poses
        poses_raw = self.center_camera_poses(opt,poses_raw)
        return poses_raw,bounds

    def center_camera_poses(self,opt,poses):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip,pose])
        return pose
