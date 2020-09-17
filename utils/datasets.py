import os
import yaml
import torch
import numpy as np
import pandas as pd

from utils._utils import get_instance

from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


def align_keypoints(kp, config):
    default_h = config['data']['default_h']
    default_w = config['data']['default_w']
    current_h = default_h
    current_w = default_w

    # Resize
    transform = config['data']['transform']
    if 'Resize' in transform.keys():
        if transform['Resize']:
            new_size = transform['Resize']
            height, width = new_size, new_size

            default = min(default_h, default_w)
            if default == default_h:
                scale = height / float(default_h)
            else:
                scale = width / float(default_w)

            '''
            kp[:,0] = kp[:,0] * scale
            kp[:,1] = kp[:,1] * scale
            '''
            kp = kp * scale

            current_h *= scale
            current_w *= scale

    if 'CenterCrop' in transform.keys():
        if transform['CenterCrop']:
            crop_size = transform['CenterCrop']
            if type(crop_size) == list:
                height, width = crop_size
            else:
                height, width = crop_size, crop_size

            kp[:,0] = kp[:,0] + (width - current_w) / 2
            kp[:,1] = kp[:,1] + (height - current_h) / 2
            
            current_h = height
            current_w = width

    return current_h, current_w, kp

# to [-1., 1.]
def kp_normalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = 2. * kp[..., 0] / (W - 1) - 1
    kp[..., 1] = 2. * kp[..., 1] / (H - 1) - 1
    return kp 


def flist_reader(flist):
    imlist = pd.read_csv(flist, header=None, delim_whitespace=True, index_col=0)

    return imlist


class CelebABase(Dataset):
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        meta = {}
        if self.use_keypoints:
            kp = self.keypoints[index].copy()
            H, W, kp = align_keypoints(kp, self.config)
            kp = torch.tensor(kp)
            meta = {
                'keypts': kp,
                'keypts_normalized': kp_normalize(H, W, kp),
                'index': index
            }
        img = self.default_imgloader(os.path.join(self.subdir, self.filenames[index]))

        if self.transform is not None:
            img = self.transform(img)
        
        return {"data": img, "meta": meta}


    def default_imgloader(self, path):
        return Image.open(path).convert('RGB')
        

class CelebAAligned4MAFLVal(CelebABase):
    def __init__(self, config, train=True, transform=None, use_keypoints=False):
        self.config = config
        self.root = config['data']['root']
        self.train = train
        self.transform = transform
        self.use_hq_ims = config['data']['use_hq_ims']
        self.val_split = config['data']['val_split']
        self.val_size = config['data']['val_size']
        self.use_keypoints = use_keypoints

        if self.use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        self.subdir = os.path.join(self.root, 'Img', subdir)

        anno = pd.read_csv(
            os.path.join(self.root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1, delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(self.root, 'Eval', 'list_eval_partition.txt'), header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(self.root, 'MAFL', 'training.txt'), header=None, delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(self.root, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True, index_col=0)

        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 3).sum() == 19000
        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index]
        elif self.val_split == 'celeba':
            self.data = anno.loc[split[split[1] == 2].index][:self.val_size]
        elif self.val_split == 'mafl':
            self.data = ann.loc[split[split[1] == 4].index]

        # left eye - right eye - nose - left mouth - right mouth
        self.keypoints = np.array(self.data, dtype=np.float).reshape(-1,5,2)
        self.filenames = list(self.data.index)

class CelebAAligned4MAFLVal_MUNIT(CelebABase):
    def __init__(self, config, flist, train=True, transform=None, use_keypoints=False):
        self.config = config
        self.root = config['data']['root']
        self.train = train
        self.transform = transform
        self.use_hq_ims = config['data']['use_hq_ims']
        self.val_split = config['data']['val_split']
        self.val_size = config['data']['val_size']
        self.use_keypoints = use_keypoints
        self.imlist = flist_reader(flist)

        if self.use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        self.subdir = os.path.join(self.root, 'Img', subdir)

        anno = pd.read_csv(
            os.path.join(self.root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1, delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(self.root, 'Eval', 'list_eval_partition.txt'), header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(self.root, 'MAFL', 'training.txt'), header=None, delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(self.root, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True, index_col=0)

        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 3).sum() == 19000
        assert (split[1] == 4).sum() == 1000

        #imlist 에 들어있으면서 3,4 가 아닌거
        self.data = anno.loc[split[(split[1] != 3) & (split[1] != 4)].index]

        self.data = self.data.loc[self.data.index.intersection(self.imlist.index)]

        # left eye - right eye - nose - left mouth - right mouth
        self.keypoints = np.array(self.data, dtype=np.float).reshape(-1,5,2)
        self.filenames = list(self.data.index)

class MAFLAligned(CelebABase):
    def __init__(self, config, train=True, transform=None, use_keypoints=True):
        self.config = config
        self.root = config['data']['root']
        self.train = train
        self.transform = transform
        self.use_hq_ims = config['data']['use_hq_ims']
        self.use_keypoints = use_keypoints
    
        if self.use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        self.subdir = os.path.join(self.root, 'Img', subdir)

        anno = pd.read_csv(
            os.path.join(self.root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1, delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(self.root, 'Eval', 'list_eval_partition.txt'), header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(self.root, 'MAFL', 'training.txt'), header=None, delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(self.root, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True, index_col=0)

        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 3).sum() == 19000
        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 3].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1,5,2)
        self.filenames = list(self.data.index)


if __name__ == '__main__':
    with open('../configs/newcode.yaml', 'r') as stream:
        config = yaml.load(stream)

    flist = '/home/junhahyung/unsuplandmark/data/celeba_munit/male_test.txt'
    celeba = CelebAAligned4MAFLVal_MUNIT(config, flist)
