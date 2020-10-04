import os
import yaml
import torch
import numpy as np
import pandas as pd

import torchvision.transforms.functional as TF

from utils._utils import get_instance
from utils._utils import pad_and_crop

from io import BytesIO
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

class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)


class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high

    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im



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


class ThreeHundredW(Dataset):
    # eye_kp_idxs = [36, 45]
    def __init__(self, config, train=True, do_augmentations=True, use_keypoints=True):
        from scipy.io import loadmat

        self.config = config
        self.root = config['data']['root']
        self.train = train
        self.imwidth = config['data']['imwidth']
        self.use_keypoints = use_keypoints
        self.crop = config['data']['crop']
        self.tunit = config['data']['task']

        afw = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_afw.mat'))
        helentr = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_helen_trainset.mat'))
        helente = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_helen_testset.mat'))
        lfpwtr = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_lfpw_trainset.mat'))
        lfpwte = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_lfpw_testset.mat'))
        ibug = loadmat(os.path.join(self.root, 'Bounding Boxes/bounding_boxes_ibug.mat')) 

        self.bounding_boxes = []
        self.keypoints = []
        self.filenames = []

        if train:
            datasets = [(afw, 'afw'), (helentr, 'helen/trainset'), (lfpwtr, 'lfpw/trainset')]
        else:
            datasets = [(helente, 'helen/testset'), (lfpwte, 'lfpw/testset'), (ibug, 'ibug')]

        for dset in datasets:
            ds = dset[0]
            ds_imroot = dset[1]
            imnames = [ds['bounding_boxes'][0,i]['imgName'][0,0][0] for i in range(ds['bounding_boxes'].shape[1])]
            bbs = [ds['bounding_boxes'][0,i]['bb_ground_truth'][0,0][0] for i in range(ds['bounding_boxes'].shape[1])]

            for i, imn in enumerate(imnames):
                if ds is not ibug or imn.startswith('image'):
                    self.filenames.append(os.path.join(ds_imroot, imn))
                    self.bounding_boxes.append(bbs[i])

                    kpfile = os.path.join(self.root, ds_imroot, imn[:-3] + 'pts')
                    with open(kpfile) as kpf:
                        kp = kpf.read()
                    kp = kp.split()[5:-1]
                    kp = [float(k) for k in kp]
                    assert len(kp) == 68 * 2
                    kp = np.array(kp).astype(np.float32).reshape(-1,2)
                    self.keypoints.append(kp)

        if train:
            assert len(self.filenames) == 3148
        else:
            assert len(self.filenames) == 689

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769], std=[0.2599, 0.2371, 0.2323])
        augmentations = [JPEGNoise(), transforms.transforms.ColorJitter(.4, .4, .4),
                         transforms.ToTensor(), PcaAug()] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose([transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        im = Image.open(os.path.join(self.root, self.filenames[index])).convert("RGB")
        xmin, ymin, xmax, ymax = self.bounding_boxes[index]
        keypts = self.keypoints[index]

        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        bcy = ymin + (bh + 1) / 2
        bcx = xmin + (bw + 1) / 2

        preresize_sz = 256

        bw_ = 200
        fac = bw_ / bw
        imr = im.resize((int(im.width * fac), int(im.height * fac)))

        bcx_ = int(np.floor(fac * bcx))
        bcy_ = int(np.floor(fac * bcy))
        bx = bcx_ - bw_ / 2 + 1
        bX = bcx_ + bw_ / 2
        by = bcy_ - bw_ / 2 + 1
        bY = bcy_ + bw_ / 2
        pp = (preresize_sz - bw_) / 2
        bx = int(bx - pp)
        bX = int(bX + pp)
        by = int(by - pp - 2)
        bY = int(bY + pp - 2)

        imr = pad_and_crop(np.array(imr), [(by - 1), bY, (bx - 1), bX])
        im = Image.fromarray(imr)

        cutl = bx - 1
        keypts = keypts.copy() * fac
        keypts[:,0] = keypts[:,0] - cutl
        cutt = by - 1
        keypts[:,1] = keypts[:,1] - cutt

        kp = None
        if self.use_keypoints:
            kp = keypts - 1
            kp = kp * self.imwidth / preresize_sz
            kp = torch.tensor(kp)
        meta = {}

        data = self.transforms(self.initial_transforms(im.convert("RGB")))
        data = data[:, :self.imwidth, :self.imwidth]
        if self.crop != 0:
            data = data[:, self.crop:-self.crop, self.crop:-self.crop]
        C, H, W = data.shape
        
        if kp is not None:
            kp = kp - self.crop
            kp = torch.tensor(kp)

        if self.use_keypoints:
            meta = {'keypts': kp, 'keypts_normalized': kp_normalize(H, W, kp), 'index': index}
        
        return {"data": data, "meta": meta}


if __name__ == '__main__':
    with open('configs/300w/300w_test.yaml', 'r') as stream:
        config = yaml.load(stream)

    w = ThreeHundredW(config)
    for i in range(10):
        print(w[i]['data'].shape)

