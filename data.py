import os
import os.path
import numpy as np
import torch.utils.data as data

from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def load_landmark(height, width, default_h, default_w, landmark_dir):
    landmarks = {}
    scale = width / default_w
    new_y = default_h * scale
    y_margin = (new_y - height) / 2

    with open(landmark_dir, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i < 2:
                continue
            line = line.strip().split()
            name = line[0]
            landmark = np.array(line[1:]).reshape(5,2).astype(int)
            landmark[:,0] = landmark[:,0] * scale
            landmark[:,1] = landmark[:,1] * scale - y_margin

            # normalize
            landmark = (landmark - height/2) / (height/2)
            landmarks[name] = landmark
    return landmarks


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader


    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img
    

    def __len__(self):
        return len(self.imgs)


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageAndLandmarks(data.Dataset):
    def __init__(self, root, flist, landmark_dir, height, width, default_h, default_w, transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader
        self.landmarks = load_landmark(height, width, default_h, default_w, landmark_dir)

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        lm = self.landmarks[impath]

        return (img, lm)

    def __len__(self):
        return len(self.imlist)
