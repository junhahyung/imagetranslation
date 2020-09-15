import yaml
import data.datasets as data_module

from utils import get_instance
from torch.utils.data import DataLoader
from torchvision import transforms


def get_default_loader(config, train, use_keypoints, num_workers=4):
    dataset = config['data']['dataset']
    batch_size = config['data']['batch_size']
    tconf = config['data']['transform']
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), 
                                            (0.5, 0.5, 0.5))]
    if 'CenterCrop' in tconf.keys():
        if tconf['CenterCrop'] is not None:
            crop_size = tconf['CenterCrop']
            if type(crop_size) == list:
                height, width = crop_size
            else:
                height, width = crop_size, crop_size
            transform_list = [transforms.CenterCrop((height, width))] + transform_list

    if 'Resize' in tconf.keys():
        if tconf['Resize'] is not None:
            new_size = tconf['Resize']
            transform_list = [transforms.Resize(new_size)] + transform_list

    transform = transforms.Compose(transform_list)

    dataset = get_instance(data_module, config, 'data', 'dataset', train=train, transform=transform, use_keypoints=use_keypoints)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    with open('../configs/newcode.yaml', 'r') as stream:
        config = yaml.load(stream)

    get_default_loader(config,train=True, use_keypoints=True)    
