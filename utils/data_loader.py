import yaml

import utils.datasets as data_module
from utils._utils import get_instance
#from utils import datasets as data_module
from torch.utils.data import DataLoader
from torchvision import transforms


def get_all_data_loaders(config):
    task = config['data']['task']
    if task == 'MUNIT':
        train_loader_a = get_munit_data_loader(config, True, 'A')
        train_loader_b = get_munit_data_loader(config, True, 'B')
        test_loader_a = get_munit_data_loader(config, False, 'A')
        test_loader_b = get_munit_data_loader(config, False, 'B')

        return train_loader_a, train_loader_b, test_loader_a, test_loader_b

    else:
        train_loader = get_default_data_loader(config, train=True)
        test_loader = get_default_data_loader(config, train=False)

        return train_loader, test_loader

def get_transform(config):
    tconf = config['data']['transform']
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), 
                                            (0.5, 0.5, 0.5))]
    #TODO
    #add random crop

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
    return transform


def get_default_data_loader(config, train, num_workers=4):
    dataset = config['data']['dataset']
    batch_size = config['data']['batch_size']
    use_keypoints = config['data']['use_keypoints']
    transform = get_transform(config)
    dataset = get_instance(data_module, config, 'data', 'dataset', train=train, transform=transform, use_keypoints=use_keypoints)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)

    return loader

def get_munit_data_loader(config, train, group, num_workers=4):
    dataset = config['data']['dataset']
    batch_size = config['data']['batch_size']

    if train:
        if group == 'A':
            file_list = config['data']['data_list_train_a']
        elif group == 'B':
            file_list = config['data']['data_list_train_b']
        else:
            raise NotImplementedError
    else:
        if group == 'A':
            file_list = config['data']['data_list_test_a']
        elif group == 'B':
            file_list = config['data']['data_list_test_b']
        else:
            raise NotImplementedError

    transform = get_transform(config)
    dataset = get_instance(data_module, config, 'data', 'dataset', flist=file_list, train=train, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    with open('/home/junhahyung/unsuplandmark/configs/newcode_munit.yaml', 'r') as stream:
        config = yaml.load(stream)

    get_all_data_loaders(config)
