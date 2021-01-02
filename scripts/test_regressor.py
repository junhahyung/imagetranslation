import os
import sys
import torch
import argparse
import tensorboardX
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torch import nn
from utils.data_loader import get_all_data_loaders
from utils._utils import get_config, prepare_sub_folder

from argparse import Namespace
from datetime import datetime
from utils.utils_tunit import *
from utils._utils import get_instance, NoGradWrapper, get_model_list
import models.model as module_arch
import models.metric as module_metric

### MUNIT ###
from models.munit_networks import AdaINGen

### TUNIT ###
from models.tunit.generator import Generator as Generator
from models.tunit.discriminator import Discriminator as Discriminator
from models.tunit.guidingNet import GuidingNet
from models.tunit.inception import InceptionV3
import models.loss as module_loss

import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mafl/mafl_regressor_tunit.yaml', help='Path to the config file.')
parser.add_argument('--model_name', default=None, type=str, metavar='PATH')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--ckpt', type=str, help='which checkpoint to use', required=True)

# tunit model loader
def load_model_tunit(args, networks, opts):
    check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
    to_restore = check_load.readlines()[-1].strip()
    load_file = os.path.join(args.log_dir, to_restore)
    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        if not args.multiprocessing_distributed:
            for name, net in networks.items():
                if name in ['inceptionNet']:
                    continue
                tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                if 'module' in tmp_keys:
                    tmp_new_dict = OrderedDict()
                    for key, val in checkpoint[name + '_state_dict'].items():
                        tmp_new_dict[key[7:]] = val
                    net.load_state_dict(tmp_new_dict)
                    networks[name] = net
                else:
                    net.load_state_dict(checkpoint[name + '_state_dict'])
                    networks[name] = net

        for name, opt in opts.items():
            opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
            opts[name] = opt
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.log_dir))

# model builder for tunit
def build_model_tunit(args):
    args.to_train = 'CDGI'
    #args.to_train = 'G'

    networks = {}
    opts = {}
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'D' in args.to_train:
        networks['D'] = Discriminator(args.img_size, num_domains=args.output_k)
    if 'G' in args.to_train:
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)
    if 'I' in args.to_train:
        networks['inceptionNet'] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]])

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for name, net in networks.items():
            networks[name] = net.cuda(args.gpu)
    else:
        for name, net in networks.items():
            networks[name] = torch.nn.DataParallel(net).cuda()

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C'].module.parameters() if args.distributed else networks['C'].parameters(),
            1e-4, weight_decay=0.001)
        networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(
            networks['D'].module.parameters() if args.distributed else networks['D'].parameters(),
            1e-4, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].module.parameters() if args.distributed else networks['G'].parameters(),
            1e-4, weight_decay=0.0001)

    return networks, opts

def main():
    args = parser.parse_args()

    config = get_config(args.config)


    config_n = Namespace(**config)
    config_n.gpu = args.gpu
    config_n.distributed = False
    config_n.multiprocessing_distributed = False
    #config_n.model_name = args.model_name if args.model_name != None else config_n.model + '_regressor'

    cudnn.benchmark = True

    kp_regressor = get_instance(module_arch, config, 'keypoint_regressor', 'type', input_dim=config_n.keypoint_regressor['args']['input_channel'])

    if config_n.model == 'TUNIT':
        networks, opts = build_model_tunit(config_n)
        #load_model_tunit(config_n, networks, opts)
        for key in networks.keys():
            networks[key].eval()
        encoder = networks['G'].cnt_encoder

    elif config_n.model == 'MUNIT':
        encoder = AdaINGen(config['input_dim'], config['gen'])
        #last_model_name = get_model_list(config["checkpoint_dir"], 'gen')
        #state_dict = torch.load(last_model_name)
        #e_domain = config['encoder_domain']
        #encoder.load_state_dict(state_dict[e_domain])
        encoder = encoder.enc_content
        encoder.eval()

    else:
        raise(NotImplementedError)
    
    encoder = NoGradWrapper(encoder)
    model = nn.Sequential(encoder, kp_regressor)
    model = model.cuda()

    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)
    print("checkpoint from ", args.ckpt)

    Loss = getattr(module_loss, config_n.loss)
    metric = [getattr(module_metric, met) for met in config['metrics']][0]

    _ , test_loader = get_all_data_loaders(config)

    test_ioe = 0
    test_loss = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            img, meta = data['data'].cuda().detach(), data['meta']
            out = model(img)
            loss = Loss(out, meta)
            ioe = metric(out, meta, config)
            test_loss += loss
            test_ioe += ioe
        test_loss /= (j+1)
        test_ioe /= (j+1)

        print('test loss: {}'.format(test_loss))
        print('test ioe: {}'.format(test_ioe))


main()

'''
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)


# data loader
train_loader, test_loader = get_all_data_loaders(config)

lt = RegressorTrainer(config)
lt.cuda()

print("learning rate : ", lt.scheduler.get_lr()[0])
# initial mse
test_mse = 0
with torch.no_grad():
    for j, data in enumerate(test_loader):
        img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
        normed_mse = lt(img, lm)
        test_mse += normed_mse
    test_mse /= (j+1)
    print("initial test mse : ", float(test_mse))

while True:
    total_iter = 0
    for i, data in enumerate(train_loader):
        total_iter += 1
        lt.update_learning_rate()
        img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
        normed_mse, lmk = lt.fit(img, lm)

        if (i+1) % 100 == 0:
            normed_mse = float(normed_mse)
            print("----iteration {}----".format(i))
            print("learning rate : ", lt.scheduler.get_lr()[0])
            print("train_mse", normed_mse)
            train_writer.add_scalar('data/train_normed_mse', normed_mse, i)

            with torch.no_grad():
                test_mse = 0
                for j, data in enumerate(test_loader):
                    img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
                    normed_mse = lt(img, lm)
                    test_mse += normed_mse
                test_mse /= (j+1)
                print("test mse : ", float(test_mse))
                print("----------------")

            train_writer.add_scalar('data/test_normed_mse', test_mse, i)
            lt.save(checkpoint_directory, i)

    if total_iter > 100000:
        with torch.no_grad():
            test_mse = 0
            for j, data in enumerate(test_loader):
                img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
                normed_mse = lt(img, lm)
                test_mse += normed_mse
            test_mse /= (j+1)
            print("final test mse : ", float(test_mse))
        sys.exit('Finish training')
'''
