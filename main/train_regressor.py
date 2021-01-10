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

## STARGAN-V2 ###
from models.stargan_v2.model import Generator as S_Generator
from models.stargan_v2.model import Stargan_enc

import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mafl/mafl_regressor_tunit.yaml', help='Path to the config file.')
parser.add_argument('--model_name', default=None, type=str, metavar='PATH')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--from_scratch', default=False, type=bool, help='train whole model from scratch')
parser.add_argument('--save_name', default=None, type=str, help='save folder name')

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

def build_encoder_stargan_v2(args):
    stargan_gen = S_Generator()
    if not args.from_scratch:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            stargan_gen.load_state_dict(checkpoint['generator'])
        else:
            return None
    encoder = Stargan_enc(stargan_gen)
    return encoder



# model builder for tunit
def build_model_tunit(args):
    args.to_train = 'CDGI'

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

def plot_imgs(model, display_data, epoch):
    with torch.no_grad():
        metric = [getattr(module_metric, met) for met in config['metrics']][0]
        data = display_data['data'].cuda().detach()
        out = model(data)
        pred = out[0]
        #fig, axs = plt.subplots(2,5, constrained_layout=True, figsize=(30,30))
        fig, axs = plt.subplots(2,5, figsize=(30,13))
        #plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0, wspace=0.4)

        for i in range(10):
            img = display_data['data'][i].cpu().permute(1,2,0)
            kp = display_data['meta']['keypts'][i]
            kp_pred = pred[i].detach().cpu()
            kp_pred = (kp_pred + 1) / 2 * 255
            _out = (out[0][i,:,:], out[1][i,:,:])
            _meta = {}
            _meta['keypts_normalized'] = display_data['meta']['keypts_normalized'][i].unsqueeze(0)
            ioe = metric(_out, _meta, config)
            row = i // 5
            col = i % 5
            axs[row][col].imshow(img)
            axs[row][col].scatter(kp[:,0], kp[:,1], color='red')
            axs[row][col].scatter(kp_pred[:,0], kp_pred[:,1], color='blue')
        plt.savefig('{}/{}_{}.png'.format(config_n.img_dir, str(epoch), ioe))

def main():
    args = parser.parse_args()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        print('[use gpu] ',args.gpu)

    if args.from_scratch:
        print('train from scratch')

    global config
    global config_n
    config = get_config(args.config)


    config_n = Namespace(**config)
    config_n.gpu = args.gpu
    config_n.distributed = False
    config_n.multiprocessing_distributed = False
    config_n.model_name = args.model_name if args.model_name != None else config_n.model + '_regressor'

    print('model name: ', config_n.model_name)

    makedirs('./logs')
    makedirs('./results')
    
    config_n.log_dir = os.path.join('./logs', config_n.model_name)
    config_n.regressor_log_dir = os.path.join(config_n.log_dir, 'regressor_logs', config['data']['dataset']) if args.save_name == None else os.path.join('./logs', args.save_name, 'regressor_logs', config['data']['dataset'])
    makedirs(config_n.regressor_log_dir)

    config_n.res_dir = os.path.join('./results', config_n.model_name, config['data']['dataset']) if args.save_name == None else os.path.join('./resutls', args.save_name, config['data']['dataset'])
    config_n.img_dir = os.path.join(config_n.res_dir, 'regressor_imgs')
    config_n.eval_results = os.path.join(config_n.res_dir, 'eval_results.txt')
    makedirs(config_n.img_dir)

    cudnn.benchmark = True

    kp_regressor = get_instance(module_arch, config, 'keypoint_regressor', 'type', input_dim=config_n.keypoint_regressor['args']['input_channel'])

    if config_n.model == 'TUNIT':
        networks, opts = build_model_tunit(config_n)
        if not args.from_scratch:
            load_model_tunit(config_n, networks, opts)
        encoder = networks['G'].cnt_encoder

    elif config_n.model == 'MUNIT':
        encoder = AdaINGen(config['input_dim'], config['gen'])
        if not args.from_scratch:
            last_model_name = get_model_list(config["checkpoint_dir"], 'gen')
            state_dict = torch.load(last_model_name)
            e_domain = config['encoder_domain']
            encoder.load_state_dict(state_dict[e_domain])
        encoder = encoder.enc_content

    elif config_n.model == 'STARGAN_V2':
        encoder = build_encoder_stargan_v2(config_n)

    else:
        raise(NotImplementedError)
    
    if args.from_scratch:
        pass
    else:
        encoder = NoGradWrapper(encoder).eval()
    model = nn.Sequential(encoder, kp_regressor)
    model = model.cuda()

    Loss = getattr(module_loss, config_n.loss)
    metric = [getattr(module_metric, met) for met in config['metrics']][0]

    #all_params = list(model.parameters())
    trainable_params = list(filter(lambda p: p.requires_grad, kp_regressor.parameters()))

    optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **config['optimizer']['args'])
    lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(optimizer, **config['lr_scheduler']['args'])


    train_loader, test_loader = get_all_data_loaders(config)
    display_data = list(test_loader)[0]

    plot_imgs(model, display_data, 'before_training')

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



    for epoch in range(config['epoch']):
        print('%%%% start epoch {} %%%%'.format(epoch+1))
        epoch_loss = 0
        epoch_ioe = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img, meta = data['data'].cuda().detach(), data['meta']
            out = model(img)
            loss = Loss(out, meta)
            ioe = metric(out, meta, config)
            epoch_ioe += ioe
            epoch_loss += loss

            loss.backward()
            optimizer.step()
        epoch_loss /= (batch_idx + 1)
        epoch_ioe /= (batch_idx + 1)

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

        _eval_txt = '--- epoch {} ---\n'.format(epoch+1) + 'train loss: {} / test loss: {}\n'.format(epoch_loss, test_loss) + 'train ioe: {} / test ioe: {}\n'.format(epoch_ioe, test_ioe) + 'learning rate: {}\n'.format(lr_scheduler.get_lr()[0])
        print(_eval_txt)
        with open(config_n.eval_results, 'a') as eval_txt:
            eval_txt.write(_eval_txt)

        lr_scheduler.step()

        if (epoch+1) % 10 == 0:
            # save image
            plot_imgs(model, display_data, epoch+1)

            # save model

            save_path = '{}/model_{}.ckpt'.format(config_n.regressor_log_dir, epoch+1)
            torch.save(model.state_dict(), save_path)
        

main()
