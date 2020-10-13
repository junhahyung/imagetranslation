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
from utils._utils import get_instance, NoGradWrapper
import models.model as module_arch
import models.metric as module_metric

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
parser.add_argument('--load_model', default=None, type=str, metavar='PATH')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')

# tunit model loader
def load_model(args, networks, opts):
    if args.load_model is not None:
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
    if args.load_model is None:
        return

    config = get_config(args.config)


    config_n = Namespace(**config)
    config_n.gpu = args.gpu
    config_n.distributed = False
    config_n.multiprocessing_distributed = False
    config_n.load_model = args.load_model
    #config.model_name = '{}-{}_{}'.format(config.load_model, 'regressor', datetime.now().strftime("%Y%m%d-%H%M%S"))
    config_n.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')
    
    config_n.log_dir = os.path.join('./logs', config_n.model_name)
    makedirs(config_n.log_dir)
    config_n.event_dir = os.path.join(config_n.log_dir, 'events')
    config_n.res_dir = os.path.join('./results', config_n.model_name)
    config_n.img_dir = os.path.join(config_n.res_dir, 'regressor_imgs')
    makedirs(config_n.img_dir)
    config_n.img_dir = os.path.join(config_n.img_dir, config['data']['dataset'])
    makedirs(config_n.img_dir)
    config_n.regressor_log_dir = os.path.join(config_n.log_dir, 'regressor_logs')
    makedirs(config_n.regressor_log_dir)
    config_n.regressor_log_dir = os.path.join(config_n.regressor_log_dir, config['data']['dataset'])
    makedirs(config_n.regressor_log_dir)
    
    cudnn.benchmark = True

    kp_regressor = get_instance(module_arch, config, 'keypoint_regressor', 'type', input_dim=config_n.keypoint_regressor['args']['input_channel'])

    if config_n.model == 'TUNIT':
        networks, opts = build_model_tunit(config_n)
        load_model(config_n, networks, opts)
        for key in networks.keys():
            networks[key].eval()
        encoder = networks['G'].cnt_encoder
    
    encoder = NoGradWrapper(encoder)
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

    data = display_data['data'].cuda().detach()
    out = model(data)
    pred = out[0]
    for i in range(10):
        fig, axs = plt.subplots(1,2)
        img = display_data['data'][i].cpu().permute(1,2,0)
        kp = display_data['meta']['keypts'][i]
        kp_pred = pred[i].detach().cpu()
        kp_pred = (kp_pred + 1) / 2 * 255
        _out = (out[0][i,:,:], out[1][i,:,:])
        _meta = {}
        _meta['keypts_normalized'] = display_data['meta']['keypts_normalized'][i].unsqueeze(0)
        ioe = metric(_out, _meta, config)
        axs[0].imshow(img)
        axs[1].imshow(img)
        axs[0].scatter(kp[:,0], kp[:,1])
        axs[1].scatter(kp_pred[:,0], kp_pred[:,1])
        plt.savefig('{}/{}_{}_{}.png'.format(config_n.img_dir, 'before_training', ioe, i+1))

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

        print('--- epoch {} ---'.format(epoch+1))
        print('train loss: {} / test loss: {}'.format(epoch_loss, test_loss))
        print('train ioe: {} / test ioe: {}'.format(epoch_ioe, test_ioe))
        print('learning rate: {}'.format(lr_scheduler.get_lr()[0]))

        lr_scheduler.step()

        if (epoch+1) % 10 == 0:
            # save image
            data = display_data['data'].cuda().detach()
            out = model(data)
            pred = out[0]
            for i in range(10):
                fig, axs = plt.subplots(1,2)
                img = display_data['data'][i].cpu().permute(1,2,0)
                kp = display_data['meta']['keypts'][i]
                kp_pred = pred[i].detach().cpu()
                kp_pred = (kp_pred + 1) / 2 * 255
                _out = (out[0][i,:,:], out[1][i,:,:])
                _meta = {}
                _meta['keypts_normalized'] = display_data['meta']['keypts_normalized'][i].unsqueeze(0)
                ioe = metric(_out, _meta, config)
                axs[0].imshow(img)
                axs[1].imshow(img)
                axs[0].scatter(kp[:,0], kp[:,1])
                axs[1].scatter(kp_pred[:,0], kp_pred[:,1])
                plt.savefig('{}/{}_{}_{}.png'.format(config_n.img_dir, epoch+1, ioe, i+1))

            # save model

            save_path = '{}/model_{}.ckpt'.format(config_n.regressor_log_dir, epoch+1)
            torch.save(model.state_dict(), save_path)
        






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
