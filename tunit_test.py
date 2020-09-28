import os
import torch
import argparse

from torch import nn

from utils._utils import get_config, get_model_list, get_scheduler
from utils.data_loader import get_all_data_loaders

from models.functions import spatial_logsoftmax
from models.tunit.generator import Generator as Generator
from models.tunit.discriminator import Discriminator as Discriminator
from models.tunit.guidingNet import GuidingNet
from models.tunit.inception import InceptionV3

ckpt = 'model_60.ckpt'

parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--dataset', default='animal_faces', help='Dataset name to use',
                            choices=['afhq_cat', 'afhq_dog', 'afhq_wild', 'animal_faces', 'photo2ukiyoe', 'summer2winter', 'lsun_car', 'ffhq'])
parser.add_argument('--data_path', type=str, default='../data',
                            help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='GAN',
                            help='Prefix of logs and results folders. '
                                                     'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=32, type=int,
                            help='Batch size for training')
parser.add_argument('--val_batch', default=10, type=int,
                            help='Batch size for validation. '
                                                     'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=10, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=128, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

parser.add_argument('--p_semi', default=0.0, type=float,
                            help='Ratio of labeled data '
                                                     '0.0 = unsupervised mode'
                                                                              '1.0 = supervised mode'
                                                                                                       '(0.0, 1.0) = semi-supervised mode')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                            help='path to latest checkpoint (default: None)'
                                                     'ex) --load_model GAN_20190101_101010'
                                                                              'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true',
                            help='Call for valiation only mode')

parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                            help='GPU id to use.')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8989', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')


args = parser.parse_args()
args.distributed = False
args.gpu = 0
args.log_dir = './logs/celeba_tunit'
args.multiprocessing_distributed = False


class RegressorTrainer(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        lr = config['lr']
        self.encoder = encoder
        
        # freeze network
        for p in self.encoder.parameters():
            p.requires_grad = False

        img_dim = (1, 3, config["data"]['transform']['Resize'], config["data"]['transform']['Resize'])
        content = self.encoder(torch.rand(*img_dim).cuda())
        self.unsup_landmarks = int(config['unsup_landmarks'])
        self.c = content.shape[1]
        self.w, self.h = content.shape[2], content.shape[3]
        self.l_in = self.unsup_landmarks * 2
        self.l_out = int(config['num_landmarks']) * 2
        self.conv = nn.Conv2d(self.c, self.unsup_landmarks, 1)
        self.linear = nn.Linear(self.l_in, self.l_out, bias=False)

        lr = config['lr']
        beta1 = config['beta1']
        beta2 = config['beta2']
        wd = config['weight_decay']
        self.opt = torch.optim.Adam([p for p in self.linear.parameters()] + [p for p in self.conv.parameters()], lr=lr, betas=(beta1, beta2), weight_decay=wd)

        self.scheduler = get_scheduler(self.opt, config)
        #self.loss = nn.MSELoss()

    def print_params(self):
        print('--------')
        for p in self.linear.parameters():
            print("linear : ", p[0])
            break
        print('---')
        for p in self.gen.parameters():
            print("content : ", p[0])
            break

        print('--------')

    # train linear layer
    def fit(self, images, y):
        self.opt.zero_grad()

        y_flat = y.view(y.shape[0], -1).float()
        yp = self.encode(images)
        #out = self.loss(yp, y_flat)
        #out.backward()

        inter_occs = torch.sqrt(torch.sum((y[:,0,:] - y[:,1,:])**2, -1))
        yp = yp.reshape(-1,5,2)
        distances = torch.sqrt(torch.sum((y-yp)**2,-1))
        out = torch.mean(distances)
        out.backward()
        self.opt.step()

        normed_mse = torch.mean(distances / inter_occs[:, None])
        return normed_mse, yp

    def encode(self, images):
        content = self.encoder(images)
        bs = content.shape[0]
        height = content.shape[2]
        width = content.shape[3]
        out = self.conv(content)
        # unsupervised landmarks
        # N x C x 2
        ulmk = spatial_logsoftmax(out) * torch.tensor([height, width]).float().cuda()
        ulmk = ulmk.view(bs, -1)
        lmk = self.linear(ulmk)
        return lmk

    def infer(self, images):
        for p in self.linear.parameters():
            p.requires_grad = False

        yp = self.encode(images)
        yp = yp.reshape(-1,5,2)

        for p in self.linear.parameters():
            p.requires_grad = True

        return yp
    
    def save(self, save_dir, iterations):
        name = os.path.join(save_dir, 'linear_%08d.pt' % (iterations + 1))
        torch.save(self.linear.state_dict(), name)

    def load(self, checkpoint_dir):
        last_model_name = get_model_list(checkpoint_dir, "linear")
        state_dict = torch.load(last_model_name)
        self.linear.load_state_dict(state_dict)

    def forward(self, images, y):
        yp = self.encode(images)

        inter_occs = torch.sqrt(torch.sum((y[:,0,:] - y[:,1,:])**2, -1))
        yp = yp.reshape(-1,5,2)
        distances = torch.sqrt(torch.sum((y-yp)**2,-1))
        normed_mse = torch.mean(distances / inter_occs[:, None])
        return normed_mse

    def update_learning_rate(self):
        self.scheduler.step()

def build_model(args):
    args.to_train = 'CDGI'

    networks = {}
    opts = {}
    is_semi = (0.0 < args.p_semi < 1.0)
    if is_semi:
        assert 'SEMI' in args.train_mode
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

    if args.distributed:
        if args.gpu is not None:
            print('Distributed to', args.gpu)
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            args.workers = int(args.workers / args.ngpus_per_node)
            for name, net in networks.items():
                if name in ['inceptionNet']:
                    continue
                net_tmp = net.cuda(args.gpu)
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[args.gpu], output_device=args.gpu)
        else:
            for name, net in networks.items():
                net_tmp = net.cuda()
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp)

    elif args.gpu is not None:
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
        if args.distributed:
            networks['C_EMA'].module.load_state_dict(networks['C'].module.state_dict())
        else:
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

def load_model(args, to_restore, networks, opts):
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

networks, opts = build_model(args)
load_model(args, ckpt, networks, opts)

networks['D'].eval()
networks['G'].eval()
networks['C'].eval()
networks['C_EMA'].eval()
networks['G_EMA'].eval()

#x = torch.rand(1,3,256,256).cuda().detach()
encoder = networks['G'].cnt_encoder
configdir = 'configs/mafl/mafl_regressor.yaml'
config = get_config(configdir)
lt = RegressorTrainer(config, encoder).cuda()
train_loader, test_loader = get_all_data_loaders(config)

test_mse = 0
with torch.no_grad():
    for j, data in enumerate(test_loader):
        img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
        normed_mse = lt(img, lm)
        test_mse += normed_mse
    test_mse /= (j+1)
    print('initial test mse :', float(test_mse))

while True:
    total_iter = 0
    for i, data in enumerate(train_loader):
        total_iter += 1
        lt.update_learning_rate()
        img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
        normed_mse, lmk = lt.fit(img, lm)

        if (i+1) % 100 == 0:
            normed_mse = float(normed_mse)
            print('----iteration {}----'.format(i))
            print('learning rate :', lt.scheduler.get_lr()[0])
            print('train_mse', normed_mse)
            #train_writer.add_scalar('data/train_normed_mse', normed_mse, i)

            with torch.no_grad():
                test_mse = 0
                for j, data in enumerate(test_loader):
                    img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
                    normed_mse = lt(img, lm)
                    test_mse += normed_mse
                test_mse /= (j+1)
                print('test mse : ', float(test_mse))
                print("--------------")

            #train_writer.add_scalar('data/test_normed_mse', test_mse, i)
            #lt.save(c)
    if total_iter > 100000:
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                img, lm = data['data'].cuda().detach(), data['meta']['keypts_normalized'].cuda().detach()
                normed_mse = lt(img, lm)
                test_mse += normed_mse
            test_mse /= (j+1)
            print('initial test mse :', float(test_mse))
        sys.exit('Finish training')
