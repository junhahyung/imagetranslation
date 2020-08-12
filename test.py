import os
import sys
import torch
import argparse
import tensorboardX
import torch.backends.cudnn as cudnn

from torch import nn
from utils import get_all_landmark_loaders, get_config, prepare_sub_folder
from tester import Landmark_Tester

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mafl_test.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help='outputs path')
opts = parser.parse_args()

model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)


cudnn.benchmark = True
config = get_config(opts.config)

# data loader
train_loader, test_loader = get_all_landmark_loaders(config)

lt = Landmark_Tester(config)
lt.cuda()

print("learning rate : ", lt.scheduler.get_lr()[0])
# initial mse
test_mse = 0
with torch.no_grad():
    for j, (img, lm) in enumerate(test_loader):
        img, lm = img.cuda().detach(), lm.cuda().detach()
        normed_mse = lt(img, lm)
        test_mse += normed_mse
    test_mse /= (j+1)
    print("initial test mse : ", float(test_mse))

while True:
    for i, (img, lm) in enumerate(train_loader):
        lt.update_learning_rate()
        img, lm = img.cuda().detach(), lm.cuda().detach()
        normed_mse, lmk = lt.fit(img, lm)

        if (i+1) % 500 == 0:
            normed_mse = float(normed_mse)
            print("----iteration {}----".format(i))
            print("learning rate : ", lt.scheduler.get_lr()[0])
            print("train_mse", normed_mse)
            train_writer.add_scalar('data/train_normed_mse', normed_mse, i)

            with torch.no_grad():
                test_mse = 0
                for j, (img, lm) in enumerate(test_loader):
                    img, lm = img.cuda().detach(), lm.cuda().detach()
                    normed_mse = lt(img, lm)
                    test_mse += normed_mse
                test_mse /= (j+1)
                print("test mse : ", float(test_mse))
                print("----------------")

            train_writer.add_scalar('data/test_normed_mse', test_mse, i)
            lt.save(checkpoint_directory, i)

        if i > 100000:
            with torch.no_grad():
                test_mse = 0
                for j, (img, lm) in enumerate(test_loader):
                    img, lm = img.cuda().detach(), lm.cuda().detach()
                    normed_mse = lt(img, lm)
                    test_mse += normed_mse
                test_mse /= (j+1)
                print("final test mse : ", float(test_mse))
            sys.exit('Finish training')