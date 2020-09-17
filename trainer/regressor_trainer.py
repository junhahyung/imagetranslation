import os
import torch

from torch import nn
from models.functions import spatial_logsoftmax
from utils._utils import get_model_list, get_scheduler

from models.munit_networks import AdaINGen


class RegressorTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        lr = config['lr']
        
        self.gen = AdaINGen(config['input_dim'], config['gen'])
        last_model_name = get_model_list(config["checkpoint_dir"], "gen")
        state_dict = torch.load(last_model_name)
        e_domain = config["encoder_domain"]
        self.gen.load_state_dict(state_dict[e_domain])
        self.gen.eval()
        
        # freeze network
        for p in self.gen.parameters():
            p.requires_grad = False

        img_dim = (1, 3, config["data"]['transform']['Resize'], config["data"]['transform']['Resize'])
        content, _ = self.gen.encode(torch.rand(*img_dim))
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
        content, style = self.gen.encode(images)
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
