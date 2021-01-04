import numpy as np
import torch.nn.functional as F


def inter_ocular_error(output, meta, config):
    dataset = config['data']['dataset']
    #eyeidxs = dataset.eye_kp_idxs
    if dataset == 'MAFLAligned':
        eyeidxs = [0,1]
    elif dataset == 'ThreeHundredW':
        eyeidxs = [36, 45]
    elif dataset == 'AFLW':
        eyeidxs = [0,1]

    pred = output[0].detach().cpu()
    gt = meta['keypts_normalized']
    iod = ((gt[:, eyeidxs[0], :] - gt[:, eyeidxs[1], :])**2.).sum(1).sqrt()[:, None]
    err = ((pred - gt)**2.).sum(2).sqrt()
    ioderr = err / iod
    return 100 * ioderr.mean()
