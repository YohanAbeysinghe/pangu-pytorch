import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys

sys.path.append("/home/yohan.abeysinghe/pangu-pytorch")
from typing import Tuple, List
import torch
import random
from torch.utils import data
from torchvision import transforms as T
import os
import time
from torch.nn.modules.module import _addindent
import matplotlib.pyplot as plt
import logging


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


def visuailze(output, target, input, var, z, step, path, cfg):
    # levels = np.linspace(-30, 90, 9)
    variables = cfg.ERA5_UPPER_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(143)

    plot1 = ax1.imshow(output[var, z, :, :], cmap="RdBu")  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('pred')

    ax2 = fig.add_subplot(142)
    plot2 = ax2.imshow(target[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('gt')

    ax3 = fig.add_subplot(141)
    plot3 = ax3.imshow(input[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('input')

    ax4 = fig.add_subplot(144)
    plot4 = ax4.imshow(output[var, z, :, :] - target[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('bias')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}_Z{}'.format(step, variables[var], z)))
    plt.close(fig)


def visuailze_surface(output, target, input, var, step, path, cfg):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(143)
    # ? to do?
    # levels = np.linspace(93000, 105000, 9)
    plot1 = ax1.imshow(output[var, :, :], cmap="RdBu")  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('pred')

    ax2 = fig.add_subplot(142)
    plot2 = ax2.imshow(target[var, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('gt')

    ax3 = fig.add_subplot(141)
    plot3 = ax3.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('input')

    ax4 = fig.add_subplot(144)
    plot4 = ax4.imshow(output[var, :, :] - target[var, :, :], cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('bias')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
    plt.close(fig)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def torch_summarize(model, show_weights=False, show_parameters=False, show_gradients=False):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    tmpstr += ', total parameters={}'.format(total_params)
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, module.parameters())])
        weights = tuple([tuple(p.size()) for p in filter(lambda p: p.requires_grad, module.parameters())])
        grads = tuple([str(p.requires_grad) for p in filter(lambda p: p.requires_grad, module.parameters())])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        if show_gradients:
            tmpstr += ', gradients={}'.format(grads)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def save_errorScores(csv_path, z, q, t, u, v, surface, error, cfg):
    score_upper_z = pd.DataFrame.from_dict(z,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_q = pd.DataFrame.from_dict(q,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_t = pd.DataFrame.from_dict(t,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_u = pd.DataFrame.from_dict(u,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_v = pd.DataFrame.from_dict(v,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_surface = pd.DataFrame.from_dict(surface,
                                orient='index',
                                columns=cfg.ERA5_SURFACE_VARIABLES)

    score_upper_z.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_z'))
    score_upper_q.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_q'))
    score_upper_t.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_t'))
    score_upper_u.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_u'))
    score_upper_v.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_v'))
    score_surface.to_csv("{}/{}.csv".format(csv_path, f'{error}_surface'))

if __name__ == "__main__":

    """
    """


