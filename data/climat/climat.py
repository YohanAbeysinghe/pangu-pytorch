import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# from era5_data.config import cfg
from era5_data import utils_data, utils
from models.pangu_model import PanguModel
from models.test_mena import test
# from models.pangu_sample import test

import os
import wandb
import copy
import logging
import argparse
import importlib
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter

def normData(upper, surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (statistics[0], statistics[1], statistics[2], statistics[3])

    upper_mean = upper_mean.permute(3, 0, 1, 2).unsqueeze(0)  # [1, V, L, 1, 1]
    upper_std = upper_std.permute(3, 0, 1, 2).unsqueeze(0)    #

    surface_mean = surface_mean.view(1, 7, 1, 1)  # [1, 7, 1, 1]
    surface_std = surface_std.view(1, 7, 1, 1)    # same

    upper = (upper - upper_mean) / upper_std
    surface = (surface - surface_mean) / surface_std
    return upper, surface

def test(test_loader, device, res_path, cfg):
    # set up empty dics for rmses and anormaly correlation coefficients

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)
        
    # Initialize accumulators
    running_sum = None
    count = 0

    for id, data in enumerate(test_loader, 0):
        
        # Check if any of the data components are empty tensors
        if (data[0].sum() == 0 or
            data[1].sum() == 0 or
            data[2].sum() == 0 or
            data[3].sum() == 0):
            # print(f"Skipping batch {id} due to missing or empty data.")
            continue  # Skip this batch if any data component is empty

        # Store initial input for different models
        print(f"predict on {id}")
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        input_test, input_surface_test, target_test, target_surface_test = input_test.to(device), input_surface_test.to(device), target_test.to(device), target_surface_test.to(device)


        if cfg.GLOBAL.MODEL == 'All_pm':
            #Log scaling
            scale = torch.log(torch.tensor(1e20))
            input_surface_test[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface_test[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)


        input, input_surface = normData(input_test, input_surface_test, aux_constants['weather_statistics'])

        input = input.reshape(1, -1, 721, 1440)
        condition = torch.cat([input, input_surface[:, 3:, :, :]], dim=1)
        condition = condition.squeeze(0)  # Add batch dimension

        if running_sum is None:
            running_sum = torch.zeros_like(condition)

        running_sum += condition
        count += 1

    # Calculate the running average
    if count > 0:
        running_avg = running_sum / count
        print("Running average calculated successfully.")
    else:
        running_avg = None
        print("No valid batches were processed.")

    return running_avg



###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config30', help='Option to load different configs')
parser.add_argument('--output', type=str, default='lora_full_finetune_loss_cropped', help='Name of the output directory')
parser.add_argument('--distri', default=False, help='Doing the distributed training')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
local_rank=0
num_gpus = 1

output_path = os.path.join(cfg.PG_OUT_PATH, args.output)
#
###########################################################################################
###########################################################################################
###########################################################################################
PATH = cfg.PG_INPUT_PATH


test_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=False,
    validation=False,
    startDate=cfg.PG.TEST.START_TIME,
    endDate=cfg.PG.TEST.END_TIME,
    freq=cfg.PG.TEST.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    batch_size=cfg.PG.TEST.BATCH_SIZE,
    drop_last=True,
    shuffle=False,
    num_workers=0,
    pin_memory=False
    )



running_average = test(test_loader=test_dataloader,
                        device=device,
                        res_path=output_path,
                        cfg = cfg)
                    #

torch.save(running_average, 'running_average.pt')