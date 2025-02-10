import sys
sys.path.append("/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/models/pangu-pytorch")

import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler

from datetime import timedelta
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# from era5_data.config import cfg
from era5_data import utils_data, utils
from models.pangu_model import PanguModel
from models.pangu_sample import test, train

import os
import logging
import argparse
import importlib
from tensorboardX import SummaryWriter

###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config1', help='Option to load different configs')
parser.add_argument('--output', type=str, default='test', help='Name of the output directory')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg
#
torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)
#
###########################################################################################
############################## Distributed Training #######################################
###########################################################################################
#
def setup_distributed():
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}")
print(f"Using device: {device}")

num_gpus = torch.cuda.device_count()
if local_rank == 0:
    print(f"Number of GPUs available: {num_gpus}")
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
output_path = os.path.join(cfg.PG_OUT_PATH, args.output)
utils.mkdirs(output_path)

writer_path = os.path.join(output_path, "writer")
if not os.path.exists(writer_path):
    os.mkdir(writer_path)
writer = SummaryWriter(writer_path)

logger_name = "finetune_fully" + str(cfg.PG.HORIZON)
utils.logger_info(logger_name, os.path.join(output_path, logger_name + '.log'))
logger = logging.getLogger(logger_name)
#
###########################################################################################
################################### Data Loading ##########################################
###########################################################################################
#
PATH = cfg.PG_INPUT_PATH

train_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=True,
    validation = False,
    startDate = cfg.PG.TRAIN.START_TIME,
    endDate= cfg.PG.TRAIN.END_TIME,
    freq=cfg.PG.TRAIN.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

train_sampler = DistributedSampler(
    train_dataset,
    shuffle=True,
    drop_last=True
    )

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.PG.TRAIN.BATCH_SIZE//num_gpus,
    num_workers=0,
    pin_memory=False,
    sampler=train_sampler
    )


val_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=False,
    validation = True,
    startDate = cfg.PG.VAL.START_TIME,
    endDate= cfg.PG.VAL.END_TIME,
    freq=cfg.PG.VAL.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

val_sampler = DistributedSampler(
    val_dataset,
    shuffle=True,
    drop_last=True
    )

val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=cfg.PG.VAL.BATCH_SIZE//num_gpus,
    num_workers=0,
    pin_memory=False,
    sampler=val_sampler
    )

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
#
###########################################################################################
################################Loading Checkpoint ########################################
###########################################################################################
#
model = PanguModel(device=device, cfg=cfg).to(device)

checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
#
###########################################################################################
####################################Hyperparameters########################################
###########################################################################################
#
#Fully finetune
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = cfg.PG.TRAIN.LR,
    weight_decay= cfg.PG.TRAIN.WEIGHT_DECAY
    )

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[25, 50],
    gamma=0.5
    )

start_epoch = 1
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
if local_rank == 0:
    msg = '\n'
    msg += utils.torch_summarize(model, show_weights=False)
    logger.info(msg)

print("weather statistics are loaded!")
#
###########################################################################################
############################## Train and Validation #######################################
###########################################################################################
#
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

model = train(
    model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    res_path = output_path,
    device=device,
    writer=writer, 
    logger = logger,
    start_epoch=start_epoch,
    cfg = cfg
    )

cleanup_distributed()
#
###########################################################################################
################################### Testing  ##############################################
###########################################################################################
#
best_model = torch.load(os.path.join(output_path,"models/best_model.pth"),
                        map_location='cuda:0',
                        weights_only=False)

logger.info("Begin testing...")

test(test_loader=test_dataloader,
     model=best_model,
     device=device,
     res_path=output_path,
     cfg = cfg)
#
###########################################################################################
###########################################################################################