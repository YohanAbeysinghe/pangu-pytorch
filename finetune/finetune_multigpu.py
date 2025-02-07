import sys
sys.path.append("/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/models/pangu-pytorch")

from era5_data import utils_data, utils
from era5_data.config import cfg

import os
import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import logging

from models.pangu_model import PanguModel
from models.pangu_sample import test, train
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


PATH = cfg.PG_INPUT_PATH

# Initialize distributed training
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    return rank, world_size

rank, world_size = init_distributed_mode(None)
device = torch.device("cuda", rank)

print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

output_path = os.path.join(cfg.PG_OUT_PATH, "test_run_3", str(cfg.PG.HORIZON))
utils.mkdirs(output_path)

writer_path = os.path.join(output_path, "writer")
if not os.path.exists(writer_path):
    os.mkdir(writer_path)

writer = SummaryWriter(writer_path)

logger_name = "finetune_fully" + str(cfg.PG.HORIZON)
utils.logger_info(logger_name, os.path.join(output_path, logger_name + '.log'))

logger = logging.getLogger(logger_name)

# DDP requires a DistributedSampler to ensure each GPU gets a distinct portion of data
train_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                            data_transform=None,
                            training=True,
                            validation=False,
                            startDate=cfg.PG.TRAIN.START_TIME,
                            endDate=cfg.PG.TRAIN.END_TIME,
                            freq=cfg.PG.TRAIN.FREQUENCY,
                            horizon=cfg.PG.HORIZON)

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
train_dataloader = data.DataLoader(dataset=train_dataset,
                                    batch_size=cfg.PG.TRAIN.BATCH_SIZE,
                                    drop_last=True, shuffle=False, num_workers=0, pin_memory=False, sampler=train_sampler)

val_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                            data_transform=None,
                            training=False,
                            validation=True,
                            startDate=cfg.PG.VAL.START_TIME,
                            endDate=cfg.PG.VAL.END_TIME,
                            freq=cfg.PG.VAL.FREQUENCY,
                            horizon=cfg.PG.HORIZON)

val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=cfg.PG.VAL.BATCH_SIZE,
                                        drop_last=True, shuffle=False, num_workers=0, pin_memory=False, sampler=val_sampler)

test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                                    data_transform=None,
                                    training=False,
                                    validation=False,
                                    startDate=cfg.PG.TEST.START_TIME,
                                    endDate=cfg.PG.TEST.END_TIME,
                                    freq=cfg.PG.TEST.FREQUENCY,
                                    horizon=cfg.PG.HORIZON)

test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                    drop_last=True, shuffle=False, num_workers=0, pin_memory=False)

# Initialize model
model = PanguModel(device=device).to(device)

# Initialize DDP model
model = DDP(model, device_ids=[rank])

checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch)
model.load_state_dict(checkpoint['model'], strict=False)

# Fully finetune
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.PG.TRAIN.LR, weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)

if rank == 0:
    msg = '\n'
    msg += utils.torch_summarize(model, show_weights=False)
    logger.info(msg)

# Weather statistics loaded message
if rank == 0:
    print("Weather statistics are loaded!")

torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)
start_epoch = 1

# Train the model
model = train(model, train_loader=train_dataloader,
              val_loader=val_dataloader,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              res_path=output_path,
              device=device,
              writer=writer, logger=logger, start_epoch=start_epoch)

# Only rank 0 should save the best model
if rank == 0:
    best_model = torch.load(os.path.join(output_path, "models/best_model.pth"), map_location='cuda:0')

    logger.info("Begin testing...")

    test(test_loader=test_dataloader,
         model=best_model,
         device=device,
         res_path=output_path)