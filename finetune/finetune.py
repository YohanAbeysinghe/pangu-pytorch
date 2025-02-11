import sys
sys.path.append("/home/yohan.abeysinghe/Pangu/pangu-pytorch")

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


PATH = cfg.PG_INPUT_PATH

# Set the target GPU (2 in this case)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



output_path = os.path.join(cfg.PG_OUT_PATH, "test")
utils.mkdirs(output_path)



writer_path = os.path.join(output_path, "writer")
if not os.path.exists(writer_path):
    os.mkdir(writer_path)

writer = SummaryWriter(writer_path)



logger_name = "finetune_fully" + str(cfg.PG.HORIZON)
utils.logger_info(logger_name, os.path.join(output_path, logger_name + '.log'))

logger = logging.getLogger(logger_name)



train_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                            data_transform=None,
                            training=True,
                            validation = False,
                            startDate = cfg.PG.TRAIN.START_TIME,
                            endDate= cfg.PG.TRAIN.END_TIME,
                            freq=cfg.PG.TRAIN.FREQUENCY,
                            horizon=cfg.PG.HORIZON)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                    batch_size=cfg.PG.TRAIN.BATCH_SIZE,
                                    drop_last=True, shuffle=True, num_workers=0, pin_memory=False)


val_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                            data_transform=None,
                            training=False,
                            validation = True,
                            startDate = cfg.PG.VAL.START_TIME,
                            endDate= cfg.PG.VAL.END_TIME,
                            freq=cfg.PG.VAL.FREQUENCY,
                            horizon=cfg.PG.HORIZON)

val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=cfg.PG.VAL.BATCH_SIZE,
                                        drop_last=True, shuffle=False, num_workers=0, pin_memory=False)

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





model = PanguModel(device=device).to(device)



checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)

#Fully finetune
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = cfg.PG.TRAIN.LR , weight_decay= cfg.PG.TRAIN.WEIGHT_DECAY)



msg = '\n'
msg += utils.torch_summarize(model, show_weights=False)
logger.info(msg)


#weather_statistics = utils.LoadStatic_pretrain()
print("weather statistics are loaded!")
torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)



lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)
start_epoch = 1



model = train(model, train_loader=train_dataloader,
              val_loader=val_dataloader,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              res_path = output_path,
              device=device,
              writer=writer, logger = logger, start_epoch=start_epoch)

best_model = torch.load(os.path.join(output_path,"models/best_model.pth"),map_location='cuda:0', weights_only=False)


logger.info("Begin testing...")


test(test_loader=test_dataloader,
        model=best_model,
        device=device,
        res_path=output_path)