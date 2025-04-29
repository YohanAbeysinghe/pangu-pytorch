import sys
sys.path.append("/home/yohan.abeysinghe/Pangu/pangu-pytorch")

import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler

# from era5_data.config import cfg
from era5_data import utils_data, utils
from models.pangu_model import PanguModel
from models.pangu_sample import test, train

import os
import wandb
import copy
import logging
import argparse
import importlib
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter


###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config4', help='Option to load different configs')
parser.add_argument('--output', type=str, default='pm_4_29', help='Name of the output directory')
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

train_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                                         data_transform=None,
                                         training=True,
                                         validation = False,
                                         startDate = cfg.PG.TRAIN.START_TIME,
                                         endDate= cfg.PG.TRAIN.END_TIME,
                                         freq=cfg.PG.TRAIN.FREQUENCY,
                                         horizon=cfg.PG.HORIZON,
                                         cfg=cfg)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=cfg.PG.TRAIN.BATCH_SIZE,
                                   drop_last=True,
                                   shuffle=True,
                                   num_workers=0,
                                   pin_memory=False)


val_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                                       data_transform=None,
                                       training=False,
                                       validation = True,
                                       startDate = cfg.PG.VAL.START_TIME,
                                       endDate= cfg.PG.VAL.END_TIME,
                                       freq=cfg.PG.VAL.FREQUENCY,
                                       horizon=cfg.PG.HORIZON,
                                       cfg=cfg)

val_dataloader = data.DataLoader(dataset=val_dataset,
                                 batch_size=cfg.PG.VAL.BATCH_SIZE,
                                 drop_last=True,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False)

test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                                        data_transform=None,
                                        training=False,
                                        validation=False,
                                        startDate=cfg.PG.TEST.START_TIME,
                                        endDate=cfg.PG.TEST.END_TIME,
                                        freq=cfg.PG.TEST.FREQUENCY,
                                        horizon=cfg.PG.HORIZON,
                                        cfg=cfg)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=cfg.PG.TEST.BATCH_SIZE,
                                  drop_last=True,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=False)
#
###########################################################################################
################################## WandB ##################################################
###########################################################################################
#
# Initialize W&B with your project name and hyperparameters
os.environ["WANDB_API_KEY"] = "f26dcc1314b4959cd257db827dcdcff1a2e54f2e"

wandb.init(project="climate_modeling", name=args.output, config={
    "learning_rate": cfg.PG.TRAIN.LR,
    "batch_size": cfg.PG.TRAIN.BATCH_SIZE,
    "num_epochs": cfg.PG.TRAIN.EPOCHS,
    # "num_gpus": num_gpus,
    "start_time": cfg.PG.TRAIN.START_TIME,
    "end_time": cfg.PG.TRAIN.END_TIME,
    "output_path": output_path,
    "pm2.5_weightage": cfg.PG.TRAIN.SURFACE_WEIGHTS[4],
    "Law_rank": cfg.PG.TRAIN.Low_Rank,
})
#
###########################################################################################
###################################Loading Checkpoint######################################
###########################################################################################
#
model = PanguModel(device=device, cfg=cfg).to(device)

module_copy = copy.deepcopy(model) # For later comparisons

checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=False, map_location='cuda')
state_dict = checkpoint['model']
#
###########################################################################################
################### Editing Checkpoint to Get New Variables ###############################
###########################################################################################
#
if cfg.GLOBAL.MODEL == 'pm1':
    # Learning rate for new variables.
    model_state_dict = model.state_dict()

    # Modify input layer for dimension matching and loading the existing weights
    # for first 112 channels. Rest is initialized randomly.
    new_input_weight = torch.zeros((192, 128, 1))
    new_input_weight[:, :112, :] = state_dict['_input_layer.conv_surface.weight']
    nn.init.xavier_uniform_(new_input_weight[:, 112:, :])
    state_dict['_input_layer.conv_surface.weight'] = new_input_weight

    # Modify output layer for dimension matching and loading the existing weights
    # for first 64 channels. Rest is initialized randomly.
    new_output_weight = torch.zeros((80, 384, 1))
    new_output_weight[:64, :, :] = state_dict['_output_layer.conv_surface.weight']
    nn.init.xavier_uniform_(new_output_weight[64:, :, :])
    state_dict['_output_layer.conv_surface.weight'] = new_output_weight

    # Modify output layer bias. Loading first 64 biases.
    new_output_bias = torch.zeros(80)
    new_output_bias[:64] = state_dict['_output_layer.conv_surface.bias']
    state_dict['_output_layer.conv_surface.bias'] = new_output_bias

# Load the modified state_dict if cfg.GLOBAL.MODEL == 'pm25'.
model.load_state_dict(state_dict, strict=False)
#
###########################################################################################
#####################################  PEFT  ##############################################
###########################################################################################
#
# print([(n, type(m)) for n, m in model.named_modules()])

target_modules = []

for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        target_modules.append(n)
        print(f"appended {n}")

config = LoraConfig(
    r=cfg.PG.TRAIN.Low_Rank,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias = "none",
    # modules_to_save=["_output_layer.conv_surface","_output_layer.conv"]
)

model = get_peft_model(model, config)

# if cfg.GLOBAL.MODEL == 'original':
#     #Fully finetune
#     for param in model.parameters():
#         param.requires_grad = True

if cfg.GLOBAL.MODEL == 'pm1':
    # Fine-tuning layers (MENA scaling)
    for param in model.parameters():
        param.requires_grad = False

    # Set requires_grad for edited layers
    for param in model._input_layer.conv_surface.parameters():
        param.requires_grad = True
    for param in model._output_layer.conv_surface.parameters():
        param.requires_grad = True

    # Unfreeze LoRA weights
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

# before_weights = {
#     "input": peft_model._input_layer.conv_surface.weight.clone().detach(),
#     "output": peft_model._output_layer.conv_surface.weight.clone().detach(),
# }

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad,model.parameters()),
    lr = cfg.PG.TRAIN.LR ,
    weight_decay= cfg.PG.TRAIN.WEIGHT_DECAY)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     milestones=[25, 50],
#     gamma=0.5
#     )


start_epoch = 1
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
msg = '\n'
msg += utils.torch_summarize(model, show_weights=False)
logger.info(msg)

print("weather statistics are loaded!")
#
###########################################################################################
############################## Train and Validation #######################################
###########################################################################################
#
model = train(
        model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
        res_path = output_path,
        device=device,
        writer=writer, 
        logger = logger,
        start_epoch=start_epoch,
        cfg = cfg
        # rank=local_rank
        )

###########################################################################################
################################### Testing  ##############################################
###########################################################################################
#
# best_model = torch.load(
#     os.path.join(output_path,"models/best_model.pth"),
#     map_location='cuda',
#     weights_only=False
#     )

# logger.info("Begin testing...")

# print(f"Length of test_loader: {len(test_dataloader)}")


# test(test_loader=test_dataloader,
#     model=best_model,
#     device=device,
#     res_path=output_path,
#     cfg = cfg
#     )
#
###########################################################################################
###########################################################################################