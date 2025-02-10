import os
import time
import logging
import argparse
import importlib

import torch
from torch.utils import data

import sys
sys.path.append("/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/models/pangu-pytorch")

from era5_data import utils, utils_data
from models.pangu_model import PanguModel
from models.pangu_sample import test

starts  = time.time()


###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config1', help='Option to load different configs')
parser.add_argument('--type_net', type=str, default='reproduce_mask0', help='Name of the output directory')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg

torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)
#
###########################################################################################
################################## GPU Setup ##############################################
###########################################################################################
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Predicting on {device}")
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
PATH = cfg.PG_INPUT_PATH

output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net)
utils.mkdirs(output_path)

logger_name = args.type_net + str(cfg.PG.HORIZON)
utils.logger_info(logger_name, os.path.join(output_path, logger_name + '_test.log'))

logger = logging.getLogger(logger_name)
#
###########################################################################################
################################### Data Loading ##########################################
###########################################################################################
#
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
model = PanguModel(device=device).to(device)
checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch)
model.load_state_dict(checkpoint['model'])
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
logger.info("Begin Test")
msg = '\n'
msg += utils.torch_summarize(model, show_weights=False)
logger.info(msg)
output_path = os.path.join(output_path, "test")
utils.mkdirs(output_path)
#
###########################################################################################
################################### Testing  ##############################################
###########################################################################################
#
test(test_loader=test_dataloader,
            model = model,
            device=model.device,
            res_path = output_path)