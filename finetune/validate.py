import sys
sys.path.append("/home/yohan.abeysinghe/Pangu/pangu-pytorch")

import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
from era5_data import utils_data
from models.pangu_model import PanguModel
import importlib
import os
from era5_data import utils_data, utils

# ----------------------------
# Load config
# ----------------------------
config_module = importlib.import_module("configs.config4")  # Use your desired config
cfg = config_module.cfg

# ----------------------------
# Set device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model
# ----------------------------
model = PanguModel(device=device, cfg=cfg).to(device)

# Path to best_model.pth
output_dir = os.path.join(cfg.PG_OUT_PATH, "pm1_1")  # Replace "test1" if needed
checkpoint_path = os.path.join(output_dir, "models/train_1.pth")

checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model'])
model.eval()

PATH = cfg.PG_INPUT_PATH

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

# Load constants and teleconnection indices
criterion = nn.L1Loss(reduction='none')
res_path = output_dir
aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
upper_weights, surface_weights = aux_constants['variable_weights']

i = 0
val_loss = 0.0

for id, val_data in enumerate(val_dataloader, 0):

    # Skip this batch if any data component is empty
    if (val_data[0].sum() == 0 or 
        val_data[1].sum() == 0 or 
        val_data[2].sum() == 0 or 
        val_data[3].sum() == 0
        ):
        # print(f"Skipping batch {id} due to missing or empty data.")
        continue
    
    input_val, input_surface_val, target_val, target_surface_val, periods_val = val_data
    input_val_raw, input_surface_val_raw = input_val, input_surface_val
    input_val, input_surface_val, target_val, target_surface_val = input_val.to(device), input_surface_val.to(device), target_val.to(device), target_surface_val.to(device)

    # Inference
    output_val, output_surface_val = model(input_val,input_surface_val,
                                            aux_constants['weather_statistics'],
                                            aux_constants['constant_maps'],
                                            aux_constants['const_h']
                                            )

    # Noralize the gt to make the loss compariable
    target_val, target_surface_val = utils_data.normData(target_val,
                                                            target_surface_val,
                                                            aux_constants['weather_statistics_last'])

    val_loss_surface = criterion(output_surface_val, target_surface_val)
    weighted_val_loss_surface = torch.mean(val_loss_surface * surface_weights)

    val_loss_upper = criterion(output_val, target_val)
    weighted_val_loss_upper = torch.mean(val_loss_upper * upper_weights)

    loss = weighted_val_loss_upper + weighted_val_loss_surface * 0.25

    val_loss += loss.item()

    # if rank == 0:
    #     logger.info(f"Epoch {i}, Iteration {id + 1}/{len(val_loader)}: Loss = {loss.item():.6f}")

# if rank == 0:
#     val_loss /= len(val_loader)
#     writer.add_scalars(
#         'Loss',
#         {'train': epoch_loss,
#             'val': val_loss},
#             i
#             )
    
    # logger.info("Validate at Epoch {} : {:.3f}".format(i, val_loss))
    # Visualize the training process
    png_path = os.path.join(res_path, "png_training")
    utils.mkdirs(png_path)
    # Normalize the data back to the original space for visualization
    output_val, output_surface_val = utils_data.normBackData(output_val, output_surface_val,
                                                                aux_constants['weather_statistics_last'])
    target_val, target_surface_val = utils_data.normBackData(target_val, target_surface_val,
                                                                aux_constants['weather_statistics_last'])

    utils.visualize(
        output_val.detach().cpu().squeeze(),
        target_val.detach().cpu().squeeze(),
        input_val_raw.squeeze(),
        var='u',
        z=12,
        step=i,
        path=png_path,
        cfg=cfg
        )
    
    utils.visualize_surface(
        output_surface_val.detach().cpu().squeeze(),
        target_surface_val.detach().cpu().squeeze(),
        input_surface_val_raw.squeeze(),
        var='u10',
        step=i,
        path=png_path,
        cfg=cfg
        )

    utils.visualize_surface(
        output_surface_val.detach().cpu().squeeze(),
        target_surface_val.detach().cpu().squeeze(),
        input_surface_val_raw.squeeze(),
        var='pm1',
        step=i,
        path=png_path,
        cfg=cfg
        )
