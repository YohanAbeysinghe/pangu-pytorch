import sys
sys.path.append("/home/yohan.abeysinghe/Pangu/pangu-pytorch")

import os
import logging
import torch
import argparse
import importlib
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from models.pangu_model import PanguModel


def load_model_for_inference(cfg, output_path, device):
    # Step 1: Reconstruct base model
    model = PanguModel(device=device, cfg=cfg).to(device)

    # Step 2: Edit conv_surface weights for input/output layers
    checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device)
    state_dict = checkpoint['model']

    if cfg.GLOBAL.MODEL == 'pm1':
        new_input_weight = torch.zeros((192, 128, 1))
        new_input_weight[:, :112, :] = state_dict['_input_layer.conv_surface.weight']
        torch.nn.init.xavier_uniform_(new_input_weight[:, 112:, :])
        state_dict['_input_layer.conv_surface.weight'] = new_input_weight

        new_output_weight = torch.zeros((80, 384, 1))
        new_output_weight[:64, :, :] = state_dict['_output_layer.conv_surface.weight']
        torch.nn.init.xavier_uniform_(new_output_weight[64:, :, :])
        state_dict['_output_layer.conv_surface.weight'] = new_output_weight

        new_output_bias = torch.zeros(80)
        new_output_bias[:64] = state_dict['_output_layer.conv_surface.bias']
        state_dict['_output_layer.conv_surface.bias'] = new_output_bias

    model.load_state_dict(state_dict)

    # Step 3: Apply LoRA
    target_modules = [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]

    lora_config = LoraConfig(
        r=cfg.PG.TRAIN.Low_Rank,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Step 4: Load LoRA and edited-layer weights from finetuned checkpoint
    best_model_path = os.path.join(output_path, "models/model_weights_1.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    return model



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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_path = os.path.join(cfg.PG_OUT_PATH, args.output)
#
###########################################################################################
###########################################################################################
###########################################################################################



# Load model for inference
best_model = load_model_for_inference(cfg, output_path, device)
