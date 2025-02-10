import sys
sys.path.append("/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/models/pangu-pytorch")
from era5_data import utils, utils_data
from torch import nn
import torch
import copy
from era5_data import score
import os


def validate(val_loader, model, device, res_path, cfg, logger, writer, start_epoch):
    # Begin to validate
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)
    upper_weights, surface_weights = aux_constants['variable_weights']
    criterion = nn.L1Loss(reduction='none')
    epochs = cfg.PG.TRAIN.EPOCHS

    for i in range(start_epoch, epochs + 1):
        if i % cfg.PG.VAL.INTERVAL == 0:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0

                print("Number of iterations in validation:", len(val_loader))

                for id, val_data in enumerate(val_loader, 0):
                    input_val, input_surface_val, target_val, target_surface_val, periods_val = val_data
                    input_val_raw, input_surface_val_raw = input_val, input_surface_val
                    input_val, input_surface_val, target_val, target_surface_val = input_val.to(
                        device), input_surface_val.to(device), target_val.to(device), target_surface_val.to(device)

                    # Inference
                    output_val, output_surface_val = model(input_val, input_surface_val,
                                                            aux_constants['weather_statistics'],
                                                            aux_constants['constant_maps'], aux_constants['const_h'])
                    # Noralize the gt to make the loss compariable
                    target_val, target_surface_val = utils_data.normData(target_val, target_surface_val,
                                                                aux_constants['weather_statistics_last'])

                    val_loss_surface = criterion(output_surface_val, target_surface_val)
                    weighted_val_loss_surface = torch.mean(val_loss_surface * surface_weights)

                    val_loss_upper = criterion(output_val, target_val)
                    weighted_val_loss_upper = torch.mean(val_loss_upper * upper_weights)

                    loss = weighted_val_loss_upper + weighted_val_loss_surface * 0.25

                    val_loss += loss.item()

                    logger.info(f"Epoch {i}, Iteration {id + 1}/{len(val_loader)}: Loss = {loss.item():.6f}")

                val_loss /= len(val_loader)
                writer.add_scalars('Loss', {'val': val_loss}, i)
                logger.info("Validate at Epoch {} : {:.3f}".format(i, val_loss))
                # Visualize the training process
                png_path = os.path.join(res_path, "png_training")
                utils.mkdirs(png_path)
                # """
                # Normalize the data back to the original space for visualization
                output_val, output_surface_val = utils_data.normBackData(output_val, output_surface_val,
                                                                aux_constants['weather_statistics_last'])
                target_val, target_surface_val = utils_data.normBackData(target_val, target_surface_val,
                                                                aux_constants['weather_statistics_last'])

                utils.visuailze(output_val.detach().cpu().squeeze(),
                                target_val.detach().cpu().squeeze(),
                                input_val_raw.squeeze(),
                                var='u',
                                z=12,
                                step=i,
                                path=png_path,
                                cfg=cfg)
                
                utils.visuailze_surface(output_surface_val.detach().cpu().squeeze(),
                                        target_surface_val.detach().cpu().squeeze(),
                                        input_surface_val_raw.squeeze(),
                                        var='msl',
                                        step=i,
                                        path=png_path,
                                        cfg=cfg)

                                    

                model_save_path = os.path.join(res_path, 'models')
                utils.mkdirs(model_save_path)

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                    # Save the best model
                    torch.save(best_model, os.path.join(model_save_path, 'best_model.pth'))
                    logger.info(
                        f"current best model is saved at {i} epoch.")
                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1
                    if epochs_since_last_improvement >= 5:
                        logger.info(f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training.")
                        break