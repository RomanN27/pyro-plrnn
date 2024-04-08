from pathlib import Path
import json

import torch
from omegaconf import OmegaConf
from trainer import get_trainer_from_config
import matplotlib.pyplot as plt
from evaluation.pse import power_spectrum_error

run_id = "83756c09a1ab42ffb4b144330d5a3684"
path =Path(fr"C:\Users\roman.nefedov\PycharmProjects\PLRNN_Family_Variational_Inference\mlruns\0\{run_id}\params\config")
model_state_dict_path  =f"mlartifacts/0/{run_id}/artifacts/model.pt"
config = OmegaConf.load(path)

trainer = get_trainer_from_config(config)
state_dict = torch.load(model_state_dict_path)
trainer.load_state_dict(state_dict)

time_series = trainer.sample_observed_time_series()

untrained_trainer = get_trainer_from_config(config)
untrained_time_series =untrained_trainer.sample_observed_time_series()
batch = next(iter(trainer.data_loader))

roi  = 17
rois = [1,5,15,18]
fig, axs = plt.subplots(2, 2)
for roi,ax in  zip(rois,axs.reshape(-1)):

    ax.set_title(f"ROI: {roi}")
    orig_time_series = batch[0]


    ax.plot(orig_time_series[0][:,roi].detach().numpy(),label="actual_data")
    ax.plot(time_series[0][:,roi].detach().numpy(), label = "trained_model")
    ax.plot(untrained_time_series[0][:,roi].detach().numpy(),label="untrained_model",alpha = 0.5)

    ax.legend()
plt.show()


pse = power_spectrum_error(time_series.detach().numpy(),orig_time_series)
print(pse)