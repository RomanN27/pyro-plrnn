from pathlib import Path
import json

import torch
from omegaconf import OmegaConf
from trainer import get_trainer_from_config
import matplotlib.pyplot as plt
from evaluation.pse import power_spectrum_error

run_id = "abb8411073a4487a888837ebb9a662fc"
path =  Path("mlruns") / "0" / run_id / "params" / "config"
time_series_model_state_dict_path  = f"mlartifacts/0/{run_id}/artifacts/time_series_model.pt"
variational_model_state_dict_path  = f"mlartifacts/0/{run_id}/artifacts/variational_model.pt"
config = OmegaConf.load(path)

trainer = get_trainer_from_config(config)
time_series_model_state_dict = torch.load(time_series_model_state_dict_path)
variational_distribution_state_dict = torch.load(variational_model_state_dict_path)
trainer.time_series_model.load_state_dict(time_series_model_state_dict)
#trainer.variational_distribution.load_state_dict(variational_distribution_state_dict)

batch = next(iter(trainer.data_loader))
time_series = trainer.time_series_model.sample_observed_time_series(batch)

untrained_trainer = get_trainer_from_config(config)
untrained_time_series =untrained_trainer.time_series_model.sample_observed_time_series(batch)


roi  = 17
rois = [0,1,2,3]
fig, axs = plt.subplots(2, 2)
for roi,ax in  zip(rois,axs.reshape(-1)):

    ax.set_title(f"ROI: {roi}")


    ax.plot(batch[0][:,roi].detach().numpy(),label="actual_data")
    ax.plot(time_series[0][:,roi].detach().numpy(), label = "trained_model")
    ax.plot(untrained_time_series[0][:,roi].detach().numpy(),label="untrained_model",alpha = 0.5)

    ax.legend()
plt.show()


pse = power_spectrum_error(time_series.detach().numpy(),batch[0].unsqueeze(0))
print(pse)