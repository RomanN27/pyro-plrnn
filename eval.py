from archive.trainer import AnnealingTrainer
import matplotlib.pyplot as plt
from src.metrics.pse import power_spectrum_error

run_id = "97836ebfe5c04f2a90f07355ecd40447"

trainer = AnnealingTrainer.get_trainer_from_run_id_config(run_id)
trainer.load(f"mlartifacts/0/{run_id}/artifacts/model.pt")

batch = next(iter(trainer.data_loader))
time_series = trainer.hidden_markov_model.generate_time_series_from_batch(batch)

untrained_trainer = AnnealingTrainer.get_trainer_from_run_id_config(run_id)
untrained_time_series =untrained_trainer.hidden_markov_model.generate_time_series_from_batch(batch)


roi  = 17
rois = [0,1,2,3]
fig, axs = plt.subplots(2, 2)
for roi,ax in  zip(rois,axs.reshape(-1)):

    ax.set_title(f"ROI: {roi}")


    ax.plot(batch[0][:,roi].detach().numpy(),label="actual_data",alpha = 0.5)
    ax.plot(time_series[0][:,roi].detach().numpy(), label = "trained_model")
    ax.plot(untrained_time_series[0][:,roi].detach().numpy(),label="untrained_model",alpha = 0.5)

    ax.legend()
plt.show()


pse = power_spectrum_error(time_series.detach().numpy(),batch[0].unsqueeze(0))
print(pse)