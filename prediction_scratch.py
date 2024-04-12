import os
from pathlib import Path
import json

import torch
from omegaconf import OmegaConf
from trainer import AnnealingTrainer
import matplotlib.pyplot as plt
from evaluation.pse import power_spectrum_error
from pyro.poutine import trace
from pyro.infer.predictive import Predictive
from urllib.parse import unquote, urlparse
from pathlib import Path
import os
import pyro
from pyro.contrib.autoname import scope
run_id = "3874118a7da743a3a6b005e1b44f832d"

path = Path(os.environ.get("MLFLOW_TRACKING_URI",__file__)).parent
trainer = AnnealingTrainer.get_trainer_from_run_id_config(run_id)
model_uri = path / f"mlartifacts/0/{run_id}/artifacts/model.pt"

model_path = urlparse(str(model_uri)).path
trainer.load(model_path)

batch = next(iter(trainer.data_loader))

#batch
#traced_guide = trace(trainer.variational_distribution)
#traced_guide(batch)

pred = Predictive(trainer.time_series_model,guide= trainer.variational_distribution,num_samples=1000,parallel=True,return_sites=[f"x_{i}" for i in range(525)])
num_samples = 1000
vectorize = pyro.plate(
        "_num_predictive_samples", num_samples,dim=-2
    )
time_steps_to_predict = 100
t_0 = batch[0].size(0) + 1
time_range = range(t_0,t_0+  time_steps_to_predict )

#with pyro.plate("MC", num_samples)
with pyro.poutine.trace() as tracer:
    z_h = vectorize(trainer.time_series_model)(batch)
    with scope(prefix = "pred"):
        trainer.time_series_model.run_over_time_range(z_h,time_range,1)
#a = tracer.trace.stochastic_nodes[-3:]

#pyro.sample("test",pyro.distributions.Normal(torch.zeros(5),torch.ones(1)), obs= torch.empty(5))

#pred = Predictive(trainer.time_series_model, guide=trainer.variational_distribution,num_samples=69,parallel=True)


def get_values_from_nodes(nodes):
    return [node["value"] for node in nodes]
def get_predicted_observed_values_from_trace(trace: pyro.poutine.trace_struct.Trace):

    inputed_observed_nodes = [trace.nodes[obs_node] for obs_node in trace.observation_nodes]
    inputed_observed_values = get_values_from_nodes(inputed_observed_nodes)
    inputed_observed_values = torch.stack(inputed_observed_values)

    #regex to get all stochastic nodes beginning with pred/x_
    pred_nodes = [trace.nodes[pred_node] for pred_node in trace.stochastic_nodes if pred_node.startswith("pred/x_")]
    pred_values = get_values_from_nodes(pred_nodes)
    pred_values = torch.stack(pred_values)

    n_mc_samples = pred_values.size(0)
    inputed_observed_values = inputed_observed_values.unsqueeze(0).expand(n_mc_samples,-1,-1,-1)

    all_values = torch.cat([inputed_observed_values,pred_values],dim=1)
    return all_values

get_predicted_observed_values_from_trace(tracer.trace)