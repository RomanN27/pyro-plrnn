import ast
import os
import subprocess


import mlflow
from omegaconf import DictConfig
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import torch
import urllib
if TYPE_CHECKING:
    pass

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:8080"))
def convert_file_uri_to_path(file_uri: str) -> Path:
    return Path(unquote(urlparse(file_uri).path))


def get_checkpoint_from_run_id(run_id: str, check_point_name: Optional[str]= None):
    ckpt_path = get_ckpt_path_from_run_id(run_id, check_point_name)

    state_dict = torch.load(ckpt_path)

    return state_dict


def get_ckpt_path_from_run_id(run_id: str, check_point_name: Optional[str]= None):
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    artifact_path = convert_file_uri_to_path(artifact_uri) / "model" / "checkpoints"
    last_checkpoint = list(artifact_path.iterdir())[0].name
    check_point_name = check_point_name if check_point_name is not None else last_checkpoint
    ckpt_path = artifact_path / check_point_name / check_point_name
    ckpt_path = ckpt_path.with_suffix(".ckpt")
    return ckpt_path


def convert_param_value(value):
    try:
        # Attempt to evaluate the string as a literal
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If evaluation fails, keep the value as a string
        pass
    return value

def get_hyperparameters_by_run_id(run_id):
    # Set the tracking URI if needed
    # mlflow.set_tracking_uri("http://your-tracking-uri")

    # Get the run details
    run = mlflow.get_run(run_id)

    # Get the hyperparameters (logged as parameters in MLflow)
    hyperparameters = run.data.params

    # Convert parameter values to their appropriate types
    converted_hyperparameters = {k: convert_param_value(v) for k, v in hyperparameters.items()}

    return converted_hyperparameters

def unflatten_logged_params(hyperparameters: dict):
    upper_level_keys = [x.split("/")[0] for x in hyperparameters.keys()]
    for upper_level_key in upper_level_keys:
        relevent_items = {k:v for k,v in hyperparameters.items() if  k.split("/")[0] == upper_level_key}
        if len(relevent_items)==1:
            continue
        unflattened_relevant_item =  {"/".join(x.split("/")[1:]): y for x,y in relevent_items.items()}
        for r in relevent_items:
            hyperparameters.pop(r)

        hyperparameters[upper_level_key] = unflatten_logged_params(unflattened_relevant_item)
    return hyperparameters

def get_config_from_run_id(run_id: str)->DictConfig:
    hyperparameters = get_hyperparameters_by_run_id(run_id)

    cfg = unflatten_logged_params(hyperparameters)
    cfg = DictConfig(cfg)
    return cfg

def start_local_mlflow_ui():
    tracking_uri = mlflow.get_tracking_uri()
    port = urllib.parse.urlparse(tracking_uri).port
    command = f"mlflow ui --port {port}"
    subprocess.Popen(command.split(" "))


def instantiate_object_by_run_id(run_id: str ):
    pass






if __name__ == '__main__':
    # Example usage

    from hydra.utils import instantiate
    run_id = "3d4e9c4664bd40fc9a6ca4df70cb52ae"
    hyperparameters = get_hyperparameters_by_run_id(run_id)

    cfg = unflatten_logged_params(hyperparameters)
    model = instantiate(cfg)
    state_dict = get_checkpoint_from_run_id(run_id)




