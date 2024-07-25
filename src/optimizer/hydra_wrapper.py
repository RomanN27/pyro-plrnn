from pyro.optim import ClippedAdam as _ClippedAdam
from omegaconf import DictConfig
def ClippedAdam(optim_args: DictConfig):
    # ClippedAdam needs to parameters to be passed as dict and not as DictcConfig ( which is the format hydra uses)
    optim_args = dict(optim_args)
    return _ClippedAdam(optim_args)