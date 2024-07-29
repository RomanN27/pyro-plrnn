from omegaconf import DictConfig
from hydra.utils import instantiate

LIST_KEYS = ["callbacks"]

def listify_config(cfg: DictConfig)->DictConfig:
    """
    Convert specific dictionary entries to lists for instantiation.

    Hydra has limitations in handling configurations where lists need to be passed as arguments
    for object instantiation (e.g., lists of callbacks or metrics for a Lightning trainer).
    (see https://github.com/facebookresearch/hydra/issues/1939)
    In YAML, these lists must be represented as dictionaries for Hydra to parse them correctly.
    However, this prevents an automatic recursive instantiation process by hydra.utils.instantiate
    since the constructor may explicitly require a list. Hence This function converts such disguised dictionaries back
    to lists.

    Args:
        cfg (DictConfig): The input Hydra configuration object.

    Returns:
        DictConfig: The transformed Hydra configuration object, with specified dictionaries
        converted to lists for instantiation.

    Notes:
        - The function specifically looks for keys defined in the LIST_KEYS constant.
        - If a key matches and its value is a dictionary, it converts the dictionary to a list
          format suitable for instantiation.
        - The function processes nested configurations recursively.
    """
    for key in cfg.keys():
        if key in LIST_KEYS:
            list_disguised_as_dict: DictConfig = cfg[key]
            list_ = list(list_disguised_as_dict.values())
            new_dict = {
                "_target_": "builtins.list",
                "_args_": [list_]
            }
            new_dict = DictConfig(new_dict)
            cfg[key] = new_dict
            return cfg
        elif isinstance(cfg[key], DictConfig):
            cfg[key] = listify_config(cfg[key])

    return cfg
