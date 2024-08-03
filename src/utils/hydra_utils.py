from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path
from  typing import Optional
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


def get_module_from_cfg_path(path):
    cfg = load_cfg(path)
    return instantiate(cfg)


def load_cfg(path):
    with open(path, "r") as f:
        cfg = OmegaConf.load(f)
    return cfg


def get_module_from_relative_cfg_path(relative_path):
    path = get_cfg_path_from_relative_path(relative_path)

    return get_module_from_cfg_path(path)


def get_cfg_path_from_relative_path(relative_path):
    config_path = get_conf_path()
    path = config_path / relative_path
    return path


def get_conf_path() -> Path:
    hydra_utils_path = Path(__file__).absolute()
    config_path = hydra_utils_path.parent.parent.parent / "conf"
    return config_path


def flatten_config(relative_cfg_path, relative_target_directory ="flattened_configs", name: Optional = None):
    conf_path = get_conf_path()
    target_directory = conf_path / relative_target_directory
    target_directory.mkdir(parents=True,exist_ok=True)
    source_cfg_path = conf_path / relative_cfg_path

    name = name if name is not None else "flattened_" + source_cfg_path.name

    source_cfg = load_cfg(source_cfg_path)


    with open(target_directory / name,"w") as f:
        OmegaConf.save(source_cfg,f)

def make_yaml_file_from_clipboard():
    #this function reads the clipboard. It expects a class definition as a string in the clipboard
    # It creates yaml string out of it and saves it to the clipboard

    #read the clipboard
    import pyperclip
    text = pyperclip.paste()

    #evaluate the string and save in locals
    exec(text)

    #get the class module path
    module_path = locals()["__module__"]

    #get the class name
    class_name = locals()["__name__"]

    # make it lower case and separate with underscores where there is a capital letter

    def camel_to_snake(name):
        return "".join([f"_{char.lower()}" if char.isupper() else char for char in name])

    class_name = camel_to_snake(class_name)

    # get the kwargs in the constructor

    import inspect
    signature = inspect.signature(locals()[class_name].__init__)

    # build the hydra yaml string

    yaml_string = f"_target_: {module_path}.{class_name}\n"

    for key in signature.parameters.keys():
        if key == "self":
            continue
        yaml_string += f"  {key}: \n"

    pyperclip.copy(yaml_string)


















