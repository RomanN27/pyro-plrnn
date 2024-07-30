import torch
from pyro.poutine import Trace
from pyro.poutine.messenger import Message
import re
from typing import Literal, Callable
from functools import partial
from src.utils.variable_group_enum import V
MsgKey = Literal[
    "type",
    "name",
    "fn",
    "is_observed",
    "args",
    "kwargs",
    "value",
    "scale",
    "mask",
    "cond_indep_stack",
    "done",
    "stop",
    "continuation",
    "infer",
    "obs",
    "log_prob",
    "log_prob_sum",
    "unscaled_log_prob",
    "score_parts",
    "packed",
    "_intervener_id"
]
BooleanFunction = Callable[[Message], bool]


def get_time_stamp(site_name: str|Message) -> int:
    if not isinstance(site_name,str):
        site_name = site_name["name"]
    time_stamp = site_name.split("_")[1]
    return int(time_stamp)


def is_group_msg(msg: Message, group_name: str) -> bool:
    name = msg["name"]
    boo = bool(re.match(f"{group_name}_\d*$", name))
    return boo

def is_group_msg_getter(group_name:str)->BooleanFunction:
    return partial(is_group_msg,group_name= group_name)

def get_group_msgs_from_trace(trace: Trace, group_name: str) -> list[Message]:
    return [msg for msg in list(trace.nodes.values()) if is_group_msg(msg, group_name)]


def get_property_from_msgs(msgs: list[Message], property_name: MsgKey) -> list:
    return [msg[property_name] for msg in msgs]


def get_property_from_trace(trace: Trace, property_name: MsgKey, boo_f: BooleanFunction) -> list:
    msgs = get_msgs_from_trace(trace, boo_f)
    return get_property_from_msgs(msgs, property_name=property_name)


def get_msgs_from_trace(trace: Trace, boo_f: BooleanFunction) -> list[Message]:
    msgs = [msg for msg in list(trace.nodes.values()) if boo_f(msg)]
    return msgs


get_values_from_msgs = partial(get_property_from_msgs, property_name="value")
get_log_prob_from_msgs = partial(get_property_from_msgs, property_name="log_prob_sum")


def get_values_from_trace(trace: Trace, boo_f: BooleanFunction = lambda x: True):
    return get_values_from_msgs(get_msgs_from_trace(trace, boo_f))


def get_log_prob_from_trace(trace: Trace, boo_f: BooleanFunction = lambda x: True):
    return get_log_prob_from_msgs(get_msgs_from_trace(trace, boo_f))


def get_observed_values_from_trace(trace:Trace):
    observed_msgs =get_group_msgs_from_trace(trace,V.OBSERVED)
    return get_values_from_msgs(observed_msgs)

def get_hidden_values_from_trace(trace:Trace):
    latent_msgs= get_group_msgs_from_trace(trace,V.OBSERVED)
    return get_values_from_msgs(latent_msgs)