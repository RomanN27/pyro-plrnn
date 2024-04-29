import torch.distributions

def __init__(
    self, base_distribution, reinterpreted_batch_ndims, validate_args=None
):
    batch_shape = base_distribution.batch_shape
    if reinterpreted_batch_ndims > (b_len := len(batch_shape)):
        raise ValueError(
            "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
            f"actual {reinterpreted_batch_ndims} vs {len(batch_shape)}"
        )

    shape = batch_shape + base_distribution.event_shape
    new_event_shape_start = b_len - reinterpreted_batch_ndims
    batch_shape = shape[: new_event_shape_start]
    event_shape = shape[new_event_shape_start:]
    self.base_dist = base_distribution
    self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
    torch.distributions.Distribution.__init__(self,batch_shape, event_shape, validate_args=validate_args)


setattr(torch.distributions.Independent, "__init__", __init__)