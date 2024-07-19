from pyro.infer import Trace_ELBO
from pyro.poutine import trace, Trace, replay
from src.training.messengers.handlers import force, subspace_replay


class TeacherForcingTraceELBO(Trace_ELBO):

    def __init__(self, forcing_interval: int, alpha: float, signal_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forcing_interval = forcing_interval
        self.alpha = alpha
        self.signal_dim = signal_dim

    def _get_trace(self, model, guide, args, kwargs):
        """
        intercept guide trace for later forcing
        """
        #TODO understand where this get_trace method is coming from
        self.guide_trace = trace(guide, graph_type="flat").get_trace(
            *args, **kwargs
        )

        model_trace = trace(
            subspace_replay(model, trace=self.guide_trace,sub_space_dim=self.signal_dim), graph_type="flat"
        ).get_trace(*args, **kwargs)



        return model_trace, self.guide_trace

    def get_forced_trace(self, model, *args, **kwargs):
        forced_model_trace = trace(
            force(model, trace=self.guide_trace, forcing_interval=self.forcing_interval,sub_space_dim=self.signal_dim)
        ).get_trace(*args, **kwargs)

        return forced_model_trace

    def dsr_loss(self, model, *args, **kwargs) -> Trace:
        forced_model_trace = self.get_forced_trace(model, *args, **kwargs)
        dsr_loss = - forced_model_trace.log_prob_sum(site_filter=lambda s, t: "x" in s)
        return dsr_loss

    def differentiable_loss(self, model, guide, *args, **kwargs):
        vanilla_elbo = super().differentiable_loss(model, guide, *args, **kwargs)
        dsr_loss = self.dsr_loss(model, *args, **kwargs)
        mtf_loss = vanilla_elbo + dsr_loss
        return mtf_loss
