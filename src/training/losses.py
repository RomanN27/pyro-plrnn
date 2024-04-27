from pyro.infer import Trace_ELBO
from pyro.poutine import trace, Trace
from src.training.forcing_interval_replay import force
class TeacherForcingTraceELBO(Trace_ELBO):

    def __init__(self,forcing_interval,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.forcing_interval = forcing_interval

    def _get_trace(self, model, guide, args, kwargs):
        """
        intercept guide trace for later forcing
        """
        model_trace, self.guide_trace = super()._get_trace(model, guide, args, kwargs)

        return model_trace, self.guide_trace


    def get_forced_trace(self,model, *args,**kwargs):

        forced_model_trace = trace(
            force(model, trace=self.guide_trace, forcing_interval=self.forcing_interval)
        ).get_trace(*args, **kwargs)

        return forced_model_trace

    def dsr_loss(self, model, *args, **kwargs) -> Trace:
        forced_model_trace = self.get_forced_trace( model, *args, **kwargs)
        dsr_loss = - forced_model_trace.log_prob_sum(site_filter=lambda s,t:  "x" in s)
        return dsr_loss
    def differentiable_loss(self, model, guide, *args, **kwargs):
        vanilla_elbo = super().differentiable_loss( model, guide, *args, **kwargs)
        dsr_loss = self.dsr_loss( model, *args, **kwargs)
        mtf_loss = vanilla_elbo + dsr_loss
        return mtf_loss

