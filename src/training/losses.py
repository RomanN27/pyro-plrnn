import torch
from pyro.infer import Trace_ELBO
from pyro.poutine import trace, Trace, replay, block
from src.training.messengers.handlers import force, subspace_replay
from src.utils.variable_group_enum import V

class TeacherForcingTraceELBO(Trace_ELBO):

    def __init__(self, forcing_interval: int, alpha: float, subspace_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forcing_interval = forcing_interval
        self.alpha = alpha
        self.subspace_dim = subspace_dim

    def _get_trace(self, model, guide, args, kwargs):
        """
        intercept variational_distribution trace for later forcing
        """
        #TODO understand where this get_trace method is coming from
        self.guide_trace = trace(guide, graph_type="flat").get_trace(
            *args, **kwargs
        )

        model_trace = trace(
            subspace_replay(model, trace=self.guide_trace, subspace_dim=self.subspace_dim,group_name=V.LATENT), graph_type="flat"
        ).get_trace(*args, **kwargs)

        model_trace.compute_log_prob()
        self.guide_trace.compute_score_parts()

        return model_trace, self.guide_trace

    def get_forced_trace(self, model, *args, **kwargs):
        forced_model_trace = trace(
            force(model, latent_group_name=V.LATENT, trace=self.guide_trace, forcing_interval=self.forcing_interval, subspace_dim=self.subspace_dim, alpha=self.alpha)
        ).get_trace(*args, **kwargs)

        return forced_model_trace

    def dsr_loss(self, model, *args, **kwargs) -> torch.Tensor:

        forced_model_trace = self.get_forced_trace(model, *args, **kwargs)
        dsr_loss = - forced_model_trace.log_prob_sum(site_filter=lambda s, t: "x_" in s)
        return dsr_loss

    def vanilla_loss(self,model,guide,*args,**kwargs):
        model_trace, guide_trace = self._get_trace(model,guide,args,kwargs)



    def differentiable_loss(self, model, guide, *args, **kwargs):
        vanilla_elbo = super().differentiable_loss(model, guide, *args, **kwargs)
        dsr_loss = self.dsr_loss(model, *args, **kwargs)
        mtf_loss = vanilla_elbo +  dsr_loss
        return mtf_loss,vanilla_elbo,dsr_loss
