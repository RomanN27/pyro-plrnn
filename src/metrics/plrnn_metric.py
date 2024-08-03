from typing import Any

import torch
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

import matplotlib as mpl
from src.metrics.metric_base import Metric, MetricLogType
from src.models.hidden_markov_model import HiddenMarkovModel, ObservationModelType
from src.models.transition_models.plrnns.plrnn_base import PLRNN
import plotly.express as px

from src.utils.lightning_utils import update_mean


class PLRNNMetric(Metric):
    log_types = [MetricLogType.plotly_figure,MetricLogType.dict]
    def __init__(self):
        super().__init__()
        self.add_state("diag", default=torch.tensor(0.), dist_reduce_fx=torch.mean)
        self.add_state("off_diag", default=torch.tensor(0.), dist_reduce_fx=torch.mean)
        self.add_state("bias", default=torch.tensor(0.), dist_reduce_fx=torch.mean)

        self.add_state("n", default=torch.tensor(0,dtype=torch.int32))

    def update(self, hmm: HiddenMarkovModel[PLRNN, ObservationModelType]) -> None:
        diag = hmm.transition_sampler.model.diag.A_diag.detach()
        off_diag =hmm.transition_sampler.model.off_diag.W.detach()
        bias = hmm.transition_sampler.model.bias.detach()

        self.diag = update_mean(self.diag, diag, self.n)
        self.off_diag = update_mean(self.off_diag, off_diag, self.n)
        self.bias = update_mean(self.bias, bias, self.n)
        self.n += 1


    def compute(self) -> dict:
        dict_= {
            "diag": self.diag.tolist(),
            "off_diag": self.off_diag.tolist(),
            "bias": self.bias.tolist()
        }
        return dict_

    def plot(self) -> Any:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "heatmap"}, {"type": "table"}]])

        heat_map = self.build_heatmap()

        table = self.build_table()
        fig.add_trace(*heat_map.data,row=1,col=1)
        fig.add_trace(*table.data,row=1,col=2)
        fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")

        return fig

    def build_heatmap(self):
        W = self.off_diag.detach().clone()
        W[torch.eye(len(W)).bool()] = 0
        fig = px.imshow(W.tolist())
        return fig

    def build_table(self):
        fill_colors = [self.get_colors(self.diag.tolist()), self.get_colors(self.bias.tolist())]

        return go.Figure([go.Table(
            header=dict(
                values=["Diag", "Bias"],
                line_color='white', fill_color='white',
                align='center', font=dict(color='black', size=12)
            ),
            cells=dict(
                values=[self.diag.tolist(), self.bias.tolist()], fill_color=fill_colors,
                align='center', font=dict(color='black', size=11)
            ))
        ])

    @staticmethod
    def get_colors(values:list[float]):
        bound = np.array(values).__abs__().max()
        cmap = mpl.colormaps["RdYlGn"]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)

        colors = [cmap(norm(value)) for value in values]
        colors = ["rgb" + str(tuple(int(255 * x) for x in color[:3])) for color in colors]


        return colors




if __name__ == '__main__':
    from src.utils.hydra_utils import get_module_from_relative_cfg_path
    import plotly.io as pio

    pio.renderers.default = "browser"

    rel = "flattened_configs/flattened_config_3.yaml"
    module = get_module_from_relative_cfg_path(rel)
    trans = module.lightning_module.hidden_markov_model.transition_sampler.model
    metric = PLRNNMetric()
    metric.update(hmm = module.lightning_module.hidden_markov_model)
    fig =metric.plot()
    fig.show()



