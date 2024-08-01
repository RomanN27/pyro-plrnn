import io

import plotly.io as pio
from plotly import graph_objects as go


def get_plotly_html_string(fig: go.Figure) -> str:
    buffer = io.StringIO()
    pio.write_html(fig=fig, file=buffer)
    return buffer.getvalue()
