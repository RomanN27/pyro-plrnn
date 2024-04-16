from pytorch_forecasting.data import TimeSeriesDataSet
from dataloader import  IndicateDataSet
import pandas as pd
a = IndicateDataSet.from_path()
tensors = a.tensors

a = pd.DataFrame(tensors[0])
dfs = []
for j,tensor in  enumerate(tensors):
    df_ = pd.DataFrame(tensor)
    df_.columns = (roi_names := [f"ROI_{i}" for i in df_.columns])
    df_["subject"] = j
    df_["time_idx"] = list(range(len(df_)))
    dfs.append(df_)

df = pd.concat((dfs))

df.reset_index(inplace=True)

ts_data = TimeSeriesDataSet(df, group_ids=["subject"], time_idx="time_idx", target = roi_names,time_varying_unknown_reals = roi_names)