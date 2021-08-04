'''Function selection to apply onto the transport dataset'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def TS_mask(ds, S_min=0, S_max=40, T_min=-4, T_max=20):
    ''''''
    if not ('temp' in list(ds.keys())) and ('salt' in list(ds.keys())):
        raise ValueError(
            'temperature and salinity are not part of the dataset, re-run cross_section_transport_v2.py with add_TS=True')

    ds['velocity_across'] = ds.velocity_across.where((ds.temp >= T_min) & (
        ds.temp <= T_max) & (ds.salt >= S_min) & (ds.salt <= S_max), 0)
    ds['transport_across'] = ds.transport_across.where((ds.temp >= T_min) & (
        ds.temp <= T_max) & (ds.salt >= S_min) & (ds.salt <= S_max), 0)

    return ds


def transport_timeseries(ds):
    ''''''
    return ds.transport_across.sum(dim=['elem', 'nz1'])
