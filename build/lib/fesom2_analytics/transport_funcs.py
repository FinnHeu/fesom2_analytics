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


def get_timeseries(ds, parameter='transport_across'):
    ''''''
    return ds[parameter].sum(dim=['elem', 'nz1'])


def heat_transport(ds, t_thresh=0, rho=1035, cp=4190, t_range=(-10, 30), s_range=(0, 40), scale=1e-12, climatology=False):
    ''''''

    # compute in, out and net transport and set all other transports to 0
    transp_in = ds.transport_across.where((ds.temp > t_thresh) & (ds.velocity_across > 0) & (
        ds.temp >= t_range[0]) & (ds.temp < t_range[-1]) & (ds.salt >= s_range[0]) & (ds.salt < s_range[-1]), 0)
    transp_out = ds.transport_across.where((ds.temp > t_thresh) & (ds.velocity_across < 0) & (
        ds.temp >= t_range[0]) & (ds.temp < t_range[-1]) & (ds.salt >= s_range[0]) & (ds.salt < s_range[-1]), 0)
    #transp_net = ds.transport_across.where((ds.temp > t_thresh) & (ds.temp >= t_range[0]) & (ds.temp < t_range[-1]) & (ds.salt >= s_range[0]) & (ds.salt < s_range[-1]), 0)

    # extract the temperature field
    temp = ds.temp

    # make temperature relative to t_thresh
    if t_thresh > 0:
        temp = temp - t_thresh
    elif t_thresh < 0:
        temp = temp + abs(t_thresh)
    elif t_thresh == 0:
        pass

    # compute the heat transport relative to t_thresh J = rho * cp * vol * dT
    # since transports that do not fullfill the criteria are set to 0 they do not contribute to the sum

    heat_transport_in = rho * cp * transp_in * temp
    heat_transport_out = rho * cp * transp_out * temp
    #heat_transport_net = rho * cp * transp_net * temp

    # compute the sum across the section
    heat_transport_in = heat_transport_in.sum(dim=['elem', 'nz1']) * scale
    heat_transport_out = heat_transport_out.sum(dim=['elem', 'nz1']) * scale
    #heat_transport_net = heat_transport_net.sum(dim=['elem','nz1']) * scale

    if climatology:
        heat_transport_in = heat_transport_in.groupby('time.month').mean()
        heat_transport_out = heat_transport_out.groupby('time.month').mean()
        #heat_transport_net = heat_transport_net.groupby('time.month').mean()

    return heat_transport_in, heat_transport_out, heat_transport_in + heat_transport_out
