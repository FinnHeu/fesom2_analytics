"""Module to plot transport associated analytics from ocean model pyfesom2.0"""

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean.cm as cmo
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np



def depth_section(ds_transport, date='mean', variable='velocity_across', savepath=None, vmin=None, vmax=None, Sv=True, y_type='dist'):
    """
    depth_section.py

    Plots depth section of various parameters for specific time steps or all-time mean

    Inputs:
    ---------------------------------------
    ds_transport: xr.DataArray, output from across_section_transport.py or water_property_mask.py
    date: str, Either date in format: '2000-01-01' for specific time step or 'mean' for all time mean
    variable: str, ['velocity_across', 'transport_across'] from across_section_transport.py or ['mask', 'temp_mask', 'salt_mask', 'temp', 'salt'] from water_property_mask.py
    """

    # prepare depth, and lon or dist arrays for plotting with pcolor
    if y_type == 'dist':
        dist_left_arr = ds_transport.central_dist - ds_transport.central_dist[0] + 1/2 * ds_transport.width
        dist_left = [0]
        for i in range(len(dist_left_arr.values)):
            dist_left.append(dist_left_arr.values[i] / 1e3)

    elif y_type == 'lon':
        dist_left = [i for i in ds_transport.intersection_coords[:,1].values]
        dist_left.append(ds_transport.intersection_coords[:,1].values[-1])

    depth_down=[0]
    for i in range(len(ds_transport.depth.values)):
        depth_down.append(ds_transport.depth.values[i])

    # prepare data by time keyword
    if time == 'mean':

        data_to_plot = np.where(
            ds_transport[variable].mean(dim="time").transpose() == 0,
            np.nan,
            ds_transport[variable].mean(dim="time").transpose(),
        )

    else:

        data_to_plot = np.where(
            ds_transport[variable].sel(time=date).transpose() == 0,
            np.nan,
            ds_transport[variable].sel(time=date).transpose(),
        )


        # apply Sv to transport
        if (variable == 'transp_across') & (Sv):
            data_to_plot = data_to_plot * 1e-6

    # Plot

    # find color limits
    if (vmin != None) and (vmax != None):
        clim = (vmin, vmax)
    else:
        clim = (-np.nanmax(np.abs(data_to_plot)), np.nanmax(np.abs(data_to_plot)))

#     # find depth limit
    depth_limit = ds_transport.depth.isel(depth=np.where(np.isnan(data_to_plot) == False)[0].max() + 1).values

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    cb = ax.pcolor(
    np.array(dist_left),
    np.array(depth_down),
    data_to_plot,
    cmap="RdBu_r",
    edgecolor="k",
    vmin=clim[0],
    vmax=clim[1],
    )

    ###### Axis
    ax.set_ylim((0, depth_limit))
    ax.invert_yaxis()

    if y_type == 'dist':
        ax.set_xlabel("distance [km]")
    elif y_type == 'lon':
        ax.set_ylabel("depth [m]")
        ax.invert_xaxis()

    ax.set_ylabel("depth [m]")

    plt.colorbar(cb, ax=ax, label="time mean cross section " + data )

    plt.show()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='w')

    return


def plot_background(ax, extent=[17, 23, 69, 75], crs=ccrs.Mollweide(central_longitude=20)):
    '''
    plot_background.py

    Plots map projection '''

    ax.set_extent(extent)
    ax.add_feature(cfeature.LAND, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=1,
                      color="gray",
                      alpha=0.5,
                      linestyle="--",
                      x_inline=False,
                      y_inline=False
                     )

    gl.xlocator = mticker.FixedLocator(np.arange(-100,100,1))
    gl.ylocator = mticker.FixedLocator(np.arange(60,85.5,.5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    gl.ylabels_right = False

    return ax
