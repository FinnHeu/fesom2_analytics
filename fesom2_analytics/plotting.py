"""Module to plot transport associated analytics from ocean model pyfesom2.0"""

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean.cm as cmo
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy as np



# Eventually plot the overview figure
def plot_overview(
    section_start,
    section_end,
    u_array,
    v_array,
    coords_array,
    lon,
    lat,
    extent,
    elem_no_nan,
    elem_array,
    save_figures,
):
    """
    plot_overview.py

    Plots global map with chosen section and regional map with section and regional total velocity magnitude (np.sqrt(u² + v²))

    Inputs:
    ----------------------------------------
    section_start (tuple, list[1,2])
    section_end (tuple, list[1,2])
    u_array (xr.datarray)
    v_array (xr.dataarray)
    lon (np.ndarray)
    lat (np.ndarray)
    extent (int, float)
    elem_no_nan (np.ndarray)
    save_figures (bool)

    Returns:
    ----------------------------------------


    """
    # Plot section

    fig, ax = plt.subplots(
        1, 1, figsize=(20, 20), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    ext = 1

    # ax = ax.flatten()
    # for axis in ax:
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_global()
    ax.set_extent(
        (
            section_start[0] - ext,
            section_end[0] + ext,
            section_start[1] + ext,
            section_end[1] - ext,
        )
    )
    # ax.set_extent((21, 22, 70, 71))
    cb = ax.tripcolor(
        lon,
        lat,
        elem_array,
        u_array.mean(dim=("time", "nz1")).to_array().values.squeeze(),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        edgecolors="w",
    )

    ax.plot(
        [section_start[0], section_end[0]],
        [section_start[1], section_end[1]],
        transform=ccrs.PlateCarree(),
        color="r",
    )

    ax.plot(
        coords_array[:, 2],
        coords_array[:, 3],
        "*",
        transform=ccrs.PlateCarree(),
        color="w",
        markersize=2,
    )

    plt.show()

    if save_figures:
        path = join(save_regional_output, "figures_transport")
        filename = "overview_transport" + section + ".png"
        plt.savefig(join(path, filename), dpi=300, bbox_inches="tight")
        print("\n----> Saving figure")
        print(join(path, filename))

    return





def time_mean_section(ds_transport, savepath=None, data='velocity', vmin=None, vmax=None, Sv=True, y_type='dist'):
    """"""

    # prepare depth, and dist arrays for plotting with pcolor
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

    if data == 'velocity':
        data_to_plot = np.where(
            ds_transport.velocity_across.mean(dim="time").transpose() == 0,
            np.nan,
            ds_transport.velocity_across.mean(dim="time").transpose(),
        )

    elif data == 'transport':
        data_to_plot = np.where(
                                ds_transport.transport_across.mean(dim="time").transpose() == 0,
                                np.nan,
                                ds_transport.transport_across.mean(dim="time").transpose(),
                                )

        if Sv:
            data_to_plot = data_to_plot * 1e-6


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
