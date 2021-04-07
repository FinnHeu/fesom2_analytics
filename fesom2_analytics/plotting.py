"""Module to plot transport associated analytics from ocean model pyfesom2.0"""

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



# Eventually plot the overview figure
def plot_overview(
    section_start,
    section_end,
    data_u_reg,
    data_v_reg,
    lon,
    lat,
    extent,
    elem_no_nan,
    save_figures,
):
    """
    plot_overview.py

    Plots global map with chosen section and regional map with section and regional total velocity magnitude (np.sqrt(u² + v²))

    Inputs:
    ----------------------------------------
    section_start (tuple, list[1,2])
    section_end (tuple, list[1,2])
    data_u_reg (xr.datarray)
    data_v_reg (xr.dataarray)
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

    # ax.plot(coords_array[:,0],
    #            coords_array[:,1],
    #            transform=ccrs.PlateCarree(),
    #            color='w',
    #            marker='.',
    #            markersize=1
    #        )

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
