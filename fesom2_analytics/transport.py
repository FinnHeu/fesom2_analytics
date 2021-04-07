"""Module for computing transports across sections in ocean model pyfesom2"""

import pyfesom2 as pf
import xarray as xr
import numpy as np
import scipy as sc
from os.path import join, isdir, isfile
from os import mkdir
import shapely.geometry as sg
from great_circle_calculator.great_circle_calculator import distance_between_points
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")

from dask.diagnostics import ProgressBar
import glob
from .plotting import *






def process_inputs(year_start, year_end, section, savepath_regional_data, savepath_transport_data, save_transport_output, save_regional_output):
    """
    process_inputs.py

    Processes the input arguments for further use.

    Inputs:
    ----------------------------------------------
    year start
    year end
    section: str or list, if list of form [lon, lat, lon, lat]
    savepath_regional_data

    Returns:
    ----------------------------------------------
    time_range
    section_start
    section_end
    """

    # Create time vector for pf.get_data
    time_range = np.arange(year_start, year_end + 1)

    print("\n----> Chosen timerange: " + str(year_start) + " to " + str(year_end))

    # Get presets for start and end of section
    preset_sections = ["BSO", "BSX", "BEAR_SVAL", "SVAL_FJL"]
    if isinstance(section, str):
        if section in preset_sections:

            if section == "BSO":
                section_start = (19.0544028822795, 74.44233057788212)
                section_end = (21.925523912516116, 70.18597139727804)

            elif section == "BSX":
                section_start = (59.5, 80.0)
                section_end = (67.5, 76.8)

            elif section == "BEAR_SVAL":
                section_start = (16.682139, 76.737635)
                section_end = (19.0544028822795, 74.44233057788212)

            elif section == "SVAL_FJL":
                section_start = (24.110024558329297, 79.92502447814552)
                section_end = (46.69791436099627, 80.60826231880695)

            # add further sections
            print(
                "\n----> Preset section chosen:",
                section,
                section_start,
                section_end,
                "(°E, °N)",
            )

        else:
            raise ValueError(
                "The chosen preset "
                + section
                + " does not exist: Please choose from: BSO, BSX, BEAR_SVAL, SVAL_FJL"
            )

    # If section is custom convert to list
    elif isinstance(section, list):
        section_start = (section[0], section[1])
        section_end = (section[2], section[3])

        print("Custom section:" + section_start + section_end + "(°E,°N)")

    ######################################## Check if folder structure exists

    # Check if figure folder exists
    if not isdir(join(savepath_regional_data, "figures_transport")):
        print("\n----> Figure folder does not exist: Creating folder")
        print(join(savepath_regional_data, "figures_transport"))

        mkdir(join(savepath_regional_data, "figures_transport"))
    else:
        print("\n----> Figure folder exists")
        print(join(savepath_regional_data, "figures_transport"))

    # Check if transport output folder exists
    if save_transport_output:
        if not isdir(savepath_transport_data):
            print("\n----> Transport output folder does not exist: Creating folder")
            print(savepath_transport_data)

            mkdir(savepath_transport_data)

    # check if regional data output exists
    if save_regional_output:
        if not isdir(savepath_regional_data):
            print("\n----> Regional data output folder does not exist: Creating folder")
            print(savepath_regional_data)

            mkdir(savepath_regional_data)

    return time_range, section_start, section_end




def load_data(path_mesh, path_data, years):
    """
    load_data.py

    Load fesom2.0 mesh and velocity data

    Inputs:
    ---------------------------------
    path_mesh (str)
    path_data (str)
    years (int or list or array)

    Returns:
    ---------------------------------
    mesh
    data_u (xr.dataarray)
    data_v (xr.dataarray)
    """

    ################################# Load mesh
    print("\n----> Loading mesh file")

    mesh = pf.load_mesh(path_mesh, abg=[50, 15, -90])

    ################################# Load velocity data
    print("\n----> Loading velocity files")
    print(path_data)

    # grab files
    files_u = glob.glob(join(path_data, "u.fesom.*.nc"))
    files_v = glob.glob(join(path_data, "v.fesom.*.nc"))

    data_u = xr.open_mfdataset(
        files_u, chunks={"time": 120, "elem": 10000}, combine="by_coords"
    )

    data_v = xr.open_mfdataset(
        files_v, chunks={"time": 120, "elem": 10000}, combine="by_coords"
    )

    return mesh, data_u, data_v





def cut_to_region(section_start, section_end, mesh, data_u, data_v, filename_regional_data, savepath_regional_data, save_regional_output, extent=2):
    """
    cut_to_region.py

    Cuts the original (global) fesom velocity data to the region of the section to save disk space and accelerate the computation time

    Inputs:
    -----------------------------------
    section_start (tuple, list[1,2])
    section_end (tuple, list[1,2])
    save_regional_output (bool)
    extent (int, float)

    Returns:
    -----------------------------------
    box
    elem_no_nan
    no_nan_triangles
    data_u_reg
    data_v_reg
    lon
    lat
    extent

    """
    # create empty list
    box = []

    # sort start and end point of section in ascending order to apply pf.cut_region
    if section_start[0] < section_end[0]:
        box.append(section_start[0] - extent)
        box.append(section_end[0] + extent)
    else:
        box.append(section_end[0] - extent)
        box.append(section_start[0] + extent)

    if section_start[1] < section_end[1]:
        box.append(section_start[1] - extent)
        box.append(section_end[1] + extent)
    else:
        box.append(section_end[1] - extent)
        box.append(section_start[1] + extent)

    print("\n----> Cutout box: " + str(box))

    # compute elements that belong to region

    # elem_no_nan: beinhaltet die indizes der dreiecke, die innerhalb der region liegen
    # no_nan_triangles: boolean array der größe von u, v mit True für innerhalb und False für außerhalb der region
    elem_no_nan, no_nan_triangles = pf.cut_region(mesh, box)

    # cut data to region
    print("\n----> Crop data to cutout box")

    # check if files already exist
    output_1 = filename_regional_data[:-3] + "_u.nc"
    output_2 = filename_regional_data[:-3] + "_v.nc"

    reg_file_1 = join(savepath_regional_data, output_1)
    reg_file_2 = join(savepath_regional_data, output_2)
#     print(reg_file_1, reg_file_2)

    if (isfile(reg_file_1)) & (isfile(reg_file_2)):

        print("files already exist, loading files from disk")
        # Enable a Dask progress bar
        ProgressBar().register()

        data_u_reg = xr.open_dataset(reg_file_1).load()
        data_v_reg = xr.open_dataset(reg_file_2).load()

    else:
        # load data
        data_u_reg = data_u.isel(elem=no_nan_triangles)
        data_v_reg = data_v.isel(elem=no_nan_triangles)

        # Enable a Dask progress bar
        ProgressBar().register()

        data_u_reg = data_u_reg.compute()
        data_v_reg = data_v_reg.compute()

        # save eventually
        if save_regional_output:

            # save data cutout
            data_u_reg.to_netcdf(reg_file_1)
            data_v_reg.to_netcdf(reg_file_2)
            print("\n----> Save regional files")

    # extract longitude and latitude
    lon = mesh.x2
    lat = mesh.y2

    return box, elem_no_nan, no_nan_triangles, data_u_reg, data_v_reg, lon, lat, extent




def create_polygons_and_line(elem_no_nan, mesh, section_start, section_end):
    """
    Create_polygons_and_line.py

    Uses shapely to create alist of all polygons and a section line in the regional Dataset

    Inputs:
    --------------------------------
    elem_no_nan
    section_start
    section_end

    Returns:
    --------------------------------
    line_section
    polygon_list"""
    # Create a shapely line element that represents the section
    print("\n----> Compute line element for section")
    line_section = sg.LineString([section_start, section_end])

    # Create a list of all polygons in the region from the coordinates of the nods
    print("\n----> Compute polygon elements for every grid cell in region")
    polygon_list = list()

    for ii in tqdm(range(elem_no_nan.shape[0])):
        polygon_list.append(
            sg.Polygon(
                [
                    (mesh.x2[elem_no_nan][ii, 0], mesh.y2[elem_no_nan][ii, 0]),
                    (mesh.x2[elem_no_nan][ii, 1], mesh.y2[elem_no_nan][ii, 1]),
                    (mesh.x2[elem_no_nan][ii, 2], mesh.y2[elem_no_nan][ii, 2]),
                ]
            )
        )

    return line_section, polygon_list





def start_end_is_land(section_start, section_end, polygon_list):
    """Start_end_is_land.py

    Checks if the start and end of section are on land or in ocean

    Inputs:
    ------------------------------
    section_start
    section_end,
    polygon_list

    Returns:
    ------------------------------
    start_point_bool
    end_point_bool

    """

    # Check if the first and last coordinate of the section are on land or ocean
    print(
        "\n----> Check if the start and end coordinates of the section are on land or in ocean"
    )

    # make shapely points from section start/ end
    start_point = sg.Point(section_start)
    end_point = sg.Point(section_end)

    start_point_bool = list()
    end_point_bool = list()

    # check if start and end or inside any polygon (ocean)
    for ii in range(len(polygon_list)):
        if start_point.within(polygon_list[ii]):
            start_point_bool.append(True)
        else:
            start_point_bool.append(False)

        if end_point.within(polygon_list[ii]):
            end_point_bool.append(True)
        else:
            end_point_bool.append(False)

    # Check the total number of polygons that contain start/ end (must not be >= 1)
    if np.sum(start_point_bool) == 1:
        print(
            "The section start coordinate:", section_start, "(°E, °N) is in the ocean"
        )
    elif np.sum(start_point_bool) == 0:
        print("The section start coordinate:", section_start, "(°E, °N) is on land")
    else:
        raise ValueError("Section start neither on land nor in ocean")

    if np.sum(end_point_bool) == 1:
        print("The section end coordinate:", section_end, "(°E, °N) is in the ocean")
    elif np.sum(end_point_bool) == 0:
        print("The section end coordinate:", section_end, "(°E, °N) is on land")
    else:
        raise ValueError("Section end neither on land nor in ocean")

    return start_point_bool, end_point_bool


def find_polygon_intersects(polygon_list, line_section, elem_no_nan, lon, lat, data_u_reg, data_v_reg, mesh):
    """
    Find_polygon_intersects.py


    Compute poylgons that are intersected by the section

    Inputs:
    -------------------------------
    polygon_list
    line_section
    elem_no_nan
    lon,
    lat
    data_u_reg
    data_v_reg
    mesh

    Returns:
    -------------------------------
    ntersect_bool
    coords_array
    dist_array
    depth_array
    area_array
    u_array
    v_array
    elem_array
    lon_array
    lat_array
    """
    # Find intersecting polygons (grid cells) and compute the intersection coordinates and the great circle distance between them
    print(
        "\n----> Find polygons that intersect with the section and compute the intersection coordinates and distances"
    )

    intersect_bool = list()
    coords = list()
    gc_dist = list()
    intersect_points = list()

    # check for intersections
    for ii in range(elem_no_nan.shape[0]):
        coords.append(polygon_list[ii].intersection(line_section).coords)

        # if no intersections (coords == [])
        if not coords[ii]:
            # fill boolean array with False (no intersects)
            intersect_bool.append(False)
            # fill distance list with nan
            gc_dist.append(np.nan)

        # if exist intersections (coords != [] )
        else:
            # fill boolean array with True (intersects exists)
            intersect_bool.append(True)
            # if there are intersects compute the distance between the points
            gc_dist.append(
                distance_between_points(
                    list(coords[ii])[0],
                    list(coords[ii])[1],
                    unit="meters",
                    haversine=True,
                )
            )
    print("Found " + str(np.nansum(intersect_bool)) + " polygons intersected")

    # Convert to numpy array
    intersect_bool = np.array(intersect_bool, dtype=bool)

    ###################### Create numpy arrays that are filled with ONLY those datapoints of polygons intersected by the section

    # Find grid cells that are intersected
    h = np.where(intersect_bool)[0]

    # Fill array with intersection coords [lon_1, lat_1, lon_2, lat_2]
    coords_array = np.ones((len(h), 4)) * np.nan
    for ii in range(len(h)):
        coords_array[ii, 0] = list(coords[h[ii]][0])[0]
        coords_array[ii, 1] = list(coords[h[ii]][0])[1]
        coords_array[ii, 2] = list(coords[h[ii]][1])[0]
        coords_array[ii, 3] = list(coords[h[ii]][1])[1]

    # Fill array with the distances of the intersections
    dist_array = np.array(gc_dist)[h]

    # Fill array with the velocities
    u_array = data_u_reg.isel(elem=h)
    v_array = data_v_reg.isel(elem=h)

    # Compute and fill depth of grid cell array
    depth_array = abs(np.diff(mesh.zlev))

    # Compute area of intersected grid cell
    area_array = dist_array[:, np.newaxis] * depth_array[np.newaxis, :]

    # Select nods
    elem_array = elem_no_nan[h, :]

    # select lons, lats
    lon_array = lon[h]
    lat_array = lat[h]

    ############################### Compare the total distance to great circle distance from start point to end point

    print(
        "\n----> Compute the ratio of great circle distance to the sum of all polygon intersection segments"
    )
    # Compute the great circle distance between outer intersection coordinates
    great_circle_dist = distance_between_points(
        (np.min(coords_array[:, [0, 2]]), np.max(coords_array[:, [1, 3]])),
        (np.max(coords_array[:, [0, 2]]), np.min(coords_array[:, [1, 3]])),
        unit="meters",
        haversine=True,
    )

    # Compute ratio between sum of segments and full great circle
    dist_ratio = np.nansum(dist_array) / great_circle_dist

    print("The ratio (sum(segments) / great_circle) is: " + str(dist_ratio))

    if (dist_ratio >= 1.01) | (dist_ratio <= 0.99):
        raise Warning(
            "The difference between the full great circle and the sum of segments is > 1%. This might yield inaccurate results!"
        )
        UI_1 = input(
            "The segements are multiplied by the ratio to get the right distances, okay? [True or False]"
        )

        if UI_1:
            dist_array = dist_array * dist_ratio

    else:
        print("This is withtin the limits of uncertainty")

    return (
        intersect_bool,
        coords_array,
        dist_array,
        depth_array,
        area_array,
        u_array,
        v_array,
        elem_array,
        lon_array,
        lat_array,
    )




def sort_by_dist_to_section_start(coords_array, section_start):

    """
    Sort_by_dist_to_section_start.py

    Sorts the transport array by distance to start point of section

    Inputs:
    ------------------------------------
    coords_array
    section_start

    Returns:
    ------------------------------------
    h_sort
    central_dist

    """

    d_1 = list()
    d_2 = list()

    # Compute the central distance between each pair of intersection coords to the start of the section

    for i in range(210):
        d_1.append(
            distance_between_points(
                p1=(coords_array[i, 0], coords_array[i, 1]), p2=section_start
            )
        )

        d_2.append(
            distance_between_points(
                p1=(coords_array[i, 2], coords_array[i, 3]), p2=section_start
            )
        )

    # Compute the distance from the section start to the central point of the polygon intersection
    central_dist = np.array(d_1) + (np.array(d_2) - np.array(d_1)) / 2

    # Sort by distance
    h_sort = np.argsort(central_dist)

    return h_sort, central_dist




def compute_transport(section_end, section_start, u_array, v_array, area_array):
    """
    compute_transport.py

    Computes the transport along a section in fesom2.0 ocean model

    Inputs:
    ----------------------------------
    section_end
    section_start
    u_array
    v_array,
    area_array

    Returns:
    ----------------------------------
    velocity_across
    transport_across
    section_normal_vec
    section_normal_vec_normed
    """


    print("\n----> Compute the across section transport")

    # Compute the Normalenvektor of the section
    section_vector_x = section_end[0] - section_start[0]
    section_vector_y = section_end[1] - section_start[1]

    # Set y component of Normalenvektor to 1 and compute the x component
    section_normal_vec = np.array([-(section_vector_y / section_vector_x), 1])

    # Compute norm of section normal vector
    section_normal_vec_len = np.sqrt(
        section_normal_vec[0] ** 2 + section_normal_vec[1] ** 2
    )

    # Shrink section normal vec to length 1
    section_normal_vec_normed = (1 / section_normal_vec_len) * section_normal_vec

    if (
        np.abs(
            1
            - np.sqrt(
                section_normal_vec_normed[0] ** 2 + section_normal_vec_normed[1] ** 2
            )
        )
        > 1e-5
    ):
        raise Warning("The length of the sections Normalenvektor is != 1")

    # Compute the across section velocity: u_across = u_vec * n_vec [time x nods x depth]
    print("Compute across section velocity")
    velocity_across = (u_array.u * section_normal_vec_normed[0]) + (
        v_array.v * section_normal_vec_normed[1]
    )

    # Compute the across section transport: velocity_across * area
    print("Compute across section transport")
    transport_across = velocity_across * area_array[np.newaxis, :, :]

    # print('The (time-) mean transport across the section is: ', str(np.nansum(velocity_across * area_array[np.newaxis,:,:], axis=(1,2)) /1e6))

    return velocity_across, transport_across, section_normal_vec, section_normal_vec_normed


def create_output(
    transport_across,
    velocity_across,
    lon,
    lat,
    elem_array,
    savepath_transport_data,
    filename_transport_data,
    central_dist,
    h_sort,
    dist_array,
    save=False,
):
    """
    Create_output.py

    Creates xr.dataset from transport calculations

    Inputs:
    -----------------------------------
    transport_across
    velocity_across
    lon
    lat
    elem_array
    savepath_transport_data
    filename_transport_data
    central_dist
    h_sort
    dist_array
    save=True

    Returns:
    -----------------------------------
    ds

    """

    print("\n----> Preparing final dataset")
    ds = xr.Dataset(
        {
            "transp_across": xr.DataArray(
                data=transport_across.values,
                dims=["time", "dist", "depth"],
                coords={
                    "time": transport_across.time.values,
                    "dist": central_dist,
                    "depth": transport_across.nz1.values,
                },
                attrs={"_FillValue": np.nan, "units": "m³/s"},
            ),
            "velocity_across": xr.DataArray(
                data=velocity_across.values,
                dims=["time", "dist", "depth"],
                coords={
                    "time": velocity_across.time.values,
                    "dist": central_dist,
                    "depth": velocity_across.nz1.values,
                },
                attrs={"_FillValue": np.nan, "units": "m³/s"},
            ),
            "longitude": xr.DataArray(data=lon),
            "latitude": xr.DataArray(data=lat),
            "elem_array": xr.DataArray(
                data=elem_array, dims=["dist", "tri"], coords={"tri": np.arange(3)}
            ),
            "width": xr.DataArray(data=dist_array, dims=["dist"]),
        }
    )

    # sort by distance to section start
    ds = ds.isel(dist=h_sort)

    if save:
        filename = join(savepath_transport_data, filename_transport_data)
        ds.to_netcdf(path=filename)

        print("Dataset saved at: " + filename)
    return ds



def section_transport(path_data,
path_mesh,
savepath_regional_data,
filename_regional_data,
savepath_transport_data,
filename_transport_data,
save_transport_output,
plot_figures,
save_figures,
save_regional_output=True,
year_start=1958,
year_end=2005,
section='BSO'
):
    """
    section_transport.py

    Computes the transport across a given section (land to land) in ocean model fesom2.0

    Inputs:
    --------------------------------------
    path_data
    path_mesh
    savepath_regional_data
    filename_regional_data
    savepath_transport_data
    filename_transport_data
    save_transport_output
    save_regional_output=True
    year_start=1958
    year_end=2005
    section='BSO'

    Returns:
    ---------------------------------------
    ds_transport

    """
    # Process input data
    time_range, section_start, section_end = process_inputs(year_start,
                                                            year_end,
                                                            section,
                                                            savepath_regional_data,
                                                            savepath_transport_data,
                                                            save_transport_output,
                                                            save_regional_output
                                                           )


    # Load mesh and data
    mesh, data_u, data_v = load_data(path_mesh,
                                     path_data,
                                     time_range
                                    )

    # Cut to specific region to save diskspace
    box, elem_no_nan, no_nan_triangles, data_u_reg, data_v_reg, lon, lat, extent = cut_to_region(section_start,
                                                                                                 section_end,
                                                                                                 mesh,
                                                                                                 data_u,
                                                                                                 data_v,
                                                                                                 filename_regional_data,
                                                                                                 savepath_regional_data,
                                                                                                 save_regional_output=save_regional_output,
                                                                                                 extent=.2
                                                                                                )

    # Create polygons and lines in shapely
    line_section, polygon_list = create_polygons_and_line(elem_no_nan,
                                                          mesh,
                                                          section_start,
                                                          section_end
                                                         )


    # Find polygon intersects
    Intersect_bool, coords_array, dist_array, depth_array, area_array, u_array, v_array, elem_array, lon_array, lat_array = find_polygon_intersects(polygon_list,
                                                                                                                                                    line_section,
                                                                                                                                                    elem_no_nan,
                                                                                                                                                    lon,
                                                                                                                                                    lat,
                                                                                                                                                    data_u_reg,
                                                                                                                                                    data_v_reg,
                                                                                                                                                    mesh
                                                                                                                                                   )

    # Check if start and end of section are on land
    start_point_bool, end_point_bool = start_end_is_land(section_start,
                                                         section_end,
                                                         polygon_list
                                                        )

    # compute across section velocity and transport
    velocity_across, transport_across, section_normal_vec, section_normal_vec_normed = compute_transport(section_end,
                                                                                                         section_start,
                                                                                                         u_array,
                                                                                                         v_array,
                                                                                                         area_array
                                                                                                        )

    # Plot figures
    if plot_figures:
        plot_overview(section_start,
                      section_end,
                      data_u_reg,
                      data_v_reg,
                      lon,
                      lat,
                      extent,
                      elem_no_nan,
                      save_figures=save_figures
                     )

    # Get indices to sort array by distance to section start
    h_sort, central_dist = sort_by_dist_to_section_start(coords_array,
                                                         section_start
                                                        )

    # Create output file
    ds_transport = create_output(transport_across,
                                 velocity_across,
                                 lon,
                                 lat,
                                 elem_array,
                                 savepath_transport_data,
                                 filename_transport_data,
                                 central_dist,
                                 h_sort,
                                 dist_array,
                                 save=save_transport_output
                                )

    return ds_transport
