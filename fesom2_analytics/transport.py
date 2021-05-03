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
import pyproj
from .plotting import *






def process_inputs(
    year_start,
    year_end,
    section,
    savepath_regional_data,
    savepath_transport_data,
    save_transport_output,
    save_regional_output,
):
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

    ############################ Get presets for start and end of section or convert custom values
    preset_sections = ["BSO", "BSX", "BEAR_SVAL", "SVAL_KVITOYA", "KVITOYA_FJL", "ST_ANNA_THROUGH"]
    if isinstance(section, str):
        if section in preset_sections:

            if section == "BSO":
                section_start = (19.999, 79)
                section_end = (19.999, 69)

            elif section == "BSX":
                section_start = (64.1, 80.9)
                section_end = (64.1, 76)

            # elif section == "BEAR_SVAL":
            #     section_start = (16.682139, 76.737635)
            #     section_end = (19.0544028822795, 74.44233057788212)

            elif section == "SVAL_KVITOYA":
                section_start = (28, 80.2)
                section_end = (32.5, 80.2)

            elif section == "KVITOYA_FJL":
                section_start = (32.5, 80.2)
                section_end = (47.15, 80.2)

            elif section == "ST_ANNA_THROUGH":
                section_start = (64.7, 80.9)
                section_end = (79.75, 80.9)


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

        print("\n----> Custom section:" + str(section_start) + str(section_end) + "(°E,°N)")


    # Check if 0°E is crossed
    if (section_start[0] > 0) & (section_end[0]) < 0 | (section_start[0] < 0) & (section_end[0]) > 0:
        across_0E = True

        print('Section crosses 0°E: Rotating Grid by 90° westward')
        section_start[0] = section_start[0] + 90
        section_end[0] = section_end[0] + 90

    else: across_0E = False



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




def load_data(path_mesh, path_data, time_range, across_0E):
    """
    load_data.py

    Load fesom2.0 mesh and velocity data

    Inputs:
    ---------------------------------
    path_mesh (str)
    path_data (str)
    time_range (np.ndarray)

    Returns:
    ---------------------------------
    mesh: pyfesom mesh object
    data_u (xr.dataarray): dataset containing global velocity fields
    data_v (xr.dataarray): dataset containing global velocity fields
    """

    ################################# Load mesh
    print("\n----> Loading mesh file")

    mesh = pf.load_mesh(path_mesh)#, abg=[50, 15, -90])

    # Rotate grid 90° westward when section crosses 0°E
    if across_0E:
        mesh.x2 = mesh.x2 + 90

    ################################# Load velocity data
    print("\n----> Loading velocity files")
    print(path_data)

    # grab files dependent on the years chosen
    year_str = [str(year) for year in time_range]

    file_str_u = ["u.fesom." + year + ".nc" for year in year_str]
    file_str_v = ["v.fesom." + year + ".nc" for year in year_str]

    file_path_u = [join(path_data, file) for file in file_str_u]
    file_path_v = [join(path_data, file) for file in file_str_v]

    # Load data into dataset
    data_u = xr.open_mfdataset(
        file_path_u, chunks={"time": 120, "elem": 10000}, combine="by_coords"
    )

    data_v = xr.open_mfdataset(
        file_path_v, chunks={"time": 120, "elem": 10000}, combine="by_coords"
    )

    return mesh, data_u, data_v





def cut_to_region(
    section_start,
    section_end,
    mesh,
    data_u,
    data_v,
    filename_regional_data,
    savepath_regional_data,
    save_regional_output,
    extent=2.5,
):
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

    # cut data to region
    print("\n----> Crop data to cutout box")

    # elem_no_nan: beinhaltet die indizes der dreiecke, die innerhalb der region liegen
    # no_nan_triangles: boolean array der größe von u, v mit True für innerhalb und False für außerhalb der region
    elem_no_nan, no_nan_triangles = pf.cut_region(mesh, box)

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





def create_polygons_and_line(elem_no_nan, mesh, section_start, section_end, use_great_circle):
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
    polygon_list
    """

    # Create a shapely line element that represents the section
    print("\n----> Compute line element for section")

    if use_great_circle:
        print('computing waypoints along great circle with WGS84 ellipsoid')

        g = pyproj.Geod(ellps='WGS84')

        lonlat = g.npts(section_start[0],
                        section_start[1],
                        section_end[0],
                        section_end[1],
                        1000
                        )

        lonlat = np.array(lonlat)

    else:
        if section_start[0] == section_end[0]:
            print('section along longitude')

        elif section_start[1] == section_end[1]:
            print('section along latitude')

        lonlat = [list(section_start), list(section_end)]

    line_section = sg.LineString(lonlat)

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

    return line_section, polygon_list, lonlat





def find_polygon_intersects(
    polygon_list, line_section, elem_no_nan, lon, lat, data_u_reg, data_v_reg, mesh, use_great_circle
):
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
    use_great_circle

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
        "\n----> Find polygons that intersect with the section and compute the intersection coordinates and distances")

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
                    list(coords[ii])[-1],
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
        coords_array[ii, 2] = list(coords[h[ii]][-1])[0]
        coords_array[ii, 3] = list(coords[h[ii]][-1])[1]

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

#     ############################### Compare the total distance to great circle distance from start point to end point

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

#     if (dist_ratio >= 1.01) | (dist_ratio <= 0.99):
#         raise Warning(
#             "The difference between the full great circle and the sum of segments is > 1%. This might yield inaccurate results!"
#         )
#         UI_1 = input(
#             "The segements are multiplied by the ratio to get the right distances, okay? [True or False]"
#         )

#         if UI_1:
#             dist_array = dist_array * dist_ratio

#     else:
#         print("This is withtin the limits of uncertainty")

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
        lat_array)





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
    shape_coord = coords_array.shape
    for i in range(shape_coord[0]):
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

    # Compute the central longitude and latitude of each cell
    # central_lon = coords_array[:]

    # Sort by distance
    h_sort = np.argsort(central_dist)

    return h_sort, central_dist





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
        print("The section start coordinate:", section_start, "(°E, °N) is in the ocean")
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







def normal_vector(coords_array, use_great_circle, section_start, section_end):


    if use_great_circle:

        segment_vec = coords_array[:,:2] - coords_array[:,2:] #compute the segment vector connecting the intersections
        normal_vec = np.array([segment_vec[:,1], - segment_vec[:,0]]) # compute the normal vector for each segment
        norm = np.sqrt(normal_vec[0,:]**2 + normal_vec[1,:]**2) # compute the 2-norm of each normal vector
        normed_normal_vec = normal_vec * norm[np.newaxis,:]**-1 # normalise normal vector

    else:
        # along longitude
        if section_start[0] == section_end[0]:
            normed_normal_vec = np.array((1, 0)).repeat(coords_array.shape[0]).reshape(2,coords_array.shape[0])

        # along latitude
        elif section_start[1] == section_end[1]:
            normed_normal_vec = np.array((0, 1)).repeat(coords_array.shape[0]).reshape(2,coords_array.shape[0])

        # other
        else:
            normal_vec = np.array([section_end[0] - section_start[0], section_end[1] - section_start[1]])
            norm = np.sqrt(normal_vec[0,:]**2 + normal_vec[1,:]**2) # compute the 2-norm of each normal vector
            normed_normal_vec = normal_vec * norm[np.newaxis,:]**-1 # normalise normal vector
            normed_normal_vec = normed_normal_vec.repeat(coords_array.shape[0]).reshape(2,coords_array.shape[0])


    # length test == 1
    length_test = np.sqrt(normed_normal_vec[0]**2 + normed_normal_vec[1]**2)
    if any(1 - np.abs(length_test) > 1e-10):
        raise ValueError('Length of the normalized normal vector != 1 +- 1e-10')

    # angle test == 0
    angle_test = [np.dot(normed_normal_vec[:,i], segment_vec[i]) for i in range(segment_vec.shape[0])]
    if not any(np.abs(length_test) > 1e-10):
        raise ValueError('Angle between normalized normal vector and segment vector != 90°')


    return normed_normal_vec






def compute_transport(u_array, v_array, normed_normal_vec, area_array):
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

    # Compute the across section velocity: u_across = u_vec * n_vec [time x nods x depth]
    print("Compute across section velocity")
    velocity_across = u_array.u * normed_normal_vec[0,:][np.newaxis,:,np.newaxis] + v_array.v * normed_normal_vec[1,:][np.newaxis,:,np.newaxis]

    # Compute the across section transport: velocity_across * area
    print("Compute across section transport")
    transport_across = velocity_across * area_array[np.newaxis, :, :]

#     weight = np.abs(np.diff(mesh.zlev))[np.newaxis, np.newaxis, :]

    return velocity_across, transport_across#, section_normal_vec, section_normal_vec_normed





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
    area_array,
    coords_array,
    across_0E,
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
    # Rotate longitude back
    if across_0E:
        lon = lon - 90

        coords_array[:,0] = coords_array[:,0] - 90
        coords_array[:,2] = coords_array[:,2] - 90

    print("\n----> Preparing final dataset")
    ds = xr.Dataset(
        {
            "transport_across": xr.DataArray(
                data=transport_across.values,
                dims=["time", "central_dist", "depth"],
                coords={
                    "time": transport_across.time.values,
                    "central_dist": central_dist,
                    "depth": transport_across.nz1.values,
                },
                attrs={"_FillValue": np.nan, "units": "m³/s"},
            ),
            "velocity_across": xr.DataArray(
                data=velocity_across.values,
                dims=["time", "central_dist", "depth"],
                coords={
                    "time": velocity_across.time.values,
                    "central_dist": central_dist,
                    "depth": velocity_across.nz1.values,
                },
                attrs={"_FillValue": np.nan, "units": "m³/s"},
            ),
            "longitude": xr.DataArray(data=lon),
            "latitude": xr.DataArray(data=lat),
            "elem_array": xr.DataArray(
                data=elem_array, dims=["central_dist", "tri"], coords={"tri": ["ind1", "ind2", "ind3"]}
            ),
            "width": xr.DataArray(data=dist_array,
                                  dims=["central_dist"]
                                 ),
            "area_weight": xr.DataArray(data=area_array,
                                        dims=["central_dist", "depth"]
                                       ),
            "intersection_coords": xr.DataArray(data=coords_array,
                                                dims=["central_dist", "lonlatlonlat"],
                                                coords={"lonlatlonlat": ["lon1", "lat1", "lon2", "lat2"]}
            )

#             "lon_section": xr.DataArray(data=lon_array,
#                                         dims=["central_dist"]
#                                        ),
#             "lat_section": xr.DataArray(data=lat_array,
#                                         dims=["central_dist"]
#                                        )
        }
    )

    # sort by distance to section start
    ds = ds.isel(central_dist=h_sort)

    if save:
        filename = join(savepath_transport_data, filename_transport_data)
        ds.to_netcdf(path=filename)

        print("Dataset saved at: " + filename)
    return ds

def across_section_transport(year_start,
year_end,
section,
path_mesh,
path_data,
savepath_regional_data,
savepath_transport_data,
filename_regional_data,
filename_transport_data,
save_transport_output,
save_regional_output,
use_great_circle
):

    time_range, section_start, section_end = process_inputs(year_start,
                                                        year_end,
                                                        section,
                                                        savepath_regional_data,
                                                        savepath_transport_data,
                                                        save_transport_output,
                                                        save_regional_output,
                                                    )

    mesh, data_u, data_v = load_data(path_mesh, path_data, time_range)

    box, elem_no_nan, no_nan_triangles, data_u_reg, data_v_reg, lon, lat, extent = cut_to_region(
                                                                                            section_start,
                                                                                            section_end,
                                                                                            mesh,
                                                                                            data_u,
                                                                                            data_v,
                                                                                            filename_regional_data,
                                                                                            savepath_regional_data,
                                                                                            save_regional_output,
                                                                                            extent=2.5,
                                                                                        )

    line_section, polygon_list, lonlat = create_polygons_and_line(elem_no_nan,
                                                                  mesh,
                                                                  section_start,
                                                                  section_end,
                                                                  use_great_circle
                                                                 )

    start_point_bool, end_point_bool = start_end_is_land(section_start,
                                                         section_end,
                                                         polygon_list
                                                        )

    intersect_bool, coords_array, dist_array, depth_array, area_array, u_array, v_array, elem_array, lon_array, lat_array = find_polygon_intersects(
    polygon_list, line_section, elem_no_nan, lon, lat, data_u_reg, data_v_reg, mesh, use_great_circle)

    h_sort, central_dist = sort_by_dist_to_section_start(coords_array,
                                                         section_start
                                                        )

    normed_normal_vec = normal_vector(coords_array,
                                      use_great_circle,
                                      section_start,
                                      section_end
                                     )

    velocity_across, transport_across = compute_transport(u_array,
                                                          v_array,
                                                          normed_normal_vec,
                                                          area_array
                                                         )

    ds_transport = create_output(
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
    area_array,
    coords_array,
    save=save_transport_output,
)
    return ds_transport, mesh
