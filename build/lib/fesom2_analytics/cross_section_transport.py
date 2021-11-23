from transports_v2 import *


def cross_section_transports(section,
                             mesh_path,
                             data_path,
                             mesh_diag_path,
                             years,
                             use_great_circle=True,
                             how='mean',
                             add_extent=1,
                             abg=[50, 15, -90],
                             add_TS=False,
                             chunks={'elem': 1e4}

                             ):
    '''
    cross_section_transports.py

    Computes the horizontal transport across a vertical section from fesom2 velocity output on mesh elements, by computing the intersections
    of the section with the elements.

    Inputs
    ------
    section (list, str)
        either a list of the form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section: 'FRAMSTRAIT', 'BSO'
    mesh_path (str)
        directory where the mesh files are stored
    data_path (str)
        directory where the data is stored
    mesh_diag_path (str: optional, default=None)
        directory where the mesh_diag file is stored, if None it is assumed to be located in data_path
    use_great_circle (bool)
        compute the section waypoints along a great great circle (default=True)
    how (str)
        either 'mean' for time mean transport or 'ori' for original data (default='mean')
    add_extent (int, float)
        the additional extent of the cutoutbox [lon_start, lon_end, lat_start, lat_end],
        choose as small as possible (small for high resolution meshes and large for low resolution meshes)
        this will impove the speed of the function (default = 1°)
    abg (list)
        rotation of the velocity data (default=[50,15,-90])
    add_TS (bool)
        add temperature and salinity to the section (default=False)
    chunks (dict)
        chunks for parallelising the velocity data (default: chunks={'elem': 1e4})

    Returns
    -------
    ds (xarray.Dataset)
        dataset containing all output variables
    section (dict)
        dictionary containing all section information

    '''

    # Wrap the subfunctions up
    mesh, mesh_diag, files, section = _ProcessInputs(
        section, mesh_path, data_path, mesh_diag_path, years, how, use_great_circle)

    section_waypoints, mesh, section = _ComputeWaypoints(
        section, mesh, use_great_circle)

    elem_box_nods, elem_box_indices = _ReduceMeshElementNumber(
        section_waypoints, mesh, section, add_extent)

    elem_box_nods, elem_box_indices, cell_intersections = _LinePolygonIntersections(
        mesh, section_waypoints, elem_box_nods, elem_box_indices)

    distances_between, distances_to_start, layer_thickness, grid_cell_area = _CreateVerticalGrid(
        cell_intersections, section, mesh)

    ds = _CreateDataset(files, mesh, elem_box_indices, elem_box_nods,
                        distances_between, distances_to_start, grid_cell_area, how, abg, chunks)

    ds = _ComputeTransports(ds, mesh, section, cell_intersections,
                            section_waypoints, use_great_circle)

    if add_TS:
        ds = _AddTempSalt(section, ds, data_path, mesh)

    return ds, section


if __name__ == "__main__":

    import sys
    import xarray as xr

    section = sys.argv[1]
    mesh_path = sys.argv[2]
    data_path = sys.argv[3]
    mesh_diag_path = sys.argv[4]
    years = sys.argv[5]
    use_great_circle = sys.argv[6],
    how = sys.argv[7],
    add_extent = sys.argv[8],
    abg = sys.argv[9],
    add_TS = sys.argv[10],
    chunks = sys.argv[11],
    save_path = sys.argv[12]

    # Arange years to ascending vector
    years = np.arange(years[0], years[-1]+1)

    ds, section = cross_section_transports(section,
                                           mesh_path,
                                           data_path,
                                           mesh_diag_path,
                                           years,
                                           use_great_circle=use_great_circle,
                                           how=how,
                                           add_extent=add_extent,
                                           abg=abg,
                                           add_TS=add_TS,
                                           chunks=chunks
                                           )

    ds.to_netcdf(save_path)
