import pyfesom2 as pf
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
ProgressBar().register()
from tqdm.notebook import tqdm



def regional_heat_content(path_temp, path_mesh, years, box=[20, 60, 70, 80], rho=1030, cp=4190, T_ref=0, create_dataset=True):

    '''
    regional_heat_content.py

    compute the total heat content (relative to T_ref) in a chosen region

    Inputs
    ------
    path_temp: str, folder to fesom temperature files
    path_mesh: str, folder to fesom mesh files
    years: int, years to compute

    Returns
    -------

    '''
    data_base_path = path_temp
    mesh_base_path = path_mesh

    mesh = pf.load_mesh(mesh_base_path, abg=[0,0,0])
    mesh_diag = xr.open_dataset(data_base_path + 'fesom.mesh.diag.nc')

    # find regional indices
    inds_reg = np.where((mesh.x2 >= box[0]) & (mesh.x2 <= box[1]) & (mesh.y2 >= box[2]) & (mesh.y2 <= box[3]))[0]

    # Compute the mean nodal area at the midpoints between two layers (at the temperature and salinity locations)
    # by computing the mean area of the above and below areas and multiplying with the layer thickness (?)
    a = mesh_diag.nod_area.shape
    nodal_area = np.zeros((a[0]-1, a[1]))

    for depth in tqdm(range(mesh.nlev-1)):
        nodal_area[depth,:] = (mesh_diag.nod_area[depth,:] + mesh_diag.nod_area[depth+1,:]).values / 2

    # Compute the Layer thickness
    layer_depth = np.diff(mesh_diag.zbar)[:, np.newaxis]
    layer_z = np.cumsum(layer_depth)

    # Compute the Volume of each Nod
    nodal_volume = nodal_area * abs(layer_depth)

    # Load temperature data
    ds_temp = pf.get_data(data_base_path,
                      'temp',
                      years,
                      mesh,
                      how='ori',
                      compute=False,
                      chunks={'nod2': 1e4}
                     )


    # Crop to Barents Sea mask
    lon = mesh.x2[inds_reg]
    lat = mesh.y2[inds_reg]

    ds_temp = ds_temp.isel(nod2=inds_reg).load()
    nodal_volume = nodal_volume[:,inds_reg].transpose()[np.newaxis,:,:]

    # Create a mask that masks topography values and temperatures below 0°C
    mask_temp = ds_temp.values > T_ref

    # Compute heat content for each nodal volume
    heat_content = rho * cp * ds_temp.values * nodal_volume * mask_temp


    if create_dataset:

        times = ds_temp.time
        nod2 = inds_reg
        depth = mesh_diag.Z.values


        # create dataset
        ds = xr.Dataset({
            'heat_content': xr.DataArray(
                        data   = heat_content,
                        dims   = ['time','nod2','depth'],
                        coords = {'time': times,
                                  'nod2': nod2,
                                  'depth': depth
                                 },
                        attrs  = {
                            'units'     : 'Joule'
                            }
                        ),

            'temp': xr.DataArray(
                        data   = ds_temp.values,
                        dims   = ['time','nod2','depth'],
                        coords = {'time': times,
                                  'nod2': nod2,
                                  'depth': depth
                                 },
                        attrs  = {
                            'units'     : 'degreee Celsius'
                            }
                        ),

            'nodal_volume': xr.DataArray(
                        data   = nodal_volume * np.ones_like(heat_content),
                        dims   = ['time','nod2','depth'],
                        coords = {'time': times,
                                  'nod2': nod2,
                                  'depth': depth
                                 },
                        attrs  = {
                            'units'     : 'm^3'
                            }
                        ),

            'longitude': xr.DataArray(
                        data   = lon,
                        dims   = ['nod2'],
                        coords = {'nod2': nod2,
                                  },
                        attrs  = {
                            'units'     : 'degree E'
                            }
                        ),

            'latitude': xr.DataArray(
                        data   = lat,
                        dims   = ['nod2'],
                        coords = {'nod2': nod2,
                                  },
                        attrs  = {
                            'units'     : 'degree E'
                            }
                        ),

        }
        )

    return ds
