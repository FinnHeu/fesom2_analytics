import numpy as np
import xarray as xr
import tqdm.notebook as tqdm
from dask.diagnostics import ProgressBar


def crop_and_save_on_nods(files, mesh, save_name, box='BS'):
    '''
    crop_and_save.py

    Crop global datasets to regional

    Inputs:
    ---------------------------------
    files: list of all file paths to process
    mesh: fesom.mesh (pf.load_mesh())
    save_name: list of all file paths for the cropped data
    box: region name (default Barents Sea 'BS')
    merge: megre all the cropped files by xr.open_mfdataset() (default True)'''

    # find indices for region
    if box == 'BS':
        h = np.where((mesh.x2 >= 0) & (mesh.x2 <= 90) & (mesh.y2 >= 66) & (mesh.y2 <=90))[0]

    # open each file, convert to dask array and select regional data
    for ii in tqdm(range(len(files))):
        # open file
        ds = xr.open_dataset(files[ii], chunks={'nod2': 1e5}).isel(nod2=h).astype('f4')
        # add x, y
        ds['lon'] = mesh.x2[h]
        ds['lat'] = mesh.y2[h]

        ds.to_netcdf(save_name[ii])
        print(save_name[ii])

    print('----> Finished cropping and saving files')
    
    return
