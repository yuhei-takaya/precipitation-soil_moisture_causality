import gc
import itertools
import os
import subprocess
import sys
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from pprint import pprint

import numpy as np
import xarray as xr

import Liang_funcs

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

year_start=1997
year_end=2023
time_len = 4 # number of data in each month
lat_len=60  # 3deg
lon_len=120   # 3deg

for month in range(1, 13) :

  x1  = np.zeros(((time_len)*(year_end-year_start+1), lat_len, lon_len ))
  x2  = np.zeros(((time_len)*(year_end-year_start+1), lat_len, lon_len ))

  x1c = np.zeros(((time_len), lat_len, lon_len )) # climatology
  x2c  = np.zeros(((time_len), lat_len, lon_len ))
  
  
  for year in range(year_start, year_end +1) : 

    print(f"==== year: {year}")
    
    infile2 = f"../Data/CMSAF_CLARA-A3_regridded_3.0deg_7day/SISdm_3deg_7d_{year}{month:02}.nc"

    ds2 = xr.open_dataset(infile2)
    ds2 = ds2.rename({'lon': 'longitude', 'lat': 'latitude'})    
    ds2 = ds2.transpose('latitude', 'longitude', 'time')
    #print(ds2.dims)    
    
    lat = ds2["latitude"].values
    lon = ds2["longitude"].values

    for time in range(0,time_len) :    
      x2c[time,:,:] += ds2["__xarray_dataarray_variable__"].isel(time=time) / (year_end-year_start+1)
    
    for time in range(0,time_len) :
      x2[(time_len-1)*(year-year_start)+time,:,:] = ds2["__xarray_dataarray_variable__"].isel(time=time) 

  np.save(f"Fig_data/SIS_clim_mon{month:02}.npy",np.mean(x2c, axis=0))
