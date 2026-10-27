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
  x2c = np.zeros(((time_len), lat_len, lon_len ))
  
  x1_wk  = np.zeros((time_len-1)*(year_end-year_start+1))
  x2_wk  = np.zeros((time_len-1)*(year_end-year_start+1))

  var1    = np.zeros((lat_len, lon_len))
  var2    = np.zeros((lat_len, lon_len))

  sig = np.zeros((lat_len, lon_len))
  
  for year in range(year_start, year_end +1) : 

    print(f"==== year: {year}")
    
    infile1 = f"../Data/ESA_CCI_SM_GPCP_regridded_3.0deg_7day/soilmoisture_3deg_7d_{year}{month:02}.nc"
    infile2 = f"../Data/ESA_CCI_SM_GPCP_regridded_3.0deg_7day/precip_3deg_7d_{year}{month:02}.nc"

    ds1 = xr.open_dataset(infile1)
    #print(ds1.dims)

    ds1 = ds1.rename({'lon': 'longitude', 'lat': 'latitude'})
    #print(ds1.dims)
    ds1 = ds1.transpose('latitude', 'longitude', 'time')
    #print(ds1.dims)

    #print(ds1)       
    #for var in ds1.data_vars:
    #   print(var, ds1[var].dims)

    ds2 = xr.open_dataset(infile2)    
    ds2 = ds2.transpose('latitude', 'longitude', 'time')
    #print(ds2.dims)    
    
    #print(ds2)       
    #for var in ds2.data_vars:
    #   print(var, ds2[var].dims)

    lat = ds2["latitude"].values
    lon = ds2["longitude"].values

    for time in range(0,time_len) :    
      x1c[time,:,:] += ds1["sm"].isel(time=time) / (year_end-year_start+1)
      x2c[time,:,:] += ds2["__xarray_dataarray_variable__"].isel(time=time) / (year_end-year_start+1)
    
    for time in range(0,time_len) :
      x1[(time_len-1)*(year-year_start)+time,:,:] = ds1["sm"].isel(time=time) 
      x2[(time_len-1)*(year-year_start)+time,:,:] = ds2["__xarray_dataarray_variable__"].isel(time=time) 

  # Output
  np.save(f"Fig_data/SM_clim_mon{month:02}.npy",np.mean(x1c, axis=0))
  np.save(f"Fig_data/Pr_clim_mon{month:02}.npy",np.mean(x2c, axis=0))

  
