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

for month in range(7, 13) :

  x1  = np.zeros(((time_len)*(year_end-year_start+1), lat_len, lon_len ))
  x2  = np.zeros(((time_len)*(year_end-year_start+1), lat_len, lon_len ))
  x1d = np.zeros(((time_len-1)*(year_end-year_start+1), lat_len, lon_len ))
  x2d = np.zeros(((time_len-1)*(year_end-year_start+1), lat_len, lon_len ))

  x1c = np.zeros(((time_len), lat_len, lon_len )) # climatology
  x2c  = np.zeros(((time_len), lat_len, lon_len ))
  
  x1_wk  = np.zeros((time_len-1)*(year_end-year_start+1))
  x2_wk  = np.zeros((time_len-1)*(year_end-year_start+1))
  x1d_wk = np.zeros((time_len-1)*(year_end-year_start+1))
  x2d_wk = np.zeros((time_len-1)*(year_end-year_start+1))  

  T21    = np.zeros((lat_len, lon_len))
  tau21l = np.zeros((lat_len, lon_len))  
  tau21b = np.zeros((lat_len, lon_len))

  T12    = np.zeros((lat_len, lon_len))
  tau12l = np.zeros((lat_len, lon_len))  
  tau12b = np.zeros((lat_len, lon_len))  

  tau12l_wk = np.zeros((lat_len, lon_len))    
  
  sig = np.zeros((lat_len, lon_len))
  
  output = f"LKIF_ESA_CCI_SM-GPCP_3deg_7d_{month:02}.nc"

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

  for i in range(0,lon_len) :
    for j in range(0, lat_len) :
      for year in range(year_start, year_end +1) :
        for time in range(0,time_len-1) :
          x1_wk[(time_len-1)*(year-year_start)+time]=x2[(time_len-1)*(year-year_start)+time,j,i] - x2c[time,j,i] #pr 
          x2_wk[(time_len-1)*(year-year_start)+time]=x1[(time_len-1)*(year-year_start)+time,j,i] - x1c[time,j,i] #sm
          x1d_wk[(time_len-1)*(year-year_start)+time]=x2[(time_len-1)*(year-year_start)+time+1,j,i] - x2[(time_len-1)*(year-year_start)+time,j,i] \
            - (x2c[time+1,j,i] - x2c[time,j,i]) #pr
          x2d_wk[(time_len-1)*(year-year_start)+time]=x1[(time_len-1)*(year-year_start)+time+1,j,i] - x1c[time+1,j,i] \
                                                     -(x1[(time_len-1)*(year-year_start)+time,j,i] - x1c[time,j,i]) #sm

      dt=1.0

      #T21[j,i], tau21l[j,i], tau21b[j,i], err99, err95, err90 = Liang_funcs.IF_np(x1_wk, x2_wk, x1d_wk, x2d_wk, dt)
      T12[j,i], tau12l[j,i], tau12b[j,i], err99, err95, err90 = Liang_funcs.IF_np(x2_wk, x1_wk, x2d_wk, x1d_wk, dt)


  # Output
  np.savez(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz", T12=T12, tau12l=tau12l)
      
  # Permutation test

  for b in range(0,5000) :

    print("month",month,"permutation test", b)
    boot_years = np.random.choice(
    np.arange(year_start, year_end + 1),
    size=(year_end - year_start + 1),
    replace=False
    )
  
    for i in range(0,lon_len) :
      for j in range(0, lat_len) :
        for year in range(year_start, year_end +1) :
          for time in range(0,time_len-1) :
            x1_wk[(time_len-1)*(year-year_start)+time]=x2[(time_len-1)*(year-year_start)+time,j,i] - x2c[time,j,i] 
            x1d_wk[(time_len-1)*(year-year_start)+time]=x2[(time_len-1)*(year-year_start)+time+1,j,i] - x2[(time_len-1)*(year-year_start)+time,j,i] \
              - (x2c[time+1,j,i] - x2c[time,j,i])
        for idx, year in enumerate(boot_years):
          for time in range(0,time_len-1) :
            x2_wk[(time_len-1)*idx+time]=x1[(time_len-1)*(year-year_start)+time,j,i] - x1c[time,j,i]
            x2d_wk[(time_len-1)*idx+time]=x1[(time_len-1)*(year-year_start)+time+1,j,i] - x1c[time+1,j,i] \
                                         -(x1[(time_len-1)*(year-year_start)+time,j,i] - x1c[time,j,i])

        dt=1.0

        #T21[j,i], tau21l[j,i], tau21b[j,i], err99, err95, err90 = Liang_funcs.IF_np(x1_wk, x2_wk, x1d_wk, x2d_wk, dt)
        T12[j,i], tau12l_wk[j,i], tau12b[j,i], err99, err95, err90 = Liang_funcs.IF_np(x2_wk, x1_wk, x2d_wk, x1d_wk, dt)      

      
        if tau12l[j,i] > tau12l_wk[j,i] : 
          sig[j,i] = sig[j,i]+0.0002
          #sig[j,i] = sig[j,i]+0.01
      
      #print("tau12:",j,i,tau12l[j,i])

  np.save(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy", sig)
  
