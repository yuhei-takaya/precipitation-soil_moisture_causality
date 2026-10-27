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
import matplotlib.colors as mcolors

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

year_start=1997
year_end=2023
time_len = 4 # number of data in each month
lat_len=60  # 3deg
lon_len=120   # 3deg


regions = [
    {"name":"North Africa", "west":-20, "east":30,  "south": 0,  "north":20},   # North Africa
    {"name":"South Africa", "west": 15, "east":40,  "south":-25, "north":-5},   # South Africa
    {"name":"Indian", "west": 65, "east":90,  "south": 5,  "north":30},         # Indian
    {"name":"Southeast", "west": 95, "east":110, "south": 5,  "north":25},      # Indo-China
    {"name":"East Asia", "west":110, "east":140, "south":20,  "north":40},      # East Asia
    {"name":"Australia",  "west":120, "east":150, "south":-20, "north":-5},     # Australia
    {"name":"North America", "west":-115,"east":-95, "south":15,  "north":35},  # N. America
    {"name":"South America", "west":-70, "east":-45, "south":-25, "north":-5},  # S. America
]

# Land-sea mask
lsm = np.zeros((lat_len, lon_len))
infile_lsm = "../Data/ERA5_regridded_3.0deg_7day/lsm.nc"
dsm=xr.open_dataset(infile_lsm)
dsm = dsm.rename({'lon': 'longitude', 'lat': 'latitude'})  
dsm.transpose('latitude', 'longitude')
#print(dsm)
lsm=dsm["__xarray_dataarray_variable__"]

months = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2] 
for i, month in enumerate(months) :

  if i % 3 == 0:
    tau12l_3mon = np.zeros((lat_len, lon_len))
    tau12l_3mon_ave = np.zeros((lat_len, lon_len))    
    sig_3mon = np.zeros((lat_len, lon_len))
    sm_3mon = np.zeros((lat_len, lon_len))
  
  data=np.load(f"Fig_data/LKIF_SM-aPr_mon{month:02}.5000.npz")
  #print(data)
  
  T12    = data['T12']
  tau12l = data['tau12l']

  sig=np.load(f"Fig_data/LKIF_SM-aPr_mon{month:02}_p-value.5000.npy")
  #print(sig)
  
  mask = (np.abs(tau12l) > np.abs(tau12l_3mon)) & (tau12l > 0.0)
  tau12l_3mon[mask] = tau12l[mask]
  sig_3mon[mask] = sig[mask]
  tau12l_3mon_ave += np.clip(tau12l, 0, None) / 3.0
  
  sm = np.load(f"Fig_data/SM_clim_mon{month:02}.npy")
  sm_3mon += sm / 3.0
  
  #--- Plot ---

  if i == 0 :

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    labels = ['(a) Max normalized LKIF (MAM)','(b) Soil moisture climatology (MAM)', \
              '(c) Max normalized LKIF (JJA)','(d) Soil moisture climatology (JJA)', \
              '(e) Max normalized LKIF (SON)','(f) Soil moisture climatology (SON)', \
              '(g) Max normalized LKIF (DJF)','(h) Soil moisture climatology (DJF)']

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,6.75), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.ravel()

    fig.subplots_adjust(
       wspace=0.12,  # 横方向の間隔
       hspace=0.05   # 縦方向の間隔
    )
    
  if i % 3 == 2 :  

    idx = 2*(i // 3)
    
    lat2 = np.linspace(-90, 90, tau12l.shape[0])
    lon2 = np.linspace(0, 360, tau12l.shape[1])

    lon2d, lat2d = np.meshgrid(lon2, lat2)

    bounds = np.arange(0, 35.1, 2.5)
    #bounds = np.arange(0, 30.1, 2.5)
    norm = BoundaryNorm(bounds, ncolors=256)

    tau12l_3mon[np.isnan(tau12l_3mon_ave)] = np.nan
    pcm = axs[idx].pcolormesh(lon2d, lat2d, np.abs(tau12l_3mon)*100.0, cmap='magma_r', \
                      norm=norm, shading='auto', transform=ccrs.PlateCarree())
    
    mask = ( ((sig_3mon > 0.975) | (sig_3mon < 0.025)) & (~np.isnan(tau12l_3mon)) ) & (~np.isnan(tau12l_3mon_ave)) 

    axs[idx].scatter(
    lon2d[mask],
    lat2d[mask],
    s=1.5,
    c='k',
    marker='o',
    linewidths=0,
    transform=ccrs.PlateCarree(),
    zorder=10
    )

    for r  in regions:
      axs[idx].plot(
        [r["west"], r["east"], r["east"], r["west"], r["west"]],
        [r["south"], r["south"], r["north"], r["north"], r["south"]],
        transform=ccrs.PlateCarree(),
        lw=0.7,
        color="blue"
      ) 
    
    # Area fraction of soil moisture influence
    tropics = (lat2d >= -30) & (lat2d <= 30)
    valid = tropics & (~np.isnan(tau12l_3mon)) & (~np.isnan(tau12l_3mon_ave))
    sigmask = tropics & mask
    weights = np.cos(np.deg2rad(lat2d))
    area_fraction = (
    np.sum(weights[sigmask])
    / np.sum(weights[valid])
    )
    print(f"Area fraction (%) = {area_fraction*100:.2f}")
    
    axs[idx].coastlines(resolution='50m')

    axs[idx].set_extent([-120, 165, -45, 45], crs=ccrs.PlateCarree())

    # --- 線なし・tick付き ---

    # 緯度
    lat_ticks = np.arange(-30, 31, 30)
    axs[idx].set_yticks(lat_ticks)
    axs[idx].tick_params(axis='x', labelsize=8)
    axs[idx].xaxis.set_major_formatter(LongitudeFormatter())
    # 経度
    lon_ticks = np.arange(-120, 166, 60)
    axs[idx].set_xticks(lon_ticks)
    axs[idx].tick_params(axis='y', labelsize=8)
    axs[idx].yaxis.set_major_formatter(LatitudeFormatter())
    
    gl = axs[idx].gridlines(draw_labels=False, linewidth=0)

    axs[idx].text(
    0.0, 1.15,
    labels[idx],
    transform=axs[idx].transAxes,
    fontsize=10,
    fontweight='normal',
    va='top',
    ha='left'
    )

    if idx==6 :
      # [left, bottom, width, height] （Figure座標: 0〜1）
      cax = fig.add_axes([0.133, 0.08, 0.35, 0.015])
      cbar = fig.colorbar(pcm,cax=cax,orientation='horizontal')
      cbar.set_label(r'Normalized LKIF (SM $\rightarrow$ aPr) [%]', fontsize=8)
      cbar.ax.tick_params(labelsize=8)
    
    #--- SM

    idx = 2*(i // 3) + 1
    
    lat2 = np.linspace(-90, 90, tau12l.shape[0])
    lon2 = np.linspace(0, 360, tau12l.shape[1])

    lon2d, lat2d = np.meshgrid(lon2, lat2)


    base = plt.get_cmap('gist_earth_r')

    # カラーマップの最初の5%を捨てる
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'gist_earth_r_trim',
        base(np.linspace(0.02, 1.0, 256))
    )

    #bounds = np.arange(0.0, 0.37, 0.02)
    bounds = np.arange(0.0, 0.43, 0.02) 
    norm = BoundaryNorm(bounds, ncolors=256)

    pcm = axs[idx].pcolormesh(lon2d, lat2d, sm_3mon, cmap=new_cmap, \
                      norm=norm, shading='auto', transform=ccrs.PlateCarree())
    
    axs[idx].coastlines(resolution='50m')

    axs[idx].set_extent([-120, 165, -45, 45], crs=ccrs.PlateCarree())

    # --- 線なし・tick付き ---

    # 緯度
    lat_ticks = np.arange(-30, 31, 30)
    axs[idx].set_yticks(lat_ticks)
    axs[idx].tick_params(axis='x', labelsize=8)
    axs[idx].xaxis.set_major_formatter(LongitudeFormatter())
    # 経度
    lon_ticks = np.arange(-120, 166, 60)
    axs[idx].set_xticks(lon_ticks)
    axs[idx].tick_params(axis='y', labelsize=8)
    axs[idx].yaxis.set_major_formatter(LatitudeFormatter())
    
    gl = axs[idx].gridlines(draw_labels=False, linewidth=0)

    axs[idx].text(
    0.0, 1.15,
    labels[idx],
    transform=axs[idx].transAxes,
    fontsize=10,
    fontweight='normal',
    va='top',
    ha='left'
    )

    if idx==7 :
      # [left, bottom, width, height] （Figure座標: 0〜1）
      cax = fig.add_axes([0.544, 0.08, 0.35, 0.015])
      cbar = fig.colorbar(pcm,cax=cax,orientation='horizontal')
      cbar.set_label(r'SM climatology [m$^3$/m$^3$]', fontsize=8)
      cbar.ax.tick_params(labelsize=8)
    
#cbar = fig.colorbar(pcm, ax=axs, orientation='horizontal',
#                    fraction=0.02, pad=0.04)
#cbar.set_label('Normalized LKIF (SM -> aPr) [%]', fontsize=8)
#cbar.ax.tick_params(labelsize=8)

plt.savefig(f"Figure1_positiveLKIF.pdf", format='pdf', bbox_inches='tight')
plt.close()
