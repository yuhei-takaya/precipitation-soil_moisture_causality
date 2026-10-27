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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

year_start=1997
year_end=2023
time_len = 4 # number of data in each month
lat_len=60  # 3deg
lon_len=120  # 3deg

ibin_sm=25 # bins soil moisture, 0-0.5 (int: 0.02)
ibin_sw=16 # bins of shortwave radiation,

tau_bin_sm = [[] for _ in range(ibin_sm)]
ef_bin_sm  = [[] for _ in range(ibin_sm)]

tau_bin_sm_sw = [[[] for _ in range(ibin_sw)] for _ in range(ibin_sm)]

sm_edges = np.arange(0.0, 0.51, 0.02)
sw_edges = np.arange(0.0, 405.0, 25.0)

# Land-sea mask
lsm = np.zeros((lat_len, lon_len))
infile_lsm = "../Data/ERA5_regridded_3.0deg_7day/lsm.nc"
dsm=xr.open_dataset(infile_lsm)
dsm = dsm.rename({'lon': 'longitude', 'lat': 'latitude'})  
dsm.transpose('latitude', 'longitude')
#print(dsm)
lsm=dsm["__xarray_dataarray_variable__"]

months = np.arange(1,13)
for i, month in enumerate(months) :

  data=np.load(f"Fig_data/LKIF_SM-aPr_mon{month:02}.5000.npz")
  #print(data)
  
  T12    = data['T12']
  tau12l = data['tau12l']

  sig=np.load(f"Fig_data/LKIF_SM-aPr_mon{month:02}_p-value.5000.npy")

  sm=np.load(f"Fig_data/SM_clim_mon{month:02}.npy")

  sshf=np.load(f"Fig_data/SSHF_clim_mon{month:02}.npy")
  slhf=np.load(f"Fig_data/SLHF_clim_mon{month:02}.npy")

  sw=np.load(f"Fig_data/SIS_clim_mon{month:02}.npy")    

  ef = slhf / (slhf + sshf)

  mask = (
        np.isfinite(tau12l)
        & np.isfinite(sm)
        & np.isfinite(ef)
        & (sm >= 0.0)
        & (sm < 0.5)
  )

  sm_valid  = sm[mask]
  sw_valid =  sw[mask]
  tau_valid = tau12l[mask]
  ef_valid  = ef[mask]

  ibin_sm_idx = np.digitize(sm_valid, sm_edges) - 1
  ibin_sw_idx = np.digitize(sw_valid, sw_edges) - 1

  for k, tau, efv in zip(ibin_sm_idx, tau_valid, ef_valid):
    if(tau >= 0.0 ) :
      if 0 <= k < ibin_sm:
        tau_bin_sm[k].append(abs(tau))
        ef_bin_sm[k].append(efv)

  for i, j, tau, efv in zip(ibin_sm_idx, ibin_sw_idx, tau_valid, ef_valid):
    if(tau >= 0.0 ) :
      if 0 <= i < ibin_sm and 0 <= j < ibin_sw:
        tau_bin_sm_sw[i][j].append(abs(tau))
          
# mean, median, p80
tau_max = np.full(ibin_sm, np.nan)
tau_p95 = np.full(ibin_sm, np.nan)
tau_mean = np.full(ibin_sm, np.nan)
tau_median = np.full(ibin_sm, np.nan)
tau_p85 = np.full(ibin_sm, np.nan)
tau_top20_mean = np.full(ibin_sm, np.nan)
ef_top20_mean = np.full(ibin_sm, np.nan)

for i in range(ibin_sm):

    if len(tau_bin_sm[i]) > 0:

        x = np.asarray(tau_bin_sm[i])
        tau_max[i]   = np.max(x)*100.0
        tau_p95[i]   = np.percentile(x, 95)*100.0
        tau_mean[i]   = np.mean(x)*100.0
        tau_median[i] = np.median(x)*100.0
        tau_p85[i]    = np.percentile(x, 85)*100.0

for i in range(ibin_sm):

    if len(tau_bin_sm[i]) > 0:

        x = np.asarray(tau_bin_sm[i])
        p80 = np.percentile(x, 80)
        tau_top20_mean[i] = np.mean(x[x >= p80]) * 100.0
        y = np.asarray(ef_bin_sm[i])
        ef_top20_mean[i] = np.mean(y[x >= p80]) * 100.0
        
ef_mean = np.full(ibin_sm, np.nan)
ef_median = np.full(ibin_sm, np.nan)
ef_p25 = np.full(ibin_sm, np.nan)
ef_p75 = np.full(ibin_sm, np.nan)

for i in range(ibin_sm):

    if len(ef_bin_sm[i]) > 0:

        x = np.asarray(ef_bin_sm[i])

        ef_mean[i]   = np.mean(x) * 100.0
        ef_median[i] = np.median(x) * 100.0
        ef_p25[i]    = np.percentile(x, 25) * 100.0
        ef_p75[i]    = np.percentile(x, 75) * 100.0

# --- average in bins

tau_mean = np.full((ibin_sm, ibin_sw), np.nan)
ef_mean  = np.full((ibin_sm, ibin_sw), np.nan)

for i in range(ibin_sm):
    for j in range(ibin_sw):
        if len(tau_bin_sm_sw[i][j]) > 0:
            tau_mean[i, j] = np.mean(tau_bin_sm_sw[i][j])*100.0

        
#--- Plot ---
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10

sm_center = np.arange(0.01, 0.50, 0.02)

fig, ax1 = plt.subplots(figsize=(6,3))

ax1.text(
    0.0, 1.1,
    r"(a) Dependence of normalized LKIF (SM $\to$ aPr) on SM", 
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=12
)

# =========================
# Tau (left axis)
# =========================
line3 = ax1.plot(
    sm_center,
    tau_top20_mean,
    color='black',    
    lw=2,
    label='LKIF top 20% mean'
)[0]

line4 = ax1.plot(
    sm_center,
    tau_median,
    color='gray',    
    linestyle='--',
    lw=2,
    label='LKIF median'
)[0]

band1 = ax1.fill_between(
    sm_center,
    tau_p95,
    tau_p85,
    color='gray',
    alpha=0.15,
    label='LKIF 85-95%'    
)

ax1.set_ylim(0, 35)
ax1.set_ylabel(r'Normalized LKIF (SM $\to$ aPr) [%]')

ax2 = ax1.twinx()

line1 = ax2.plot(
    sm_center,
    ef_top20_mean,
    color='blue',    
    lw=2,
    linestyle='--',    
    alpha=0.5,
    label='EF mean for LKIF top 20%'
)[0]

line2 = ax2.plot(
    sm_center,
    ef_median,
    color='blue',    
    lw=2,
    label='EF median'
)[0]


band2 = ax2.fill_between(
    sm_center,
    ef_p25,
    ef_p75,
    color='blue',
    alpha=0.15,
    label='EF 25–75%'
)


ax2.set_ylim(0, 110)
ax2.set_ylabel('Climatological evaporative fraction [%]',color='blue')

ax1.set_xlabel('Soil moisture [m$^3$/m$^3$]')
ax1.set_xlim(0,0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2,
           labels1+labels2,
           loc='upper left',
           frameon=False,
           fontsize=8)

ax2.tick_params(axis='y', colors='blue')
ax2.spines['right'].set_color('blue')

#########################

# ヒートマップ用のAxesを追加
ax2 = fig.add_axes([0.125, -0.925, 0.775, 0.8])  # [left, bottom, width, height]

ax2.text(
    0.0, 1.1,
    r"(b) Dependence of normalized LKIF (SM $\to$ aPr) on SM and SW", 
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=12
)

bounds = np.arange(0, 17, 2)
norm = mcolors.BoundaryNorm(bounds,ncolors=256)

im = ax2.imshow(
    tau_mean.T,
    cmap="viridis_r",
    norm=norm,
    origin="lower",
    aspect="auto",
    extent=[
        sm_edges[0], sm_edges[-1],
        sw_edges[0], sw_edges[-1]
    ]
)

ax2.set_xlim(0,0.4)

ax2.set_xlabel('Soil moisture [m$^3$/m$^3$]')
ax2.set_ylabel("Surface downward\nshortwave radiation [W/m$^2$]")
#ax2.set_title(r"Normalized LKIF (SM  $\to$ aPr) [%]")

fig.colorbar(im, ax=ax2, orientation='horizontal', ticks=bounds, label=r"Average normalized LKIF (SM $\to$ aPr) [%]",
             pad=0.225, fraction=0.05,shrink=1.0, aspect=30)

#plt.tight_layout()

plt.savefig(f"Figure2_positiveLKIF.pdf", format='pdf', bbox_inches='tight')
plt.close()


