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
import pandas as pd

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

#Moon and Ha (2020)
# https://www.nature.com/articles/s41612-020-00151-w

regions = [
    {"name":"North African monsoon", "west":-20, "east":30,  "south": 0,  "north":20},   # North Africa
    {"name":"South African monsoon", "west": 15, "east":40,  "south":-25, "north":-5},   # South Africa
    {"name":"South Asian monsoon", "west": 65, "east":90,  "south": 5,  "north":30},   # Indian
    {"name":"Southeast Asian monsoon", "west": 95, "east":110, "south": 5,  "north":25},   # Indo-China
    {"name":"East Asian monsoon",  "west":110, "east":140, "south":20,  "north":40},   # East Asia
    {"name":"Australian monsoon",  "west":120, "east":150, "south":-20, "north":-5},   # Australia
    {"name":"North American monsoon", "west":-115,"east":-95, "south":15,  "north":35},   # N. America
    {"name":"South American monsoon", "west":-70, "east":-45, "south":-25, "north":-5},   # S. America
    {"name":"All", "west":-180, "east":180, "south":-30, "north":30},   # All
]


# Land-sea mask
lsm = np.zeros((lat_len, lon_len))
infile_lsm = "../Data/ERA5_regridded_3.0deg_7day/lsm.nc"
dsm=xr.open_dataset(infile_lsm)
dsm = dsm.rename({'lon': 'longitude', 'lat': 'latitude'})  
dsm.transpose('latitude', 'longitude')
#print(dsm)
lsm=dsm["__xarray_dataarray_variable__"]

lat = lsm.latitude.values
lon = lsm.longitude.values

# convert longitude to [-180,180]
lon2 = ((lon + 180) % 360) - 180


# area weight
weights_lat = np.cos(np.deg2rad(lat))
weights = weights_lat[:, None] * np.ones((len(lat), len(lon)))

# land mask
landmask = lsm.values > 0.5

all_results = []

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
for i, month in enumerate(months) :

  data=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")
  #print(data)
  
  T12    = data['T12']
  tau12l = data['tau12l']

  #print(tau12l.shape)
  #print(len(lat), len(lon))
  
  sig=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")

  sigmask =  (sig > 0.975) | (sig < 0.025)

  sm = np.load(f"Fig_data/SM_clim_mon{month:02}.npy")
  sw = np.load(f"Fig_data/SIS_clim_mon{month:02}.npy")  

  for reg in regions:

    # region mask
    regmask = (
        (lat[:, None] >= reg["south"]) &
        (lat[:, None] <= reg["north"]) &
        (lon2[None, :] >= reg["west"]) &
        (lon2[None, :] <= reg["east"])
    )

    mask = regmask & landmask

    if np.sum(mask) == 0:
        continue

    # -----------------------------
    # Area-weighted regional average
    # -----------------------------
    w = weights[mask]

    valid = mask & np.isfinite(tau12l) #& (tau12l >= 0.0 )

    if np.any(valid):
        region_ave = (
            np.sum(np.abs(tau12l[valid]) * weights[valid])
            / np.sum(weights[valid])
        )
    else:
        region_ave = np.nan

    # --- SM average ---
    
    if np.any(valid):
        region_ave_sm = (
            np.sum(np.abs(sm[valid]) * weights[valid])
            / np.sum(weights[valid])
        )
    else:
        region_ave_sm = np.nan

    # --- SW average ---
    
    if np.any(valid):
        region_ave_sw = (
            np.sum(np.abs(sw[valid]) * weights[valid])
            / np.sum(weights[valid])
        )
    else:
        region_ave_sw = np.nan        
        
    # -----------------------------
    # Top 10% average
    # -----------------------------

    if np.any(valid):
      # 領域内の有効データを1次元化
      values = np.abs(tau12l[valid])
      w = weights[valid]

      # 90パーセンタイル（上位10%の閾値）
      threshold = np.percentile(values, 90)

      # 上位10%のみ抽出
      top = values >= threshold
        
      # 面積重み付き平均
      region_top10_ave = (
          np.sum(values[top] * w[top])
          / np.sum(w[top])
      )
    else:
      region_top10_ave = np.nan
    
    # -----------------------------
    # Significant area ratio
    # -----------------------------
    valid = mask & np.isfinite(tau12l) 
    sig_region = valid & sigmask
    
    area_total = np.sum(weights[valid])
    area_sig   = np.sum(weights[sig_region])

    if area_total > 0:    
      sig_area_ratio = area_sig / area_total
    else:
      sig_area_ratio = np.nan
      
    # -----------------------------
    # Significant area ratio (tau12l >= 0)
    # -----------------------------
    mask_pos = tau12l >= 0.0
    sig_pos_region = sig_region & mask_pos
    area_sig_pos = np.sum(weights[sig_pos_region])

    if area_total > 0:
      sig_area_ratio_pos = area_sig_pos / area_total
    else:
      sig_area_ratio_pos = np.nan

    # ----
    
    all_results.append([
        month,
        reg["name"],
        region_ave*100.0,
        region_ave_sm,
        region_ave_sw,                
        region_top10_ave*100.0,
        sig_area_ratio,
        sig_area_ratio_pos        
    ])

# ==========================================
# Output
# ==========================================
df = pd.DataFrame(
    all_results,
    columns=[
        "month",
        "region",
        "region_ave",
        "region_ave_sm",
        "region_ave_sw",        
        "region_top10_ave",
        "sig_area_ratio",
        "sig_area_ratio_pos"
    ]
)

print(df)

df.to_csv(
    "regional_statistics.csv",
    index=False
)

#================================
# Plot

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

regions = [
    "South Asian monsoon",
    "Southeast Asian monsoon",
    "East Asian monsoon",
    "Australian monsoon",
    "North American monsoon",
    "South American monsoon",
    "North African monsoon",
    "South African monsoon",    
]

fig, axes = plt.subplots(
    4, 2,
    figsize=(8.5, 10),
    sharex=True
)

axes = axes.flatten()

# 共通スケール
ymax_ave = 17.5 #np.nanmax(df["region_top10_ave"]) * 1.1
ymax_area = 60.0 #np.nanmax(df["sig_area_ratio"]) * 100 * 1.1

for i, region in enumerate(regions):

    ax = axes[i]

    tmp = (
        df[df["region"] == region]
        .sort_values("month")
    )

    print(tmp[[
    "month",
    "sig_area_ratio",
    "sig_area_ratio_pos"
    ]])
    
    # ------------------------
    # 左軸 : region_ave
    # ------------------------
    line1 = ax.plot(
        tmp["month"],
        tmp["region_ave"],
        "--o",
        dashes=[3, 1],
        color="k",
        lw=1.8,
        ms=4,
        mfc="white",   # marker face color
        mec="k",       # marker edge color
        mew=1.2,       # marker edge width
        label="Regional ave",
        zorder=10
    )

    line2 = ax.plot(
        tmp["month"],
        tmp["region_top10_ave"],
        "-o",
        color="red",
        lw=1.8,
        ms=4,
        label="Regional top 10% ave",
        zorder=5
    )

    ax.set_ylim(0, ymax_ave)
    ax.set_title(region)

    #3つ目の軸
    #ax3 = ax.twinx()
    #ax3.spines['right'].set_position(('axes', 1.3)) # 3つ目の軸を右にずらす
    #line3 = ax3.plot(
    #    tmp["month"],
    #    tmp["region_ave_sw"]/100.0,
    #    "-o",
    #    color="green",
    #    lw=1.8,
    #    ms=4,
    #    label="SW regional ave",
    #    zorder=6
    #)        
    #ax3.set_ylabel('Surface shortwave radiation')
    #ax3.set_ylim(1.5, 3.0)
    
    # ------------------------
    # 右軸 : sig_area_ratio
    # ------------------------
    ax2 = ax.twinx()

    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0)
    
#    bars1 = ax2.bar(
#        tmp["month"],
#        tmp["sig_area_ratio"] * 100,
#        width=0.75,
#        color="#7FA9C9", #blue
#        label="Area fraction (p<0.05) [%]",
#        zorder=1
#    )

    bars2 = ax2.bar(
        tmp["month"],
        tmp["sig_area_ratio_pos"] * 100,
        width=0.75,
        color="#7FA9C9", #blue
        zorder=2
    )    

    ax2.set_ylim(0, ymax_area)

    if i % 2 == 0:
        ax.set_ylabel(r"Average normalized LKIF [%] ")
    if i % 2 == 1:
        ax2.set_ylabel("Area fraction (p<0.05) [%]")

    ax.set_xticks(np.arange(1, 13))

    
# 下段のみ x ラベル
for ax in axes[-2:]:
    ax.set_xlabel("Month")

# 共通凡例
fig.legend(
    [line1[0], line2[0], bars2],
    ["Regional average (positive LKIF)",
     "Regional upper 10% average (positive LKIF)",
     r"Area fraction (positive LKIF, p<0.05)"
     ],
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.005),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

#plt.show()

plt.savefig(f"Figure3_positiveLKIF.pdf", format='pdf', bbox_inches='tight')
plt.close()
