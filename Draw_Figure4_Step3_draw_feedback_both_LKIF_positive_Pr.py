import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as patches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

fig = plt.figure(figsize=(9, 12))  # 4x3で綺麗に収まるようサイズを微調整

# 比率を指定してグリッドを作成(3行4列)

h_ratios = [1, 1, 1, 1] # 各行の高さの比率 (1行目:2行目:3行目 = 1 : 1 : 1 : 1)
w_ratios = [1, 1, 1] # 各列の幅の比率 (1列目:2列目:3列目 = 1 : 1 : 1 : 1)

gs = fig.add_gridspec(4, 3, height_ratios=h_ratios, width_ratios=w_ratios)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#---lsm ---
#lsm = np.zeros((lat_len, lon_len))
infile_lsm = "../Data/ERA5_regridded_3.0deg_7day/lsm.nc"
dsm=xr.open_dataset(infile_lsm)
dsm = dsm.rename({'lon': 'longitude', 'lat': 'latitude'})  
dsm.transpose('latitude', 'longitude')
#print(dsm)
lsm=dsm["__xarray_dataarray_variable__"]
mask_lsm = lsm > 0.5 

# 1行目の配置 (左・中央・右)
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.set_title(r'(a) LKIF SM$\to$Pr (June)', loc='left',fontsize=10,pad=0)

# Read data
month=6
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy")
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax1.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax1.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax1.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax1.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(0, 46, 15)
ax1.set_yticks(lat_ticks)
ax1.tick_params(axis='x', labelsize=8)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(55, 101, 15)
ax1.set_xticks(lon_ticks)
ax1.tick_params(axis='y', labelsize=8)
ax1.yaxis.set_major_formatter(LatitudeFormatter())
    
ax1.set_extent([55, 100, 0, 45], crs=ccrs.PlateCarree())

#######
ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax4.set_title(r'(d) LKIF Pr$\to$SM (June)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax4.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax4.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax4.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax4.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(0, 46, 15)
ax4.set_yticks(lat_ticks)
ax4.tick_params(axis='x', labelsize=8)
ax4.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(55, 101, 15)
ax4.set_xticks(lon_ticks)
ax4.tick_params(axis='y', labelsize=8)
ax4.yaxis.set_major_formatter(LatitudeFormatter())

ax4.set_extent([55, 100, 0, 45], crs=ccrs.PlateCarree())

########

# 1行目の配置 (中央)
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_title(r'(b) LKIF SM$\to$Pr (November)', loc='left',fontsize=10,pad=0)

# Read data
month=11
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy") 
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax2.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax2.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax2.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax2.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-40, 6, 15)
ax2.set_yticks(lat_ticks)
ax2.tick_params(axis='x', labelsize=8)
ax2.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(-75, -29, 15)
ax2.set_xticks(lon_ticks)
ax2.tick_params(axis='y', labelsize=8)
ax2.yaxis.set_major_formatter(LatitudeFormatter())

ax2.set_extent([-75, -30, -40, 5], crs=ccrs.PlateCarree())

#######
ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax5.set_title(r'(e) LKIF Pr$\to$SM (November)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax5.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax5.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax5.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax5.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-40, 6, 15)
ax5.set_yticks(lat_ticks)
ax5.tick_params(axis='x', labelsize=8)
ax5.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(-75, -29, 15)
ax5.set_xticks(lon_ticks)
ax5.tick_params(axis='y', labelsize=8)
ax5.yaxis.set_major_formatter(LatitudeFormatter())

ax5.set_extent([-75, -30, -40, 5], crs=ccrs.PlateCarree())

########

# 1行目の配置 (右)
ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax3.set_title(r'(a) LKIF SM$\to$Pr (November)', loc='left',fontsize=10,pad=0)

# Read data
month=11
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy")
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax3.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax3.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax3.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax3.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-45, 1, 15)
ax3.set_yticks(lat_ticks)
ax3.tick_params(axis='x', labelsize=8)
ax3.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(110, 156, 15)
ax3.set_xticks(lon_ticks)
ax3.tick_params(axis='y', labelsize=8)
ax3.yaxis.set_major_formatter(LatitudeFormatter())
    
ax3.set_extent([110, 155, -45, 0], crs=ccrs.PlateCarree())

#######
ax6 = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
ax6.set_title(r'(f) LKIF Pr$\to$SM (November)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax6.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax6.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax6.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax6.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-45, 1, 15)
ax6.set_yticks(lat_ticks)
ax6.tick_params(axis='x', labelsize=8)
ax6.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(110, 156, 15)
ax6.set_xticks(lon_ticks)
ax6.tick_params(axis='y', labelsize=8)
ax6.yaxis.set_major_formatter(LatitudeFormatter())

ax6.set_extent([110, 155, -45, 0], crs=ccrs.PlateCarree())

########

# 3行目の配置 (左)
ax7 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
ax7.set_title(r'(g) LKIF SM$\to$Pr (March)', loc='left',fontsize=10,pad=0)

# Read data
month=3
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy")
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax7.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax7.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax7.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax7.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(0, 46, 15)
ax7.set_yticks(lat_ticks)
ax7.tick_params(axis='x', labelsize=8)
ax7.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(55, 101, 15)
ax7.set_xticks(lon_ticks)
ax7.tick_params(axis='y', labelsize=8)
ax7.yaxis.set_major_formatter(LatitudeFormatter())
    
ax7.set_extent([55, 100, 0, 45], crs=ccrs.PlateCarree())

#######
ax10 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
ax10.set_title(r'(j) LKIF Pr$\to$SM (March)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax10.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax10.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax10.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax10.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(0, 46, 15)
ax10.set_yticks(lat_ticks)
ax10.tick_params(axis='x', labelsize=8)
ax10.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(55, 101, 15)
ax10.set_xticks(lon_ticks)
ax10.tick_params(axis='y', labelsize=8)
ax10.yaxis.set_major_formatter(LatitudeFormatter())

ax10.set_extent([55, 100, 0, 45], crs=ccrs.PlateCarree())

########

# 3行目の配置 (中央)
ax8 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
ax8.set_title(r'(h) LKIF SM$\to$Pr (January)', loc='left',fontsize=10,pad=0)

# Read data
month=1
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy") 
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax8.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax8.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax8.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax8.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-10, 36, 15)
ax8.set_yticks(lat_ticks)
ax8.tick_params(axis='x', labelsize=8)
ax8.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(15, 61, 15)
ax8.set_xticks(lon_ticks)
ax8.tick_params(axis='y', labelsize=8)
ax8.yaxis.set_major_formatter(LatitudeFormatter())

ax8.set_extent([15, 60, -10, 35], crs=ccrs.PlateCarree())

#######
ax11 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())
ax11.set_title(r'(k) LKIF Pr$\to$SM (January)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax11.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax11.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax11.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax11.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-10, 36, 15)
ax11.set_yticks(lat_ticks)
ax11.tick_params(axis='x', labelsize=8)
ax11.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(15, 61, 15)
ax11.set_xticks(lon_ticks)
ax11.tick_params(axis='y', labelsize=8)
ax11.yaxis.set_major_formatter(LatitudeFormatter())

ax11.set_extent([15, 60, -10, 35], crs=ccrs.PlateCarree())
########

# 3行目の配置 (右)
ax9 = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
ax9.set_title(r'(i) LKIF SM$\to$Pr (March)', loc='left',fontsize=10,pad=0)

# Read data
month=3
data_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}.5000.npz")  
T12_sm_pr    = data_sm_pr['T12']
tau12l_sm_pr = data_sm_pr['tau12l']
sig_sm_pr=np.load(f"Fig_data/LKIF_SM-Pr_mon{month:02}_p-value.5000.npy")
mask_sm_pr = ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & (~np.isnan(tau12l_sm_pr)) & mask_lsm

data_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}.5000.npz")  
T12_pr_sm    = data_pr_sm['T12']
tau12l_pr_sm = data_pr_sm['tau12l']
sig_pr_sm=np.load(f"Fig_data/LKIF_Pr-SM_mon{month:02}_p-value.5000.npy") 
mask_pr_sm = ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) &  (~np.isnan(tau12l_pr_sm)) & mask_lsm

mask_both =  ((sig_sm_pr > 0.975) | (sig_sm_pr < 0.025)) & ((sig_pr_sm > 0.975) | (sig_pr_sm < 0.025)) & (tau12l_sm_pr > 0.0) &  ( tau12l_pr_sm >0.0) & mask_lsm
#---

lat2 = np.linspace(-90, 90, tau12l_pr_sm.shape[0])
lon2 = np.linspace(0, 360, tau12l_pr_sm.shape[1])

lon2d, lat2d = np.meshgrid(lon2, lat2)

#bounds = np.arange(0, 35.1, 2.5)
bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

#tau12l_sm_pr_masked = np.clip( np.where(mask_sm_pr, tau12l_sm_pr, np.nan), 0, None)
tau12l_sm_pr_masked = np.where(mask_sm_pr, tau12l_sm_pr, np.nan)
pcm = ax9.pcolormesh(lon2d, lat2d, tau12l_sm_pr_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax9.scatter(
lon2d[mask_sm_pr],
lat2d[mask_sm_pr],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax9.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)

ax9.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-10, 36, 15)
ax9.set_yticks(lat_ticks)
ax9.tick_params(axis='x', labelsize=8)
ax9.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(15, 61, 15)
ax9.set_xticks(lon_ticks)
ax9.tick_params(axis='y', labelsize=8)
ax9.yaxis.set_major_formatter(LatitudeFormatter())

ax9.set_extent([-120, -75, 0, 45], crs=ccrs.PlateCarree())

#######
ax12 = fig.add_subplot(gs[3, 2], projection=ccrs.PlateCarree())
ax12.set_title(r'(l) LKIF Pr$\to$SM (March)', loc='left',fontsize=10,pad=0)

bounds = np.arange(-25, 30, 5)
norm = BoundaryNorm(bounds, ncolors=256)

tau12l_pr_sm_masked = np.where(mask_pr_sm, tau12l_pr_sm, np.nan)
pcm = ax12.pcolormesh(lon2d, lat2d, tau12l_pr_sm_masked*100.0, cmap='coolwarm', \
                          norm=norm, shading='auto', transform=ccrs.PlateCarree())

ax12.scatter(
lon2d[mask_pr_sm],
lat2d[mask_pr_sm],
s=5.0,
c='k',
marker='o',
linewidths=0,
transform=ccrs.PlateCarree(),
zorder=10
)

ax12.scatter(
lon2d[mask_both],
lat2d[mask_both],
s=25.0,
facecolors='none',
edgecolors='k',
marker='o',
linewidths=0.75,
transform=ccrs.PlateCarree(),
zorder=9
)
    
ax12.coastlines(resolution='50m')

# 緯度
lat_ticks = np.arange(-10, 36, 15)
ax12.set_yticks(lat_ticks)
ax12.tick_params(axis='x', labelsize=8)
ax12.xaxis.set_major_formatter(LongitudeFormatter())
# 経度
lon_ticks = np.arange(15, 61, 15)
ax12.set_xticks(lon_ticks)
ax12.tick_params(axis='y', labelsize=8)
ax12.yaxis.set_major_formatter(LatitudeFormatter())

ax12.set_extent([-120, -75, 0, 45], crs=ccrs.PlateCarree())
########


plt.savefig(f"Figure4new_LKIF_positive_both_way_Pr.pdf", format='pdf', bbox_inches='tight')
plt.close()
