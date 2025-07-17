# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:00:33 2025

@author: jcrompto

This is largely the same script as MLR_SEC, but the data area masked over the 
entire watershed catchment to allow the accumulation area to grow for 
calculating the sensitivity to volumetric mass balance. The beta values neede
to be computed first on the glacier mask in the MLR_SEC script

all functions are at the bottom of the script

"""

import os
import sys
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta, time

import pyproj
from pyproj import Transformer
import netCDF4
import rasterio as rio
import geoutils as gu
from shapely.geometry import Polygon, Point

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d


# %% read in glacier margin coordinates and elevation as well as ice thickness from radar survery
data = pd.read_csv('data_frame_all.csv')
data = data[data['type'] == 'ground']

x = np.round(data['east']).to_numpy()
y = np.round(data['north']).to_numpy()
labels  = data['label']
xBounds = x[labels=='bd']
yBounds = y[labels=='bd']
z = data['ice_thickness'].to_numpy() + 0
dat1  = data[['east','north', 'ice_thickness']].to_numpy()
# min_x = x.min()
# max_x = x.max()
# min_y = y.min()
# max_y = y.max()

min_x = 500395  # here are the limits of the watershed mask
max_x = 501600
min_y = 5533200
max_y = 5534895

bounds = [min_x, min_y, max_x, max_y]

# %% load the mask created from 2025 surface from polygon shapefile generated in Q
filename = 'marg_25.tif'
# rast_dem = gu.Raster(filename,downsample=10)
rast_dem_marg = gu.Raster(filename)
rast_dem_marg.crop([min_x, min_y, max_x, max_y])
rast_dat_marg = rast_dem_marg.data

plt.close('all')
fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_dat_marg)
# %% load the mask created from 2025 surface from polygon shapefile generated in Q
filename = 'helm_watershed_mask.tif'
# rast_dem = gu.Raster(filename,downsample=10)
rast_dem_ws = gu.Raster(filename)
rast_dem_ws.crop([min_x, min_y, max_x, max_y])
rast_dat_ws = rast_dem_ws.data

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_dat_ws)
# %% Compute a slope and aspect from 10m downsampled DEMs. grid cells are only
# considered for years prior to 2025 from within 2025 glacier margin

filename = 'DEM_2020.tif'
rast_dem20 = gu.Raster(filename)
rast_dem20.crop([min_x, min_y, max_x, max_y])
rast_dat20 = rast_dem20.data
rast_dat20_mask = rast_dat20*rast_dat_ws
rast_dat20_mask[rast_dat20_mask==0]=np.nan
slope20, aspect20 =  slope_aspect(rast_dat20_mask)
slim_mask = ~np.isnan(aspect20) # no slope is computed for boundary cells
z20 = rast_dat20_mask*slim_mask
s20_r = slope20.ravel(); a20_r = aspect20.ravel(); z20_r = z20.ravel()
ravel_mask = ~np.isnan(a20_r)
s20_rn = s20_r[ravel_mask]; a20_rn = a20_r[ravel_mask]; z20_rn = z20_r[ravel_mask];

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(slim_mask)

filename = 'DEM_2021.tif'
rast_dem21 = gu.Raster(filename)
rast_dem21.crop([min_x, min_y, max_x, max_y])
rast_dat21 = rast_dem21.data
rast_dat21_mask = rast_dat21*rast_dat_ws
rast_dat21_mask[rast_dat21_mask==0]=np.nan
slope21, aspect21 =  slope_aspect(rast_dat21_mask)
slim_mask = ~np.isnan(aspect21)
z21 = rast_dat21_mask*slim_mask
s21_r = slope21.ravel(); a21_r = aspect21.ravel(); z21_r = z21.ravel()
s21_rn = s21_r[ravel_mask]; a21_rn = a21_r[ravel_mask]; z21_rn = z21_r[ravel_mask];


filename = 'DEM_2022.tif'
rast_dem22 = gu.Raster(filename)
rast_dem22.crop([min_x, min_y, max_x, max_y])
rast_dat22 = rast_dem22.data
rast_dat22_mask = rast_dat22*rast_dat_ws
rast_dat22_mask[rast_dat22_mask==0]=np.nan
slope22, aspect22 =  slope_aspect(rast_dat22_mask)
slim_mask = ~np.isnan(aspect22)
z22 = rast_dat22_mask*slim_mask
s22_r = slope22.ravel(); a22_r = aspect22.ravel(); z22_r = z22.ravel()
s22_rn = s22_r[ravel_mask]; a22_rn = a22_r[ravel_mask]; z22_rn = z22_r[ravel_mask];

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(z22,cmap = 'jet')

filename = 'DEM_2023.tif'
rast_dem23 = gu.Raster(filename)
rast_dem23.crop([min_x, min_y, max_x, max_y])
rast_dat23 = rast_dem23.data
rast_dat23_mask = rast_dat23*rast_dat_ws
rast_dat23_mask[rast_dat23_mask==0]=np.nan
slope23, aspect23 =  slope_aspect(rast_dat23_mask)
slim_mask = ~np.isnan(aspect23)
z23 = rast_dat23_mask*slim_mask
s23_r = slope23.ravel(); a23_r = aspect23.ravel(); z23_r = z23.ravel()
s23_rn = s23_r[ravel_mask]; a23_rn = a23_r[ravel_mask]; z23_rn = z23_r[ravel_mask];

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(z23,cmap = 'jet')

filename = 'DEM_2024.tif'
rast_dem24 = gu.Raster(filename)
rast_dem24.crop([min_x, min_y, max_x, max_y])
rast_dat24 = rast_dem24.data
rast_dat24_mask = rast_dat24*rast_dat_ws
rast_dat24_mask[rast_dat24_mask==0]=np.nan
slope24, aspect24 =  slope_aspect(rast_dat24_mask)
slim_mask = ~np.isnan(aspect24)
z24 = rast_dat24_mask*slim_mask
s24_r = slope24.ravel(); a24_r = aspect24.ravel(); z24_r = z24.ravel()
s24_rn = s24_r[ravel_mask]; a24_rn = a24_r[ravel_mask]; z24_rn = z24_r[ravel_mask];

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(z24,cmap = 'jet')

# %% load this ice thickness data
filename = 'helm_H_IPR.tif'
rast_HR = gu.Raster(filename)
rast_HR.crop([min_x, min_y, max_x, max_y])
rast_H = rast_HR.data*rast_dat_marg.data

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_H)

# %% read in the masked yearly shortwave radiation fields
plt.close('all')
filename = '2021_summer_insol_mean.tif'
rast_SINM_21 = gu.Raster(filename)
rast_SINM_21.crop([min_x, min_y, max_x, max_y])
SINM_21_dat = rast_SINM_21.data
fig,ax = plt.subplots(figsize=(18,18))
SINM_21_dat_field = SINM_21_dat*slim_mask
art = ax.imshow(slim_mask,cmap = 'jet',vmin=0, vmax =270)
cbar = plt.colorbar(art)
SINM_21 = SINM_21_dat_field.ravel()
SINM_21_rn = SINM_21[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(SINM_21_dat_field)
cbar = fig.colorbar(art, ax=ax)
plt.title('summer masked shortwave 2021')

filename = '2022_summer_insol_mean.tif'
rast_SINM_22 = gu.Raster(filename)
rast_SINM_22.crop([min_x, min_y, max_x, max_y])
SINM_22_dat = rast_SINM_22.data
SINM_22_dat_field = SINM_22_dat*slim_mask
SINM_22 = SINM_22_dat_field.ravel()
SINM_22_rn = SINM_22[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(SINM_22_dat_field)
cbar = fig.colorbar(art, ax=ax)
plt.title('summer masked shortwave 2022')

filename = '2023_summer_insol_mean.tif'
rast_SINM_23 = gu.Raster(filename)
rast_SINM_23.crop([min_x, min_y, max_x, max_y])
SINM_23_dat = rast_SINM_23.data
SINM_23_dat_field = SINM_23_dat*slim_mask
SINM_23 = SINM_23_dat_field.ravel()
SINM_23_rn = SINM_23[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(SINM_23_dat_field)
cbar = fig.colorbar(art, ax=ax)
plt.title('summer masked shortwave 2023')

filename = '2024_summer_insol_mean.tif'
rast_SINM_24 = gu.Raster(filename)
rast_SINM_24.crop([min_x, min_y, max_x, max_y])
SINM_24_dat = rast_SINM_24.data
SINM_24_dat_field = SINM_24_dat*slim_mask
SINM_24 = SINM_24_dat_field.ravel()
SINM_24_rn = SINM_24[ravel_mask]


fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(SINM_24_dat_field)
cbar = fig.colorbar(art, ax=ax)
plt.title('summer masked shortwave 2024')
# %% read in snow depth fields
plt.close('all')

file_path = 'Helm_snow_depth.csv'
df = pd.read_csv(file_path)
yr = df['year']
zs = df['z']
sd = df['depth_m']

sd21 = sd[yr==2021]
pz21 = zs[yr==2021]
sd22 = sd[yr==2022]
pz22 = zs[yr==2022]
sd23 = sd[yr==2023]
pz23 = zs[yr==2023]
sd24 = sd[yr==2024]
pz24 = zs[yr==2024]

vmn = 0
vmx = 7
degp = 1 # degree of polynomial to interploate snow depth field

x = pz21; y = sd21
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
sd21_field = poly(z21) #z2X is the elevation field for the DEM of the corresponding year
sd21_r = sd21_field.ravel()
sd21_rn = sd21_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(sd21_field)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('snow depth field 2021')

x = pz22; y = sd22
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
sd22_field = poly(z22)
sd22_r = sd22_field.ravel()
sd22_rn = sd22_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(sd22_field)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('snow depth field 2022')

x = pz23; y = sd23
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
sd23_field = poly(z23)
sd23_r = sd23_field.ravel()
sd23_rn = sd23_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(sd23_field)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('snow depth field 2023')

x = pz24; y = sd24
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
sd24_field = poly(z24)
sd24_r = sd24_field.ravel()
sd24_rn = sd24_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art=ax.imshow(sd24_field)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('snow depth field 2024')

# %% import the era5 temp data at 1550m elevation from nearest grid cell
file_path = 'helm_era5_2000_2024.xlsx'
df = pd.read_excel(file_path)
datetimes = df.timestamp
temperature = df.temperature-273.15


# %% loop through some temperature statistics
# yrs= np.arange(2000,2024)

plt.close('all')
mean_sT = []
for y in np.arange(len(yrs)):
    mask = []
    for i in np.arange(len(datetimes)):
        t = datetimes[i]
        mask.append((t.year==yrs[y]) and (4 < t.month < 10))
        # sys.exit()
    temps = temperature[mask]
    mean_sT.append(np.mean(temps))

y = np.array(mean_sT)
yG = lapse_t*(1934-1550)+y
plt.plot(yG)
x = np.arange(len(mean_sT))
coeffs = np.polyfit(x, yG, deg=1)
poly = np.poly1d(coeffs)
delTemp = poly(x)
plt.plot(x,delTemp)

# %% compute the PDD for each year of 2021--2024 with the ravelled elevation surface
# and the ERA5 land temperature between the dates of the lidar acquisition periods

lapse_t = -5.8/1000 #set the desired lapse rate

start_t = datetime(year = 2020, month = 1, day = 1, hour = 0)
end_t = datetime(year = 2024, month = 1, day = 1, hour = 0)

num_days = (end_t - start_t).days   # include end date
day_array = [start_t + timedelta(days=i) for i in range(num_days)]

t_zji = np.zeros((len(z24_rn),4))

ac20 = datetime(year = 2020, month = 9, day = 28) #dates of acquisitions
ac21 = datetime(year = 2021, month = 9, day = 24)
ac22 = datetime(year = 2022, month = 10, day = 13)
ac23 = datetime(year = 2023, month = 9, day = 22)
ac24 = datetime(year = 2024, month = 9, day = 16)
datetime_array = [ac20, ac21, ac22, ac23, ac24]

year_diffs = [] #variables computed in loop are not used in any subsequent code
for i in range(1, len(datetime_array)):
    delta_days = (datetime_array[i] - datetime_array[i - 1]).days
    year_diff = delta_days / 365.25  # approximate year length including leap years
    year_diffs.append(year_diff)

for i in np.arange(4):
    start_ti = datetime_array[i]
    end_ti = datetime_array[i+1]
    els_d = (datetimes>=start_ti)&(datetimes<=end_ti)
    els = np.where(els_d==True)
    t_zj = np.zeros(len(z24_rn)) 
    for j in np.arange(np.shape(els)[1]):
        t_j = temperature[els[0][j]]
        if j==0:
            zr = z21_rn
        elif j==1:
            zr = z22_rn
        elif j==2:
            zr = z23_rn
        elif j==3:
            zr = z24_rn
        lapse_tzj = lapse_t*(zr-1550)+t_j
        lapse_tzj[lapse_tzj<0] = 0
        t_zj = t_zj + lapse_tzj
        
    t_zji[:,i] = t_zj #each row is an elevation of the grid cell and each column is for the year
    

# %% compute the yearly positive degree days from 2014 to 2024 using the 2024 surface
# to create an average PDD to forward model

zMap = rast_dat24_mask
datetime_array = []

ac20 = datetime(year = 2020, month = 9, day = 28) #dates of acquisitions
ac21 = datetime(year = 2021, month = 9, day = 24)
ac22 = datetime(year = 2022, month = 10, day = 13)
ac23 = datetime(year = 2023, month = 9, day = 22)
ac24 = datetime(year = 2024, month = 9, day = 16)
datetime_array = [ac20, ac21, ac22, ac23, ac24]

    
lapse_t = -5.8/1000
t_cube = np.zeros((np.shape(zMap)[0],np.shape(zMap)[1],10))
t_tot = np.zeros_like(zMap)
for i in np.arange(4):
    start_ti = datetime_array[i]
    end_ti = datetime_array[i+1]
    els_d = (datetimes>=start_ti)&(datetimes<=end_ti)
    els = np.where(els_d==True)
    t_ij = np.zeros(np.shape(zMap)) 
    for j in np.arange(np.shape(els)[1]):
        t_j = temperature[els[0][j]]
        lapse_tzj = lapse_t*(zMap-1550)+t_j
        lapse_tzj[lapse_tzj<0] = 0
        t_ij = t_ij + lapse_tzj
        
    t_cube[:,:,i] = t_ij

# %% compute the yearly positive degree days from 2014 to 2024 using the 2024 surface
# to create an average PDD to forward model

zMap = rast_dat24_mask
datetime_array = []
for i in np.arange(11):
    datetime_array.append(datetime(year = 2014+i, month = 9, day = 30))
    
lapse_t = -5.8/1000
t_forward_cube = np.zeros((np.shape(zMap)[0],np.shape(zMap)[1],10))
t_tot = np.zeros_like(zMap)
for i in np.arange(10):
    start_ti = datetime_array[i]
    end_ti = datetime_array[i+1]
    els_d = (datetimes>=start_ti)&(datetimes<=end_ti)
    els = np.where(els_d==True)
    t_ij = np.zeros(np.shape(zMap)) 
    for j in np.arange(np.shape(els)[1]):
        t_j = temperature[els[0][j]]
        lapse_tzj = lapse_t*(zMap-1550)+t_j
        lapse_tzj[lapse_tzj<0] = 0
        t_ij = t_ij + lapse_tzj
        
    t_forward_cube[:,:,i] = t_ij

avePDD_forward = np.nansum(t_forward_cube,2)/10
# %% assign PDD for each year and plot    
plt.close('all')

pdd21 = t_cube[:,:,0]
pdd21[pdd21==0]=np.nan

fig,ax=plt.subplots(figsize=(18,18))
art = ax.imshow(pdd21,cmap='jet')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('PDD')
plt.title('PDD 2021')

pdd22 = t_cube[:,:,1]
pdd22[pdd22==0]=np.nan

fig,ax=plt.subplots(figsize=(18,18))
art = ax.imshow(pdd22,cmap='jet')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('PDD')
plt.title('PDD 2022')

pdd23 = t_cube[:,:,2]
pdd23[pdd23==0]=np.nan

fig,ax=plt.subplots(figsize=(18,18))
art = ax.imshow(pdd23,cmap='jet')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('PDD')
plt.title('PDD 2023')

pdd24 = t_cube[:,:,3]
pdd24[pdd24==0]=np.nan

fig,ax=plt.subplots(figsize=(18,18))
art = ax.imshow(pdd24,cmap='jet')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('PDD')
plt.title('PDD 2024')


# %% compute average fields for radiation mask and snow depth from 2021--2024 to 
# use for forward model

aveSINM_field = (SINM_21_dat_field + SINM_22_dat_field + SINM_23_dat_field + SINM_24_dat_field)/4
aveSND_field = (sd21_field + sd22_field + sd23_field + sd24_field)/4


# %% Mass balance sensitivity, which requires allowing the glacier to grow where the net 
# elevation change from the model is positive. As such, the .tif files need to be masked with 
# the watershed mask rather than the glacier mask used to compute the MLR. In this section 
# the .tif are remasked before carrying out the sensitivity

snow_density= 0.41
ice_density = 0.917
# pdd_ave = pdd(surfc,temp,datetimes)


delArr_T = np.arange(start=-8,stop=1,step=0.2)
delArr_P = np.arange(start=-1,stop=4,step=0.2)

# delArr_T = [-2]
# delArr_P = [1]

ELA = np.zeros((len(delArr_P),len(delArr_T)))
SMB_AC = np.zeros((len(delArr_P),len(delArr_T))) # specific mass balance rate allowing the area to change
VMB_AC = np.zeros((len(delArr_P),len(delArr_T))) # volumetric mass balance rate allowing the area to change

surfc = rast_dat24_mask.data
surf_r = surfc.ravel()
slope_2024,aspect_2024=slope_aspect(surfc)
marg_r = rast_dat_marg.ravel()


for i in np.arange(len(delArr_T)):
    
    temp_offset = temperature+delArr_T[i]
    pdd_spline = pdd_ave_spline(surfc,temp_offset,datetimes)
    pdd_loop = pdd_spline(surfc)

    for j in np.arange(len(delArr_P)):
        
        SND_loop = aveSND_field.data+delArr_P[j]

        mask_arr = (slope_2024!=0).astype('int')
        SM = B0 + BS*slope_2024.data + BA*aspect_2024.data + BPD*pdd_loop+ BSM*aveSINM_field.data + BAC*SND_loop
        SM_r = SM.ravel()
        
        neg_in_ice = np.multiply(SM<0,rast_dat_marg)
        pos_in_ws = np.multiply(SM,(SM>0))
        
        # plt.close('all')
        # fig,ax=plt.subplots(figsize=(18,18))
        melt_on_ice = SM*neg_in_ice
        z_melt_on_ice = surfc*neg_in_ice
        z_acc_in_basin = surfc*(SM>0)
        # ax.imshow(melt_on_ice,cmap = 'autumn')
        winter_0 = get_jet('winter')
        # ax.imshow(pos_in_ws,cmap = winter_0)
        
        zm_rv = z_melt_on_ice.ravel()
        m_rv = melt_on_ice.ravel()
        maskOut_0M = (m_rv!=0)
        zm = zm_rv[maskOut_0M]
        mwe_ice_melt = (m_rv[maskOut_0M])*ice_density
        m_nan = ~np.isnan(mwe_ice_melt)
        mwe_m = mwe_ice_melt[m_nan]
        z_m = zm[m_nan]
        
        za_rv = z_acc_in_basin.ravel()
        a_rv = pos_in_ws.ravel()
        maskOut_0A = (a_rv!=0)
        za = za_rv[maskOut_0A]
        mwe_snow_acc = (a_rv[maskOut_0A])*snow_density
        a_nan = ~np.isnan(mwe_snow_acc)
        mwe_a = mwe_snow_acc[a_nan]
        z_a = za[a_nan]
        
        if len(np.cumsum(mwe_m))>0:
            total_melt = np.cumsum(mwe_m)[-1]*(10**2)
            specific_melt = np.cumsum(mwe_m)[-1]/len(mwe_m)
        else:
            total_melt=0
            specific_melt=0
            
        if len(np.cumsum(mwe_a))>0:
            total_acc = np.cumsum(mwe_a)[-1]*(10**2)
            specific_acc = np.cumsum(mwe_a)[-1]/len(mwe_a)
        else: 
            total_acc=0
            specific_acc=0
        VMB_AC[j,i] = total_melt+total_acc
        SMB_AC[j,i] = specific_melt+specific_acc
        
        # fig,ax=plt.subplots(figsize=(18,18))
        # ax.plot(z_m,mwe_m,'r.',label='melt on glacier')
        # ax.plot(z_a,mwe_a,'b.',label='accumulation in basin')
        # plt.title(f'$\Delta$T=-2$^\circ$C, $\Delta$P=0.41 m w.e.')
        # plt.ylabel('Surface change (m w.e.)')
        # plt.xlabel('Elevation (m)')
        # plt.legend(loc='lower right')
        # plt.text(1820,2.2,f'M.B. = {MB:.2f} m. w.e.')
        # plt.grid()
        # fig.savefig(r'C:\Users\jcrompto\Documents\code\python_scripts\mass_balance\figures\2D_sensitivity.png')
        
        
        




# %% 
plt.close('all')
field = VMB_AC
fig,ax = plt.subplots(figsize=(26,14))
plt.rcParams.update({'font.size': 36})

# compute the slope (sensitivity) of the field
dx, dy = np.gradient(np.flipud(field))
slope_dydx = np.divide(dy,dx)

delArr_T_pc = delArr_T/mean_tem
delArr_P_pc = delArr_P/mean_acc

midpoint = (np.round(len(delArr_T)/2)).astype(int)
x_swath = field[midpoint,:]
C_T=np.divide(np.diff(x_swath),np.diff(delArr_T))
CTm = C_T[midpoint]
C_T_pc=np.divide(np.diff(x_swath),np.diff(delArr_T_pc))
CTpcm = C_T_pc[midpoint]

# plt.plot((delArr_T[:-1] + delArr_T[1:]) / 2,C_T)
# plt.grid()
y_swath = field[:,midpoint]
C_P=np.divide(np.diff(y_swath),np.diff(delArr_P))
CPm = C_P[midpoint]
C_P_pc=np.divide(np.diff(y_swath),np.diff(delArr_P_pc))
CPpcm = C_P_pc[midpoint]

# plt.plot((delArr_P[:-1] + delArr_P[1:]) / 2,C_P)
# plt.grid()

ave_slope = np.mean(slope_dydx[2:-2,2:-2])

norm = TwoSlopeNorm(vmin=-3e6, vcenter=0, vmax=2e6)

extnt_mwe = [delArr_T[0], delArr_T[-1], delArr_P[0]*0.41, delArr_P[-1]*0.41]
extnt_P = [delArr_T[0]/mean_tem*100, delArr_T[-1]/mean_tem*100, (delArr_P[0]*0.41)/mean_acc*100, (delArr_P[-1]*0.41)/mean_acc*100]

ext = extnt_P

art=ax.imshow(np.flipud(field),extent=ext,cmap='RdBu',norm = norm,aspect='auto')

plt.xlabel('$\Delta$ T ($^\circ$C)')
plt.ylabel('$\Delta$ P (m w.e.)')     
# plt.xlabel('$\Delta$ T (% of mean summer temperature)')
# plt.ylabel('$\Delta$ P (% of mean winter accumulation)')   

ax.plot(0,0,'ks',markersize=10)
levels_c = np.array([-3e6,-2.5e6,-2e6,-1.5e6,-1e6,-0.5e6,0,0.5e6,1e6,1.5e6,2e6])
contours = ax.contour((field),extent=ext,levels=levels_c,colors='k',aspect='auto')
ax_pos = ax.get_position()
cax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.03, ax_pos.height])
cbar = plt.colorbar(art, cax=cax)
cbar.set_label('Mass balance (m$^3$ w.e.)')


# plt.title('black line = elevation of top of glacier')
# fig.savefig(r'C:\Users\jcrompto\Documents\code\python_scripts\mass_balance\figures\MB_contour_refS_fraction.png')
# fig.savefig(r'C:\Users\jcrompto\Documents\code\python_scripts\mass_balance\figures\VMB_contour_AC.pdf')
# fig.savefig(r'C:\Users\jcrompto\Documents\code\python_scripts\mass_balance\figures\VMB_contour_AC.svg')




# %% functions 

def slope_aspect(array):
    x, y = np.gradient(array)
    slope = np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    return slope, aspect

def hillshade(slp_dat,aspect,azimuth,angle_altitude):
    slope = np.pi/2. - slp_dat
    azimuth = 360 - azimuth
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    return (255*(shaded + 1)/2)

def compute_r_squared(observed, modeled):

    obs = observed.flatten()
    mod = modeled.flatten()
    
    valid_mask = ~np.isnan(obs) & ~np.isnan(mod)
    obs = obs[valid_mask]
    mod = mod[valid_mask]

    ss_res = np.sum((obs - mod) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def standardize_columns(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)
    X_std = (X - means) / stds
    return X_std, means, stds

def renormalize_beta_sklearn(model, X_mean, X_std, y_mean, y_std):
    beta_std = model.coef_.flatten()
    intercept_std = model.intercept_

    beta_denorm = beta_std * (y_std / X_std)
    intercept_denorm = y_mean - np.sum(beta_denorm * X_mean)
    
    return intercept_denorm, beta_denorm


def pdd_ave_spline(zMap,temp_loop,datetimes):
    
    plt.close('all')
    zMap_r = zMap.ravel()
    nanMsk = ~np.isnan(zMap_r)
    z_vals = np.sort(zMap_r[nanMsk])
    
    datetime_array = []
    for i in np.arange(10):
        datetime_array.append(datetime(year = 2014+i, month = 9, day = 30))
        
    lapse_t = -6/1000
    
    t_Mtx = np.zeros((9,len(z_vals)))
    
    for i in np.arange(9):
        start_ti = datetime_array[i]
        end_ti = datetime_array[i+1]
        els_d = (datetimes>=start_ti)&(datetimes<=end_ti)
        els = np.where(els_d==True)
        t_ij = np.zeros(len(z_vals)) 
        
        for j in np.arange(np.shape(els)[1]):
            t_j = temp_loop[els[0][j]]
            lapse_tzj = lapse_t*(z_vals-1550)+t_j
            lapse_tzj[lapse_tzj<0] = 0
            t_ij = t_ij + lapse_tzj
            t_Mtx[i,:] = t_ij.data
    
    pdd_mean = np.nanmean(t_Mtx,0)
    
    spline = UnivariateSpline(z_vals, pdd_mean, s=1)
    return spline


def get_jet(theMap):
    base_cmap = plt.get_cmap(theMap)
    num_bins = 180
    color_list = base_cmap(np.linspace(0, 1, num_bins))
    color_list[0] = [1, 1, 1, 0] 
    plasma_zero_cmap = mcolors.ListedColormap(color_list)
    return plasma_zero_cmap


