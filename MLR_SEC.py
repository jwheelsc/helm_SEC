# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:00:33 2025

@author: jcrompto

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
min_x = x.min()
max_x = x.max()
min_y = y.min()
max_y = y.max()

bounds = [min_x, min_y, max_x, max_y]

# %% load the mask created from 2025 surface from polygon shapefile generated in Q
filename = 'marg_25.tif'
# rast_dem = gu.Raster(filename,downsample=10)
rast_dem_marg = gu.Raster(filename)
rast_dem_marg.crop([min_x, min_y, max_x, max_y])
rast_dat_marg = rast_dem_marg.data

# %% Compute a slope and aspect from 10m downsampled DEMs. grid cells are only
# considered for years prior to 2025 from within 2025 glacier margin

filename = 'DEM_2020.tif'
rast_dem20 = gu.Raster(filename)
rast_dem20.crop([min_x, min_y, max_x, max_y])
rast_dat20 = rast_dem20.data
rast_dat20_mask = rast_dat20*rast_dat_marg
rast_dat20_mask[rast_dat20_mask==0]=np.nan
slope20, aspect20 =  slope_aspect(rast_dat20_mask)
slim_mask = ~np.isnan(aspect20) # no slope is computed for boundary cells
z20 = rast_dat20_mask*slim_mask
s20_r = slope20.ravel(); a20_r = aspect20.ravel(); z20_r = z20.ravel()
ravel_mask = ~np.isnan(a20_r)
s20_rn = s20_r[ravel_mask]; a20_rn = a20_r[ravel_mask]; z20_rn = z20_r[ravel_mask];

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(slope20,cmap = 'jet')

filename = 'DEM_2021.tif'
rast_dem21 = gu.Raster(filename)
rast_dem21.crop([min_x, min_y, max_x, max_y])
rast_dat21 = rast_dem21.data
rast_dat21_mask = rast_dat21*rast_dat_marg
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
rast_dat22_mask = rast_dat22*rast_dat_marg
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
rast_dat23_mask = rast_dat23*rast_dat_marg
rast_dat23_mask[rast_dat23_mask==0]=np.nan
slope23, aspect23 =  slope_aspect(rast_dat23_mask)
slim_mask = ~np.isnan(aspect23)
z23 = rast_dat23_mask*slim_mask
s23_r = slope23.ravel(); a23_r = aspect23.ravel(); z23_r = z23.ravel()
s23_rn = s23_r[ravel_mask]; a23_rn = a23_r[ravel_mask]; z23_rn = z23_r[ravel_mask];

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(z23,cmap = 'jet')

filename = 'Helm/DEM_2024.tif'
rast_dem24 = gu.Raster(filename)
rast_dem24.crop([min_x, min_y, max_x, max_y])
rast_dat24 = rast_dem24.data
rast_dat24_mask = rast_dat24*rast_dat_marg
rast_dat24_mask[rast_dat24_mask==0]=np.nan
slope24, aspect24 =  slope_aspect(rast_dat24_mask)
slim_mask = ~np.isnan(aspect24)
z24 = rast_dat24_mask*slim_mask
s24_r = slope24.ravel(); a24_r = aspect24.ravel(); z24_r = z24.ravel()
s24_rn = s24_r[ravel_mask]; a24_rn = a24_r[ravel_mask]; z24_rn = z24_r[ravel_mask];


# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(z24,cmap = 'jet')


# %% import the dh maps 

filename = 'dh_20_21.tif'
rast_dh2021 = gu.Raster(filename)
rast_dh2021.crop([min_x, min_y, max_x, max_y])
rast_dh2021 = rast_dh2021.data
rast_dh2021_mask = rast_dh2021*slim_mask
dh2021_r = rast_dh2021_mask.ravel()
dh2021_rn = dh2021_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art = ax.imshow(rast_dh2021_mask,cmap = 'jet_r')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('dh 21/20')

filename = 'dh_21_22.tif'
rast_dh2122 = gu.Raster(filename)
rast_dh2122.crop([min_x, min_y, max_x, max_y])
rast_dh2122 = rast_dh2122.data
rast_dh2122_mask = rast_dh2122*slim_mask
dh2122_r = rast_dh2122_mask.ravel()
dh2122_rn = dh2122_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_dh2122_mask,cmap=cbar.cmap, vmin = cbar.vmin, vmax = cbar.vmax)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('dh 22/21')

filename = 'dh_22_23.tif'
rast_dh2223 = gu.Raster(filename)
rast_dh2223.crop([min_x, min_y, max_x, max_y])
rast_dh2223 = rast_dh2223.data
rast_dh2223_mask = rast_dh2223*slim_mask
dh2223_r = rast_dh2223_mask.ravel()
dh2223_rn = dh2223_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_dh2223_mask,cmap=cbar.cmap, vmin = cbar.vmin, vmax = cbar.vmax)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('dh 23/22')

filename = 'dh_23_24.tif'
rast_dh2324 = gu.Raster(filename)
rast_dh2324.crop([min_x, min_y, max_x, max_y])
rast_dh2324 = rast_dh2324.data
rast_dh2324_mask = rast_dh2324*slim_mask
dh2324_r = rast_dh2324_mask.ravel()
dh2324_rn = dh2324_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
ax.imshow(rast_dh2324_mask,cmap=cbar.cmap, vmin = cbar.vmin, vmax = cbar.vmax)
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('dh 24/23')

filename = 'dh_24_25_w.tif' # this is the winter field
rast_dh2425 = gu.Raster(filename)
rast_dh2425.crop([min_x, min_y, max_x, max_y])
rast_dh2425 = rast_dh2425.data
rast_dh2425_mask = rast_dh2425*slim_mask
dh2425_r = rast_dh2425_mask.ravel()
dh2425w_rn = dh2425_r[ravel_mask]

fig,ax = plt.subplots(figsize=(18,18))
art = ax.imshow(rast_dh2425_mask,cmap='jet')
cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.title('dh 24/45_w')

# %% average the geodetic balance

melt_cube = np.zeros((np.shape(rast_dh2324_mask)[0],np.shape(rast_dh2324_mask)[1],4))

melt_cube[:,:,0] = rast_dh2021_mask
melt_cube[:,:,1] = rast_dh2122_mask
melt_cube[:,:,2] = rast_dh2223_mask
melt_cube[:,:,3] = rast_dh2324_mask

ave_melt = (rast_dh2021*rast_dat_marg+rast_dh2122*rast_dat_marg+rast_dh2223*rast_dat_marg+rast_dh2324*rast_dat_marg)/4
# plt.close('all')
# plt.figure()
# plt.imshow(ave_melt)

# %% load this ice thickness data
filename = 'helm_H_IPR.tif'
rast_H = gu.Raster(filename)
rast_H.crop([min_x, min_y, max_x, max_y])
rast_H = rast_H.data
rast_H = rast_H*rast_dat_marg

# %% read in the masked yearly shortwave radiation fields
plt.close('all')
filename = '2021_summer_insol_mean.tif'
rast_SINM_21 = gu.Raster(filename)
rast_SINM_21.crop([min_x, min_y, max_x, max_y])
SINM_21_dat = rast_SINM_21.data
# fig,ax = plt.subplots(figsize=(18,18))
SINM_21_dat_field = SINM_21_dat*slim_mask
# art = ax.imshow(SINM_21_dat_field,cmap = 'jet',vmin=0, vmax =270)
# cbar = plt.colorbar(art)
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

# %% compute the PDD for each year of 2021--2024 with the ravelled elevation surface
# and the ERA5 land temperature between the dates of the lidar acquisition periods

lapse_t = -6.5/1000 #set the desired lapse rate

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

    
lapse_t = -6.5/1000
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
    
lapse_t = -6.5/1000
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

# %% create the design matrix with slope, aspect, PDDs, masked radiation and snow depth

# shortwave ,SWM_rn, snow depth = sd21_rn
X21  = np.vstack((s21_rn,a21_rn,t_zji[:,0],SINM_21_rn,sd21_rn))
X22  = np.vstack((s22_rn,a22_rn,t_zji[:,1],SINM_22_rn,sd22_rn))
X23  = np.vstack((s23_rn,a23_rn,t_zji[:,2],SINM_23_rn,sd23_rn))
X24  = np.vstack((s24_rn,a24_rn,t_zji[:,3],SINM_24_rn,sd24_rn))

X = (np.hstack((X21,X22,X23,X24))).T
Y = (np.hstack((dh2021_rn, dh2122_rn,dh2223_rn,dh2324_rn))).T

model = LinearRegression().fit(X, Y)

B0 = model.intercept_
BS = model.coef_[0]
BA = model.coef_[1]
BPD = model.coef_[2]
BSM = model.coef_[3]
BAC = model.coef_[4]

# %% compute statistical significance of beta values with standardized variables
X_std, X_mean, X_std_vals = standardize_columns(X)
Y_std, y_mean, y_std = standardize_columns(Y.reshape(-1, 1))
Y_std = Y_std.flatten()
X_with_const = sm.add_constant(X_std)  # add a constant to the regression to get significance

model = sm.OLS(Y_std, X_with_const)
results = model.fit()

print(results.summary())

# %% apply coefficients to each year of raster data

SM2021 = B0 + BS*slope21 + BA*aspect21 + BPD*pdd21 + BSM*SINM_21_dat_field + BAC*sd21_field
SM2122 = B0 + BS*slope22 + BA*aspect22 + BPD*pdd22 + BSM*SINM_22_dat_field + BAC*sd22_field
SM2223 = B0 + BS*slope23 + BA*aspect23 + BPD*pdd23 + BSM*SINM_23_dat_field + BAC*sd23_field
SM2324 = B0 + BS*slope24 + BA*aspect24 + BPD*pdd24 + BSM*SINM_24_dat_field + BAC*sd24_field

# coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# %%
plt.close('all')
plt.rcParams.update({'font.size': 32})

fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(38,20))

for i in np.arange(4):
    
    if i == 0:
        z = z21_rn
        dat_r = rast_dh2021_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2021.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = '2020-2021'
    if i == 1:
        z = z22_rn
        dat_r = rast_dh2122_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2122.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = '2021-2022'

    if i == 2:
        z = z23_rn
        dat_r = rast_dh2223_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2223.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = '2022-2023'

    if i == 3:
        z = z24_rn
        dat_r = rast_dh2324_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2324.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = '2023-2024'

        
    r2 = compute_r_squared(dat_rn, sim_rn)
    art = ax[i].plot(z,dat_rn,'k.',label='observations')
    ax[i].set_title(title_lab)

    
    ax[i].plot(z,sim_rn,'r.',label='model',alpha=0.4)
    ax[i].set_title(title_lab)

    ax[i].text(0.05, 0.95, f'R$^2$ = {r2:.2f}', transform=ax[i].transAxes,
                fontsize=28, color='white', bbox=dict(facecolor='black', alpha=0.5))
    ax[i].grid()
    plt.legend(loc='lower left')
    ax[i].set_xlabel('Elevation (m)', fontsize = 34)
    if i == 0:
        ax[i].set_ylabel('Elevation change (m)', fontsize = 34)
    ax[i].set_ylim(-8,4)
    

plt.show()

# %% same as the section above, but plotting in 2d with elevation, BINNED
plt.close('all')
plt.rcParams.update({'font.size': 32})

fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(36,20))


for i in np.arange(4):
    
    if i == 0:
        z = z20_rn
        dat_r = rast_dh2021_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2021.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = 'dh 21/20'
    if i == 1:
        z = z21_rn
        dat_r = rast_dh2122_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2122.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = 'dh 22/21'

    if i == 2:
        z = z22_rn
        dat_r = rast_dh2223_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2223.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = 'dh 23/22'

    if i == 3:
        z = z23_rn
        dat_r = rast_dh2324_mask.ravel()
        dat_rn = dat_r[ravel_mask]
        sim_r = SM2324.ravel()
        sim_rn = sim_r[ravel_mask]
        title_lab = 'dh 24/23'

        
    r2 = compute_r_squared(dat_rn, sim_rn)
    
    midZ = []
    dat_rnBn = []
    sim_rnBn = []
    dat_rnSt = []
    sim_rnSt = []
    zBin = np.arange(1750,2150+50,50)
    for k in np.arange(len(zBin)-1):
        ll = zBin[k]
        ul = zBin[k+1]
        elsIn = np.multiply((z>ll),(z<=ul))
        midZ.append((ul+ll)/2)
        dat_rnBn.append(np.mean(dat_rn[elsIn]))
        sim_rnBn.append(np.mean(sim_rn[elsIn]))
        dat_rnSt.append(np.std(dat_rn[elsIn]))
        sim_rnSt.append(np.std(sim_rn[elsIn]))
    
    # art = ax[i].plot(z,dat_rn,'k.',label='observations')
    ax[i].plot(midZ,dat_rnBn,'k+')
    for j in np.arange(len(midZ)):
        ax[i].plot([midZ[j],midZ[j]],[dat_rnBn[j]-dat_rnSt[j],dat_rnBn[j]+dat_rnSt[j]],'k-',linewidth=6)
    if i ==3:
        ax[i].plot([midZ[0],midZ[0]],[dat_rnBn[0]-dat_rnSt[0],dat_rnBn[0]+dat_rnSt[0]],'k-',linewidth=6,label='obsevrations')
 

    
    # ax[i].plot(z,sim_rn,'r.',label='model')
    ax[i].plot(midZ,sim_rnBn,'r+')
    for j in np.arange(len(midZ)):
        ax[i].plot([midZ[j],midZ[j]],[sim_rnBn[j]-sim_rnSt[j],sim_rnBn[j]+sim_rnSt[j]],'r-',linewidth=3)
    ax[i].set_title(title_lab)
    ax[i].set_ylim(-8,4)

    ax[i].text(0.05, 0.95, f'R-square = {r2:.2f}', transform=ax[i].transAxes,
                fontsize=20, color='white', bbox=dict(facecolor='black', alpha=0.5))
    ax[i].grid()
    ax[i].set_xlabel('Elevation (m)')
    ax[i].set_ylabel('dh (m)')
    if i==3:
        ax[i].plot([midZ[0],midZ[0]],[sim_rnBn[0]-sim_rnSt[0],sim_rnBn[0]+sim_rnSt[0]],'r-',linewidth=3,label='model')
        plt.legend(loc='lower left')
    

plt.show()

# %% plot modelled and simulated
plt.close('all')

fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(36,24))

a = [0,0,0,0]
b = [0,1,2,3]
# a = [0,0,0]
# b = [0,1,2]

for i in np.arange(4):
    
    if i == 0:
        dat = rast_dh2021_mask
        sim = SM2021
        title_lab = 'dh 21/20'
    if i == 1:
        dat = rast_dh2122_mask
        sim = SM2122
        title_lab = 'dh 22/21'

    if i == 2:
        dat = rast_dh2223_mask
        sim = SM2223
        title_lab = 'dh 23/22'

    if i == 3:
        dat = rast_dh2324_mask
        sim = SM2324
        title_lab = 'dh 24/23'

        
    r2 = compute_r_squared(dat, sim)
    art = ax[a[i],b[i]].imshow(dat,cmap = 'jet_r', vmin = -7.5, vmax = 0)
    ax[a[i],b[i]].set_title(title_lab)
    ax[a[i],b[i]].set_yticklabels([])
    ax[a[i],b[i]].set_xticklabels([])
    
    ax[a[i]+1,b[i]].imshow(sim,cmap = 'jet_r', vmin = -7.5, vmax = 0)
    ax[a[i]+1,b[i]].set_title('simulated')
    ax[a[i]+1,b[i]].set_yticklabels([])
    ax[a[i]+1,b[i]].set_xticklabels([])
    ax[a[i]+1,b[i]].text(0.05, 0.95, f'R-square = {r2:.2f}', transform=ax[a[i]+1,b[i]].transAxes,
                fontsize=20, color='white', bbox=dict(facecolor='black', alpha=0.5))

cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.show()


# %% plot difference maps
plt.close('all')

fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(36,24))

a = [0,0,0,0]
b = [0,1,2,3]
# a = [0,0,0]
# b = [0,1,2]

for i in np.arange(4):
    
    if i == 0:
        dat = rast_dh2021_mask
        sim = SM2021
        title_lab = 'dh 21/20'
    if i == 1:
        dat = rast_dh2122_mask
        sim = SM2122
        title_lab = 'dh 22/21'

    if i == 2:
        dat = rast_dh2223_mask
        sim = SM2223
        title_lab = 'dh 23/22'

    if i == 3:
        dat = rast_dh2324_mask
        sim = SM2324
        title_lab = 'dh 24/23'

        
    r2 = compute_r_squared(dat, sim)
    art = ax[i].imshow(sim-dat,cmap = 'jet_r', vmin = -3, vmax = 3)
    ax[i].set_title(title_lab)
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    ax[i].text(0.05, 0.95, f'R-square = {r2:.2f}', transform=ax[i].transAxes,
                fontsize=20, color='white', bbox=dict(facecolor='black', alpha=0.5))

cbar = fig.colorbar(art, ax=ax)
cbar.set_label('dh (m)')
plt.show()

# %% run the forward model

plt.close('all')
# starting from the 20 surface with no snow
surfc = rast_dat24_mask.data
ice_H = np.copy(rast_H)
ice_H = rast_H.data
ice_H[ice_H==255]=0
# pdd_ave = pdd(surfc,temp,datetimes)
pdd_ave = avePDD_forward
nRows = int(2)
nCols = int(6)
lenSim = nCols*nRows
yr_sim = np.arange(lenSim+1)+2024
V = np.zeros(lenSim+1)
A = np.zeros(lenSim+1)
V[0] = np.sum(ice_H*100)
A[0] = np.sum((ice_H>0)*100)
ice_H_mtx = np.zeros((np.shape(ice_H)[0],np.shape(ice_H)[1],lenSim+1))
ice_H_mtx[:,:,0]=ice_H
for yr in np.arange(lenSim):
    slope_yr,aspect_yr=slope_aspect(surfc)
    mask_arr = (slope_yr!=0).astype('int')
    # pdd_ave = pdd(surfc,temp,datetimes)
    SM = B0 + BS*slope_yr.data + BA*aspect_yr.data + BPD*pdd_ave.data + BSM*aveSINM_field.data + BAC*aveSND_field.data
    surfc = (surfc+SM)*mask_arr
    ice_H = (ice_H+SM)*mask_arr
    ice_H[ice_H<=0]=np.nan
    # if yr==0:
    ice_H_mtx[:,:,yr+1] = np.copy(ice_H)
    fig,ax = plt.subplots(figsize=(18,18))
    ax.imshow(ice_H,cmap = 'jet')
    A[yr+1] = np.sum(np.multiply((ice_H!=0),~np.isnan(ice_H)))*100
    
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

def pdd(zMap,temp,datetimes):

    datetime_array = []
    for i in np.arange(11):
        datetime_array.append(datetime(year = 2014+i, month = 9, day = 30))
        
    lapse_t = -6/1000
    t_cube = np.zeros((np.shape(zMap)[0],np.shape(zMap)[1],10))
    for i in np.arange(10):
        start_ti = datetime_array[i]
        end_ti = datetime_array[i+1]
        els_d = (datetimes>=start_ti)&(datetimes<=end_ti)
        els = np.where(els_d==True)
        t_ij = np.zeros(np.shape(zMap)) 
        for j in np.arange(np.shape(els)[1]):
            t_j = temp[els[0][j]]
            lapse_tzj = lapse_t*(zMap-1550)+t_j
            lapse_tzj[lapse_tzj<0] = 0
            t_ij = t_ij + lapse_tzj
            t_cube[:,:,i] = t_ij
       
    pdd_ave = np.nansum(t_cube,2)/10
    plt.figure()
    plt.imshow(pdd_ave)
    return pdd_ave

def get_jet(theMap):
    base_cmap = plt.get_cmap(theMap)
    num_bins = 180
    color_list = base_cmap(np.linspace(0, 1, num_bins))
    color_list[0] = [1, 1, 1, 0] 
    plasma_zero_cmap = mcolors.ListedColormap(color_list)
    return plasma_zero_cmap
