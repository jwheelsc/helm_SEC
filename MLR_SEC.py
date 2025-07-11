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
data = pd.read_csv(r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\data_frame_all.csv')
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
filename = r'C:/Users/jcrompto/Documents/remote_sensing/Q_projects/Helm/marg_25.tif'
# rast_dem = gu.Raster(filename,downsample=10)
rast_dem_marg = gu.Raster(filename)
rast_dem_marg.crop([min_x, min_y, max_x, max_y])
rast_dat_marg = rast_dem_marg.data

# %% Compute a slope and aspect from 10m downsampled DEMs. grid cells are only
# considered for years prior to 2025 from within 2025 glacier margin

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/DEM_2020.tif'
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

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/DEM_2021.tif'
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


filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/DEM_2022.tif'
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

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/DEM_2023.tif'
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

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/DEM_2024.tif'
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

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/dh_20_21.tif'
rast_dh2021 = gu.Raster(filename)
rast_dh2021.crop([min_x, min_y, max_x, max_y])
rast_dh2021 = rast_dh2021.data
rast_dh2021_mask = rast_dh2021*slim_mask
dh2021_r = rast_dh2021_mask.ravel()
dh2021_rn = dh2021_r[ravel_mask]

# fig,ax = plt.subplots(figsize=(18,18))
# art = ax.imshow(rast_dh2021_mask,cmap = 'jet_r')
# cbar = fig.colorbar(art, ax=ax)
# cbar.set_label('dh (m)')
# plt.title('dh 21/20')

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/dh_21_22.tif'
rast_dh2122 = gu.Raster(filename)
rast_dh2122.crop([min_x, min_y, max_x, max_y])
rast_dh2122 = rast_dh2122.data
rast_dh2122_mask = rast_dh2122*slim_mask
dh2122_r = rast_dh2122_mask.ravel()
dh2122_rn = dh2122_r[ravel_mask]

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(rast_dh2122_mask,cmap=cbar.cmap, vmin = cbar.vmin, vmax = cbar.vmax)
# cbar = fig.colorbar(art, ax=ax)
# cbar.set_label('dh (m)')
# plt.title('dh 22/21')

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/dh_22_23.tif'
rast_dh2223 = gu.Raster(filename)
rast_dh2223.crop([min_x, min_y, max_x, max_y])
rast_dh2223 = rast_dh2223.data
rast_dh2223_mask = rast_dh2223*slim_mask
dh2223_r = rast_dh2223_mask.ravel()
dh2223_rn = dh2223_r[ravel_mask]

# fig,ax = plt.subplots(figsize=(18,18))
# ax.imshow(rast_dh2223_mask,cmap=cbar.cmap, vmin = cbar.vmin, vmax = cbar.vmax)
# cbar = fig.colorbar(art, ax=ax)
# cbar.set_label('dh (m)')
# plt.title('dh 23/22')

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/dh_23_24.tif'
rast_dh2324 = gu.Raster(filename)
rast_dh2324.crop([min_x, min_y, max_x, max_y])
rast_dh2324 = rast_dh2324.data
rast_dh2324_mask = rast_dh2324*slim_mask
dh2324_r = rast_dh2324_mask.ravel()
dh2324_rn = dh2324_r[ravel_mask]

filename = r'C:/Users/jcrompto/Documents/remote_sensing/lidar/Helm/dh_24_25_w.tif' # this is the winter field
rast_dh2425 = gu.Raster(filename)
rast_dh2425.crop([min_x, min_y, max_x, max_y])
rast_dh2425 = rast_dh2425.data
rast_dh2425_mask = rast_dh2425*slim_mask
dh2425_r = rast_dh2425_mask.ravel()
dh2425w_rn = dh2425_r[ravel_mask]

# fig,ax = plt.subplots(figsize=(18,18))
# art = ax.imshow(rast_dh2425_mask,cmap='jet')
# cbar = fig.colorbar(art, ax=ax)
# cbar.set_label('dh (m)')
# plt.title('dh 24/45_w')

# %% average the geodetic balance

melt_cube = np.zeros((np.shape(rast_dh2324_mask)[0],np.shape(rast_dh2324_mask)[1],4))

melt_cube[:,:,0] = rast_dh2021_mask
melt_cube[:,:,0] = rast_dh2122_mask
melt_cube[:,:,0] = rast_dh2223_mask
melt_cube[:,:,0] = rast_dh2324_mask

ave_melt = (rast_dh2021*rast_dat_marg+rast_dh2122*rast_dat_marg+rast_dh2223*rast_dat_marg+rast_dh2324*rast_dat_marg)/4
# plt.close('all')
# plt.figure()
# plt.imshow(ave_melt)

# %% load this ice thickness data
filename = r'C:/Users/jcrompto/Documents/remote_sensing/Q_projects/Helm/helm_H_IPR.tif'
rast_H = gu.Raster(filename)
rast_H.crop([min_x, min_y, max_x, max_y])
rast_H = rast_H.data
rast_H = rast_H*rast_dat_marg

# %% read in the masked yearly shortwave radiation fields
plt.close('all')
filename = r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\2021_summer_insol_mean.tif'
rast_SINM_21 = gu.Raster(filename)
rast_SINM_21.crop([min_x, min_y, max_x, max_y])
SINM_21_dat = rast_SINM_21.data
# fig,ax = plt.subplots(figsize=(18,18))
SINM_21_dat_field = SINM_21_dat*slim_mask
# art = ax.imshow(SINM_21_dat_field,cmap = 'jet',vmin=0, vmax =270)
# cbar = plt.colorbar(art)
SINM_21 = SINM_21_dat_field.ravel()
SINM_21_rn = SINM_21[ravel_mask]


filename = r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\2022_summer_insol_mean.tif'
rast_SINM_22 = gu.Raster(filename)
rast_SINM_22.crop([min_x, min_y, max_x, max_y])
SINM_22_dat = rast_SINM_22.data
# fig,ax = plt.subplots(figsize=(18,18))
SINM_22_dat_field = SINM_22_dat*slim_mask
# art = ax.imshow(SINM_22_dat_field,cmap = 'jet',vmin=0, vmax =270)
# cbar = plt.colorbar(art)
SINM_22 = SINM_22_dat_field.ravel()
SINM_22_rn = SINM_22[ravel_mask]

filename = r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\2023_summer_insol_mean.tif'
rast_SINM_23 = gu.Raster(filename)
rast_SINM_23.crop([min_x, min_y, max_x, max_y])
SINM_23_dat = rast_SINM_23.data
# fig,ax = plt.subplots(figsize=(18,18))
SINM_23_dat_field = SINM_23_dat*slim_mask
# art = ax.imshow(SINM_23_dat_field,cmap = 'jet',vmin=0, vmax =270)
# cbar = plt.colorbar(art)
SINM_23 = SINM_23_dat_field.ravel()
SINM_23_rn = SINM_23[ravel_mask]


filename = r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\2024_summer_insol_mean.tif'
rast_SINM_24 = gu.Raster(filename)
rast_SINM_24.crop([min_x, min_y, max_x, max_y])
SINM_24_dat = rast_SINM_24.data
# fig,ax = plt.subplots(figsize=(18,18))
SINM_24_dat_field = SINM_24_dat*slim_mask
# art = ax.imshow(SINM_24_dat_field,cmap = 'jet',vmin=0, vmax =270)
# cbar = plt.colorbar(art)
SINM_24 = SINM_24_dat_field.ravel()
SINM_24_rn = SINM_24[ravel_mask]

# %% read in snow depth fields

file_path = r'C:/Users/jcrompto/Documents/data/mass_balance/Helm/Helm_snow_depth.csv'
df = pd.read_csv(file_path)
yr = df['year']
z = df['z']
sd = df['depth_m']

sd21 = sd[yr==2021]
pz21 = z[yr==2021]
sd22 = sd[yr==2022]
pz22 = z[yr==2022]
sd23 = sd[yr==2023]
pz23 = z[yr==2023]
sd24 = sd[yr==2024]
pz24 = z[yr==2024]

vmn = 0
vmx = 7
degp = 1 # degree of polynomial to interploate snow depth field

# ax.plot(z21,bw21,label = '2021')
x = pz21; y = sd21
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
# x_fit = np.linspace(np.min(x), np.max(x), 20)
sd21_field = poly(z21) #z2X is the elevation field for the DEM of the corresponding year
# fig,ax = plt.subplots()
# art = ax.imshow(sd21_field,cmap='jet',vmin = vmn, vmax = vmx)
# cbar = fig.colorbar(art, ax=ax)
sd21_r = sd21_field.ravel()
sd21_rn = sd21_r[ravel_mask]


x = pz22; y = sd22
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
# x_fit = np.linspace(np.min(x), np.max(x), 20)
sd22_field = poly(z22)
# fig,ax = plt.subplots()
# art = ax.imshow(sd22_field,cmap='jet',vmin = vmn, vmax = vmx)
# cbar = fig.colorbar(art, ax=ax)
sd22_r = sd22_field.ravel()
sd22_rn = sd22_r[ravel_mask]


x = pz23; y = sd23
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
# x_fit = np.linspace(np.min(x), np.max(x), 20)
sd23_field = poly(z23)
# fig,ax = plt.subplots()
# art = ax.imshow(sd23_field,cmap='jet',vmin = vmn, vmax = vmx)
# cbar = fig.colorbar(art, ax=ax)
sd23_r = sd23_field.ravel()
sd23_rn = sd23_r[ravel_mask]


x = pz24; y = sd24
coeffs = np.polyfit(x, y, deg=degp)
poly = np.poly1d(coeffs)
# x_fit = np.linspace(np.min(x), np.max(x), 20)
sd24_field = poly(z24)
# fig,ax = plt.subplots()
# art = ax.imshow(sd24_field,cmap='jet',vmin = vmn, vmax = vmx)
# cbar = fig.colorbar(art, ax=ax)
sd24_r = sd24_field.ravel()
sd24_rn = sd24_r[ravel_mask]

# %% import the era5 temp data at 1550m elevation from nearest grid cell
file_path = r'C:/Users/jcrompto/Documents/code/python_scripts/mass_balance/helm_era5_2000_2024.xlsx'
df = pd.read_excel(file_path)  # or sheet_name='Sheet1'
datetimes = df.timestamp
temperature = df.temperature-273.15

# %% PDD from smooth curve

lapse_t = -6/1000 #set the desired lapse rate

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
        
    t_zji[:,i] = t_zj
    
# %% compute average fields for radiation mask and snow depth from 2021--2024 to 
# use for forward model

aveSINM_field = (SINM_21_dat_field + SINM_22_dat_field + SINM_23_dat_field + SINM_24_dat_field)/4
aveSND_field = (sd21_field + sd22_field + sd23_field + sd24_field)/4


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
    # Flatten in case inputs are 2D or higher
    obs = observed.flatten()
    mod = modeled.flatten()
    
    # Mask invalid or nan values (optional but good practice)
    valid_mask = ~np.isnan(obs) & ~np.isnan(mod)
    obs = obs[valid_mask]
    mod = mod[valid_mask]

    # Compute R²
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
    # model.coef_ is for standardized inputs and outputs
    beta_std = model.coef_.flatten()
    intercept_std = model.intercept_

    # Denormalize coefficients
    beta_denorm = beta_std * (y_std / X_std)
    intercept_denorm = y_mean - np.sum(beta_denorm * X_mean)
    
    return intercept_denorm, beta_denorm

def pdd(zMap,temp,datetimes):
    #% PDD from smooth curve
    
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
       
    pdd_ave = np.nanmean(t_cube,2)
    plt.figure()
    plt.imshow(pdd_ave)
    return pdd_ave

def get_jet(theMap):
    base_cmap = plt.get_cmap(theMap)
    num_bins = 180
    color_list = base_cmap(np.linspace(0, 1, num_bins))
    color_list[0] = [1, 1, 1, 0]  # RGBA for white
    plasma_zero_cmap = mcolors.ListedColormap(color_list)
    return plasma_zero_cmap
