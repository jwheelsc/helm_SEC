# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:00:33 2025

@author: jcrompto
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:07:19 2025

@author: jcrompto
"""

import os
import sys
from functools import reduce
from datetime import datetime, timedelta, time
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import pandas as pd
import pyproj
from pyproj import Transformer

import netCDF4
import rasterio as rio
import geoutils as gu

from shapely.geometry import Polygon, Point
from matplotlib.path import Path
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d


# %% here is script imported from the radar script just to get the boundaries and the mask 
data = pd.read_csv(r'C:\Users\jcrompto\Documents\remote_sensing\Q_projects\Helm\data_frame_all.csv')
data = data[data['type'] == 'ground']
# data['ice_thickness'] = data['ice_thickness'] - 1.8

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
