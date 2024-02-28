#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOAL: Create a means by which I can take a dot product with using the degrees 
data from WW3 with the lines from the boundary to our smoothened coastline
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os import chdir
import sys
from shapely import LineString
import geopandas as gpd
from math import sin, cos


chdir('/Users/amanapat/Documents/hybrid_systems/')
sys.path.append("aakash/")

# Import some custom files here
from omni_power_oahu import download_files, open_file
import valid_marine_energy as bounds
import wave_energy as vect_ops


download_files('201901', get_dp=True)
wave_info = open_file('201901', open_deg=True)

    
oahu_shape = bounds.load_oahu_shape('raw_data/hawaii_bounds/' +
                                   'tl_2019_us_coastline.zip')
oahu_simp = bounds.oahu_shape_simple(oahu_shape)
oahu_bounds_far = bounds.oahu_bounds_far_shore('raw_data/hawaii_bounds/' +
                                               'tl_2019_us_coastline.zip')
oahu_bounds_near = bounds.oahu_bounds_near_shore('raw_data/coast_elevation/' +
                                                 'crm_vol10.nc')

time_select = wave_info['dp'].loc[{'step': wave_info['dp']['step'][0]}]

shore_vects_far = vect_ops.vect_to_shore(oahu_simp, oahu_bounds_far)
shore_vects_near = vect_ops.vect_to_shore(oahu_simp, oahu_bounds_near)

# Extract degrees north values from raster
wave_angle_far = vect_ops.extract_vals(time_select, oahu_bounds_far)
wave_angle_near = vect_ops.extract_vals(time_select, oahu_bounds_near)

# Crete our new vector
wave_vect_far = vect_ops.wave_vector(wave_angle_far, shore_vects_far)
wave_vect_near = vect_ops.wave_vector(wave_angle_near, shore_vects_near)

# Calculate the value    
cos_similar_far = vect_ops.cos_similar(shore_vects_far, wave_vect_far)
cos_similar_near = vect_ops.cos_similar(shore_vects_near, wave_vect_near)

# TODO- use the more explicit cosine similarity function (seems to be more 
# accurate) and incorporate it into the energy generation formula
# iterate over all files and get a time series of wave energy from 2005-2019
# check net generation for near/far shore- daily average, seasonal average
# and long term trends
    
nrg_rast = 0.491 * wave_info['hs']**2 * wave_info['tp']
nrg_time = nrg_rast.loc[{'step': wave_info['dp']['step'][0]}]
    
test_far = vect_ops.energy_flux(nrg_time, oahu_bounds_far, shore_vects_far, 
                                wave_vect_far)
test_near = vect_ops.energy_flux(nrg_time, oahu_bounds_near, shore_vects_near, 
                                wave_vect_near)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    