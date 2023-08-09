#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that will loop over the WW3 data and calculate the energy fluxes
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os import chdir
import sys
from shapely import LineString
import geopandas as gpd
import pandas as pd
import cProfile


chdir('/Users/amanapat/Documents/hybrid_systems/')
sys.path.append("aakash/")

# Import some custom functions here
import omni_power_oahu as download
import valid_marine_energy as bounds
import wave_energy as vect_ops


def main():
    start = '200502'
    end = '201905'
    # Create our date range
    dates = download.assemble_dates(start, end)
    start = np.datetime64(start[:4] + '-' + start[4:] + '-01T00:00')
    end = np.datetime64(end[:4] + '-' + end[4:] + '-31T00:00')
    
    # Initiate the arrays we want to fill
    num_entries = int(1.05 * (np.int64(end - start) / 60 / 3))
    date_arr = np.full(num_entries, start)
    nrg_far_arr = np.full(num_entries, np.nan)
    nrg_close_arr = np.full(num_entries, np.nan)
    
    # Get the coastal bounds
    oahu_shape = bounds.load_oahu_shape('raw_data/hawaii_bounds/' +
                                       'tl_2019_us_coastline.zip')
    oahu_simp = bounds.oahu_shape_simple(oahu_shape)
    oahu_bounds_far = bounds.oahu_bounds_far_shore('raw_data/hawaii_bounds/' +
                                                   'tl_2019_us_coastline.zip')
    oahu_bounds_near = bounds.oahu_bounds_near_shore('raw_data/coast_' +
                                                     'elevation/crm_vol10.nc')
    # Create our vectors to shore
    shore_vects_far = vect_ops.vect_to_shore(oahu_simp, oahu_bounds_far)
    shore_vects_near = vect_ops.vect_to_shore(oahu_simp, oahu_bounds_near)
    
    # Set our initial conditions and compute!
    download.clear_folder('temp_download/')
    n = 0
    curr_time = start
    for date in dates:
        # Download and open the files from the date
        print()
        print(f'Computing for {date}')
        download.download_files(date, get_dp=True)
        wave_info = download.open_file(date, open_deg=True)
        
        # Calculate wave energy
        wave_energy = 0.491 * wave_info['hs']**2 * wave_info['tp']
        
        # Get the time array to be iterated over
        time_arr = np.array(wave_info['dp']['step'], dtype='timedelta64[h]')
        
        for count, time in enumerate(time_arr):
            download.progress_bar(count + 1, len(time_arr))
            # Query the time step of intrest
            query_nrg = wave_energy.loc[{'step': time}]
            query_dp = wave_info['dp'].loc[{'step': time}]
            # Extract degrees north values from raster
            wave_angle_far = vect_ops.extract_vals(query_dp, 
                                                   oahu_bounds_far)
            wave_angle_near = vect_ops.extract_vals(query_dp, 
                                                    oahu_bounds_near)
            # Create our wave vectors
            wave_vect_far = vect_ops.wave_vector(wave_angle_far, 
                                                 shore_vects_far)
            wave_vect_near = vect_ops.wave_vector(wave_angle_near, 
                                                  shore_vects_near)
            # Obtain energy values
            nrg_far = vect_ops.energy_flux(query_nrg, oahu_bounds_far,
                                           shore_vects_far, wave_vect_far)
            nrg_near = vect_ops.energy_flux(query_nrg, oahu_bounds_near,
                                            shore_vects_near, wave_vect_near)
            # Archive the same and start from the top!
            date_arr[n] = curr_time
            nrg_far_arr[n] = nrg_far 
            nrg_close_arr[n] = nrg_near
            # Update conditions
            n += 1
            curr_time += np.timedelta64(3, 'h')
        
        download.clear_folder('temp_download/')

    # Format our outputs, removing the padded zeros
    date_arr = date_arr[~np.isnan(nrg_far_arr)]
    nrg_far_arr = nrg_far_arr[~np.isnan(nrg_far_arr)]
    nrg_close_arr = nrg_close_arr[~np.isnan(nrg_close_arr)]

    nrg_df = pd.DataFrame({'time': date_arr,
                           'far_nrg': nrg_far_arr,
                           'close_nrg': nrg_close_arr})
    nrg_df.to_csv('mid_data/wave_nrg_ALL.csv')


if __name__ == '__main__':
    prof = cProfile.run('main()')
        
        
        
        
        
        
        
  
