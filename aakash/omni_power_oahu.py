#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOAL: Loop over all the WaveWatch III data to calculate an omnidirectional 
wave power raster averaged for the time period
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import urllib.request
import time
import sys
import rioxarray as rio


os.chdir('C:\\Users\\Aakas\\Documents\\Grad School\\NREL\\')
sys.path.append("aakash/")


def assemble_dates(early='201406', late='201905'):
    """
    Gives us the list of strings we need to plug into the URL to 
    download the WW3 files
    
    This is massively overengineered but oh well
    """
    # String slicing to get the values of intrest
    start_year = int(early[:4])
    start_month = int(early[4:])
    
    end_year = int(late[:4])
    end_month = int(late[4:])
        
    # Weird formula to get the correct number of entries, accounting
    # for including values at each end
    num_entries = (12 - start_month + 1) +\
                  12 * (end_year - start_year - 1) + end_month
    times = [None for _ in range(num_entries)]

    month = start_month
    year = start_year
    # Loop over values and merge as necessary
    for num in range(num_entries):
        month_str = str(month)
        if len(month_str) == 1:
            month_str = '0' + month_str
        
        times[num] = str(year) + month_str
        
        month += 1
        if month == 13:
            month = 1
            year += 1
            
    return times
        

def ww3_archive_url(datestr):
    """
    Returns the wavewatch urls for HS, TP, DP at the given time string
    Looking at west coast 4m resolution, since that's the analysis we are
    curous about
    
    HS is the sig. wave height, TP is the wave period, DP is the wave
    direction in degrees. Not using DP for now but will probably
    need later
    """
    global_url = 'https://polar.ncep.noaa.gov/waves/hindcasts/multi_1/'
    
    hs_time = global_url + datestr + '/gribs/multi_1.wc_4m.hs.' +\
              datestr + '.grb2'
    tp_time = global_url + datestr + '/gribs/multi_1.wc_4m.tp.' +\
              datestr + '.grb2'
    dp_time = global_url + datestr + '/gribs/multi_1.wc_4m.dp.' +\
              datestr + '.grb2'
              
    files = {'hs': hs_time, 'tp': tp_time, 'dp': dp_time}
    
    return files


def download_files(datestr, folder='temp_download/', get_dp=False):
    """
    Downloads HS, TP, DP for a file at the given datestring
    
    downloads it to the "temp_download" folder I have sitting around
    """
    filepaths = ww3_archive_url(datestr)
    
    for key, val in filepaths.items():
        # Don't download dp unless necessary to speed up current use
        # Literal 4.2 sec increase per date
        if get_dp is False and key == 'dp':
            continue
        # We will clear this folder later
        urllib.request.urlretrieve(val, folder + key + '_' +
                                   datestr + '.grb2')

def fix_coords(df):
    """
    Converts between native gfs degrees east to normal lat/long
    """
    df.coords['longitude'] = (df.coords['longitude'] + 180) % 360 - 180
    df = df.sortby(df.longitude)
    
    return df


def format_xarr(beep):
    """
    Used by the open_file command to format the raw xarray files by fixing
    the coordinate system and subsetting to Oahu
    """
    # Fixing the weird way the coordinates are read to correct values
    beep = fix_coords(beep)

    # To a useable format
    boop = beep.to_array().squeeze()
    
    min_lat = 17
    max_lat = 24

    min_lon = -162
    max_lon = -153
    
    # Crop to just Hawaii
    boop = boop.loc[dict(latitude=slice(max_lat, min_lat))]
    boop = boop.loc[dict(longitude=slice(min_lon, max_lon))]
    
    return boop 

    
def open_file(datestr, folder='temp_download/', open_deg=False):
    """
    Opens and processes the HS/TP files in the temp_download folder
    and returns a dict with both files
    
    This is essentially a time series dataset for the month in question- 
    need to do filtering by month later
    
    We will also subset this to just Hawaii so we have data in RAM of a 
    more managable size
    
    Tempted to add something to load in the degrees file but I can
    always rewrite this in a different file later
    """
    hs_array = xr.open_dataset(folder + 'hs' + '_' + datestr + '.grb2',
                               engine='cfgrib')
    tp_array = xr.open_dataset(folder + 'tp' + '_' + datestr + '.grb2',
                               engine='cfgrib')
    # Format the files as described in func
    hs_array = format_xarr(hs_array)
    tp_array = format_xarr(tp_array)
    
    wave_info = {'hs': hs_array, 'tp': tp_array}
    
    if open_deg:
        dp_array = xr.open_dataset(folder + 'dp' + '_' + datestr + '.grb2',
                                   engine='cfgrib')
        dp_array = format_xarr(dp_array)
        wave_info['dp'] = dp_array
    
    return wave_info


def calc_mean_energy(wave_info):
    """
    Calculates the energy flux raster with the two files in questions, and
    averages it for our time period
    
    Easier than expected
    """
    energy_times = 0.491 * wave_info['hs']**2 * wave_info['tp']
    energy_mean = energy_times.mean(dim='step')
    
    return energy_mean


def clear_folder(folder='temp_download/'):
    """
    Clears the contents of the temp download folder
    """
    files = os.listdir(folder)
    for f in files:
        os.remove(folder + f)


def progress_bar(n, max_val, cus_str=''):
    """
    I love progress bars in long loops
    """
    sys.stdout.write('\033[2K\033[1G')
    print(f'Computing...{100 * n / max_val:.2f}% complete ' + cus_str,
          end="\r")


def plot_output(flux_array, title='5 Year Average'):
    """
    Plots the output energy field in W/m as a contour map
    """
    levels = [0, 5, 10, 15, 20, 25, 30, 40]

    fig = plt.figure(2, figsize=(5, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plot = plt.contourf(flux_array.longitude, flux_array.latitude,
                 flux_array.variable, transform=ccrs.PlateCarree(),
                 levels=levels)

    proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
             pc in plot.collections]

    ax.coastlines()
    ax.legend(proxy, ['0-5 kW/m', '5-10 kW/m', '10-15 kW/m',
                      '15-20 kW/m', '20-25 kW/m', '25-30 kW/m',
                      '> 30 kW/m'], 
              loc='lower left')
    ax.set_title('Wave Energy Flux, ' + title)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    plt.show()


def add_data_hack(og_xarr, to_add_xarr):
    """
    Checks if the variable is defined and if it has it does a normal
    add
    
    If not it returns the same and that becomes our definition
    
    Hack to initiate the loop variable
    
    Second level hack- the coordinate basis changes between 201709 and 201710
    its a very small devision (~0.04%) but enough to break the script
    
    Because this deviation is within the grid cell, I'm just choosing to 
    overwrite the new array longitude with the new on
    
    Sue me its a 4meter error
    """
    if og_xarr is None:
        return to_add_xarr
    else:
        if og_xarr['longitude'][0] == to_add_xarr['longitude'][0]:
            return og_xarr + to_add_xarr
        else:
            og_xarr['longitude'] = to_add_xarr['longitude'] 
            return og_xarr + to_add_xarr


def main():
    dates = assemble_dates('201806', '201905')
    net_nrg = None
    
    for count, datestr in enumerate(dates):
        # My best friend
        progress_bar(count, len(dates))
        # Download the files from WW3
        download_files(datestr)
        # Open into xarray and format correctly
        wave_info = open_file(datestr)
        # Calculate energy using formula, average for month
        energy = calc_mean_energy(wave_info)
        # Add to net energy tally
        net_nrg = add_data_hack(net_nrg, energy)
        # Clear the folder - CAREFUL WITH THIS
        clear_folder(folder='temp_download/')
      
    mean_energy = net_nrg / len(dates)
    plot_output(mean_energy, 'ALL Year Average')
    
    # Output the final file!
    mean_energy.rio.to_raster('mid_data/1_year_avg_pt2.tif')
        
    
if __name__ == '__main__':
    main()
    