#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing data for Naveen- extracting values at
certain locations

Structure highly inspired by the time_series_nrg file,
although different enough to warrant new coding
"""

import xarray as xr
import numpy as np
from os import chdir
import sys
import pandas as pd
from scipy.interpolate import pchip_interpolate


chdir('/Users/amanapat/Documents/hybrid_systems/')
sys.path.append("aakash/")


# Import some custom functions here
import omni_power_oahu as download


locs = {'Oahu_N': {'lon': -157.9, 'lat': 21.76},
        'Oahu_N_call': {'lon': -158.5, 'lat': 21.7},
        'Oahu_S_call': {'lon': -158, 'lat': 21.16}}
params = ['hs', 'tp', 'dp']


def main():
    # Initiate our date range
    start = '201806'
    end = '201905'
    dates = download.assemble_dates(start, end)
    start = np.datetime64(start[:4] + '-' + start[4:] + '-01T00:00')
    end = np.datetime64(end[:4] + '-' + end[4:] + '-31T21:00')
    
    # Initiate the arrays we want to fill
    num_entries = int(1.05 * (np.int64(end - start) / 60 / 3))
    date_arr = np.full(num_entries, start)
    data_arr_nan = np.full(num_entries, np.nan)
    
    # Now, duplicate this for our three locations of intrest
    # A similar loop must be used everywhere for all processing
    
    wave_data = dict()
    for loc in locs.keys():
        wave_data[loc] = dict()
        for param in params:
            wave_data[loc][param] = data_arr_nan.copy()
    
    download.clear_folder('temp_download/')
    n = 0
    curr_time = start
    for date in dates:
        print(f'Computing for {date}')
        # Download the file of intrest
        download.download_files(date, get_dp=True)
        wave_info = download.open_file(date, open_deg=True)
        
        # Get the time array we are iterating across
        time_arr = np.array(wave_info['dp']['step'], 
                            dtype='timedelta64[h]')
        date_arr[n:n + len(time_arr)] = curr_time + time_arr
        
        # Extract values for each point over time
        for loc in wave_data.keys():
            # Loop over locations
            for param in params:
                # Loop over data needed
                wave_data[loc][param][n:n + len(time_arr)] =\
                    wave_info[param].sel(latitude=locs[loc]['lat'], 
                    longitude=locs[loc]['lon'], method='nearest').data
                    
        # Update our indices
        n += len(time_arr)
        curr_time += time_arr[-1]
        download.clear_folder('temp_download/')
    
    # Remove the padded nan's
    date_arr = date_arr[~np.isnan(wave_data['Oahu_N']['hs'])] 
    for loc in wave_data.keys():
        for param in params:
            wave_data[loc][param] = wave_data[loc][param][~np.isnan(
                wave_data[loc][param])]
    
    # Create the pandas dataframes containing our data
    for key in wave_data.keys():
        wave_data[key] = pd.DataFrame(wave_data[key])
        wave_data[key].index = date_arr
        wave_data[key] = wave_data[key][~wave_data[key].index.duplicated(keep='first')]
    
    # Let's get this into a 1-hour time series 
    start = date_arr[0]
    end = date_arr[-1]
    date_arr_new = np.arange(start, end + np.timedelta64(1, 'h'),
                             np.timedelta64(1, 'h'))
    date_arr_new = pd.to_datetime(date_arr_new)
    
    wave_data_new = dict()
    print('Interpolating...')
    for key in wave_data.keys():
        wave_data_new[key] = pd.DataFrame(index=date_arr_new)
        for param in params:
            wave_data_new[key][param] = pchip_interpolate(wave_data[key].index,
                                                   wave_data[key][param],
                                                   date_arr_new)
        
    # Let's now try to format in the Naveen-style
    for key in wave_data_new.keys():
        wave_data_new[key] = wave_data_new[key].rename(columns={
            'hs': 'wave height','tp': 'wave period', 'dp': 'degrees true'})
        # I tried to do this intelligently but it didn't matter
        # so its a manual mess
        wave_data_new[key]['Year'] = wave_data_new[key].index.year
        wave_data_new[key]['Month'] = wave_data_new[key].index.month
        wave_data_new[key]['Day'] = wave_data_new[key].index.day
        wave_data_new[key]['Hour'] = wave_data_new[key].index.hour
        wave_data_new[key]['Minute'] = wave_data_new[key].index.minute
        # Remove the original index
        wave_data_new[key].reset_index(inplace=True, drop=True)
        
    # Add in the intro string that he wants
    line_1 = 'Source,Location ID,Jurisdiction,Latitude,Longitude,' +\
             'Time Zone,Local Time Zone,Distance to Shore,' +\
             'Directionality Coefficient,Energy Period,' +\
             'Maximum Energy Direction,Mean Absolute Period,' +\
             'Mean Wave Direction,Mean Zero-Crossing Period,' +\
             'Omni-Directional Wave Power,' +\
             'Peak Period,Significant Wave Height,' +\
             'Spectral Width,Water Depth,Version\n'
             
    # Coords formatted for each location
    line_2_Oahu_N = f'WaveWatchIII,-,Hawaii,{locs["Oahu_N"]["lat"]},' +\
                    f'{locs["Oahu_N"]["lon"]},0,-10,-,-,s,deg,s,' +\
                    'deg,s,W/m,s,m,-,-,v1.10\n{}'
    line_2_Oahu_N_call = f'WaveWatchIII,-,Hawaii,{locs["Oahu_N_call"]["lat"]},' +\
                    f'{locs["Oahu_N_call"]["lon"]},0,-10,-,-,s,deg,s,' +\
                     'deg,s,W/m,s,m,-,-,v1.10\n{}'
    line_2_Oahu_S_call = f'WaveWatchIII,-,Hawaii,{locs["Oahu_S_call"]["lat"]},' +\
                    f'{locs["Oahu_S_call"]["lon"]},0,-10,-,-,s,deg,s,' +\
                     'deg,s,W/m,s,m,-,-,v1.10\n{}'
    
    # Output our files!
    # North of Oahu
    with open('mid_data/wave_resource_Oahu_N.csv', 'w') as fp:
        template = line_1 + line_2_Oahu_N
        fp.write(template.format(wave_data_new['Oahu_N'].to_csv(index=False)))
    # North call area
    with open('mid_data/wave_resource_Oahu_N_call.csv', 'w') as fp:
        template = line_1 + line_2_Oahu_N_call
        fp.write(template.format(wave_data_new['Oahu_N_call'].to_csv(index=False)))
    # South call area
    with open('mid_data/wave_resource_Oahu_S_call.csv', 'w') as fp:
        template = line_1 + line_2_Oahu_S_call
        fp.write(template.format(wave_data_new['Oahu_S_call'].to_csv(index=False)))
   
        
if __name__ == '__main__':
    main()
        


