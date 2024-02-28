#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal: mean insolation across Oahu fusing hypyd
"""

import h5pyd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


os.chdir('/Users/amanapat/Documents/hybrid_systems/')


locs = {'honolulu': np.array([21.31, -157.86]),
        'pearl': np.array([21.39, -157.97]),
        'waimanalo': np.array([21.34, -157.72]),
        'aiea': np.array([21.37, -157.93]),
        'kailua': np.array([21.39, -157.74]),
        'kaneohe': np.array([21.40, -157.79]),
        'wahiawa': np.array([21.5, -158.02]),
        'waikane': np.array([21.48, -157.84])}

pops = {'honolulu': 345510,
        'pearl': 45159,
        'waimanalo': 5599,
        'aiea': 8839,
        'kailua': 37900,
        'kaneohe': 33540,
        'wahiawa': 5123,
        'waikane': 13700}


def file_query():
    """
    Queries the relevant wind data for the years 2014-2019,
    returns a list of the API objects
    """
    years = {str(x): None for x in range(2014, 2020)}
    
    for year in years.keys():
        f = h5pyd.File("/nrel/nsrdb/v3/nsrdb_" + year + ".h5", 'r',
                       bucket="nrel-pds-hsds")
        years[year] = f
    
    return years


def return_data_recur(file, field, idx, k=0):
    """
    Recursively tries to return data from a field to the file
    """
    try: 
        data = file[field][:, idx]
        return data
    except OSError:
        if k <= 15:
            k += 1
            print(k)
            return return_data_recur(file, field, idx, k)
        else:
            raise OSError('WeeWooWeeWoooooo')


def coords_recur(file_list, year, m=0):
    """
    Recursively returns the coordinates for the file in question
    """
    try:
        return np.array(file_list[year]['coordinates'])
    except:
        if m <=15:
            m += 1
            print(m)
            return coords_recur(file_list, year, m)
        else:
            raise OSError('Meowwwww')



def solar_series(coord, files):
    """
    Returns time series for GHI
    """
    grid_coords = coords_recur(files, '2017')
    
    dist_mat = grid_coords - coord
    dist_mat = dist_mat**2
    dist_sq = dist_mat[:, 0] + dist_mat[:, 1]
    # Index of our location of intrest
    idx_min = dist_sq.argmin()

    # Store the time series values we need
    ghi_info = [None for x in files.keys()]
        
    
    k = 0
    for n, sol_file in enumerate(files.values()):
         ghi = return_data_recur(sol_file, 'ghi',
                                                 idx_min, k)
         
         ghi_info[n] = ghi / sol_file['ghi'].attrs['psm_scale_factor']
        
    ghi_info = np.concatenate(ghi_info)
        
    return ghi_info


def plot_day(ghi_hourly, year, month, day):
    """
    Plots a single day's GHI
    """
    rel_data = ghi_hourly[ghi_hourly.index.year == year]
    rel_data = rel_data[rel_data.index.month == month]
    rel_data = rel_data[rel_data.index.day == day]
    
    # Create our time axis
    base = np.datetime64('2020-01-01 00:00:00')
    time = [base + np.timedelta64(x, 'h') for x in range(0, 24, 1)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xticks(ticks)
    
    plt.plot(time, rel_data)
    plt.xlabel('Hour')
    plt.ylabel('GHI')
    plt.title(f'GHI for {year}-{month}-{day}')
    plt.grid()
    
    return rel_data


def get_dates(files):
    """
    Extracts the dates from the files since something is acting
    up
    """
    dates = [None for x in range(len(file_query()))]
    
    for n, sol_file in enumerate(files.values()):
         dates[n] = pd.to_datetime(sol_file['time_index'][...].astype(str))
    
    return dates[0].union_many(dates[1:]) 


def weighted_avg_ghi(data, pop_data):
    """
    Weighted average for GHI based on population to represent
    distribution of solar panel installation
    """
    new_df = data.copy()
    
    total_pop = sum([pop for pop in pop_data.values()])
    
    for key, value in pop_data.items():
        new_df[key] *= value
        
    ghi_series = new_df.sum(axis=1) / total_pop
    
    return pd.DataFrame({'ghi': ghi_series})


def main():
    files = file_query()
    
    ghi_loc = {key: None for key in locs.keys()}
    
    for key, coord in locs.items():
        ghi_loc[key] = solar_series(coord, files)
    
    # Let's add in our dates
    start = np.datetime64('2014-01-01T00:00')
    end = np.datetime64('2020-01-01T00:00')
    dates_long = np.arange(start, end, np.timedelta64(30, 'm'))
    
    # Write to a df
    
    ghi_loc['time'] = dates_long
    
    ghi_full = pd.DataFrame(ghi_loc)
    ghi_full.set_index('time', drop=True, inplace=True)
    
    ghi_full.index = ghi_full.index.tz_localize(tz='UTC')
    ghi_full.index = ghi_full.index.tz_convert(tz='Pacific/Honolulu')
        
    # Downscale to hourly timeline to match waves/wind
    
    ghi_hourly = ghi_full[::2]  
    
    # Mean across locations for a "typical" day- weighted average
    # for a little more representative value
    
    ghi_mean = weighted_avg_ghi(ghi_hourly, pops)
    
    ghi_mean.to_csv('mid_data/ghi_5_year.csv')

    
if __name__ == '__main__':
    main()

