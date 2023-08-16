#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal: mean insolation across Oahu fusing hypyd
"""

import h5pyd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cartopy.crs as ccrs
from scipy.interpolate import interp1d


os.chdir('/Users/amanapat/Documents/hybrid_systems/')
sys.path.append("aakash/")
import valid_marine_energy as bounds 
import wind_time_series as wind_ops
from omni_power_oahu import progress_bar


honolulu = np.array([21.31, 157.86])


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


def solar_series(coord, files):
    """
    Returns time series for GHI
    """
    grid_coords = np.array(files['2017']['coordinates'])
    
    dist_mat = grid_coords - coord
    dist_mat = dist_mat**2
    dist_sq = dist_mat[:, 0] + dist_mat[:, 1]
    # Index of our location of intrest
    idx_min = dist_sq.argmin()

    # Store the time series values we need
    ghi_info = [None for x in files.keys()]
        
    
    k = 0
    for n, sol_file in enumerate(files.values()):
        ghi_info[n] = wind_ops.return_data_recur(sol_file, 'ghi',
                                                 idx_min, k)
        
    ghi_info = np.concatenate(ghi_info)
        
    return ghi_info


def main():
    files = file_query()

    ghi_series = solar_series(honolulu, files)

    # Let's add in our dates
    start = np.datetime64('2014-01-01T00:00')
    end = np.datetime64('2020-01-01T00:00')
    dates_long = np.arange(start, end, np.timedelta64(30, 'm'))

    # Downscale to a 1-hour timeline
    start = np.datetime64('2014-01-01T00:00')
    end = np.datetime64('2020-01-01T00:00')
    dates_short = np.arange(start, end, np.timedelta64(1, 'h'))


    downsamp_func = interp1d(np.array(dates_long, dtype=float),
                             ghi_series)

    ghi_hourly = downsamp_func(np.array(dates_short, dtype=float))
    
    # Write to a df
    ghi_hourly.to_csv('mid_data/ghi_5_year.csv',
                      index=False)


if __name__ == '__main__':
    main()


























