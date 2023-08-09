#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOAL: Extract wind data information for offshore Oahu sites

Structure from time_series_nrg, and funcs from naveen_wind
"""


# 6.1 km distance from shore
# 100 m hub height
# 150 m rotor diameter
# 800 m spacing between turbines
# up to 27 turbine current proposal
# 12-14 km/hr windspeed needed 
# 55 mph max speed 

# 3 m/s cut in
# 25 m/s cut out
# 12.5 m/s max


import h5pyd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import sys, os

os.chdir('/Users/amanapat/Documents/hybrid_systems/')
sys.path.append("aakash/")
import valid_marine_energy as bounds 
import naveen_wind as wind_query
from omni_power_oahu import progress_bar


# global variables that can be tweaked
n_turbines = 100
speed_range = {'min': 3, 
               'max': 12.5,
               'cut-out': 25}  # m/s
turb_props = {'height': 100,
              'radius': 75,
              'space': 800}
turb_props['area'] = np.pi * turb_props['radius']**2


def block_print():
    """
    Disable prints
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Restore prints
    """
    sys.stdout = sys.__stdout__


def get_bounds(fpath, buffer):
    """
    REcursively gets the bounds because this shit is broken for no reason
    """
    try:
        wind_bounds = bounds.oahu_bounds_far_shore(fpath, buffer=buffer)
        _ = wind_bounds.plot();
        return wind_bounds
    except ValueError:
        return get_bounds(fpath, buffer)
    

def extract_coords(bounds, n_sample):
    """
    Exctracts coordinates along the line corresponding to 800m
    spacing 
    
    returns a list of coords
    """
    profile = np.zeros((n_sample, 2))
    shapely_circle = bounds.loc[0]
    
    for n in range(n_sample):
        # Get next point in the line
        point = shapely_circle.interpolate(n / (n_sample - 1), 
                                           normalized=True)
        # Store the value and iterate
        coord = np.array(point.coords)
        profile[n] = coord
    
    return profile


def file_query():
    """
    Queries the relevant wind data for the years 2014-2019,
    returns a list of the API objects
    """
    years = {str(x): None for x in range(2014, 2020)}
    
    for year in years.keys():
        f = h5pyd.File("/nrel/wtk/hawaii/Hawaii_" + year + ".h5", 'r',
                       bucket="nrel-pds-hsds")
        years[year] = f
    
    return years


def return_data_recur(file, field, idx, k=0):
    """
    Recursively tries to return data from a field to the file
    """
    try: 
        data = file[field][:, idx]
        return data / 100
    except OSError:
        if k <= 15:
            k += 1
            print(k)
            return return_data_recur(file, field, idx, k)
        else:
            raise OSError('WeeWooWeeWoooooo')


def wind_series(coords, files, n_turbines):
    """
    Fetches the wind locations with mean windspeed faster than a
    threshold value, returning a time series at that location
    
    Returns windspeed time series at 100 best locs around Oahu,
    along with locations of the same
    """
    grid_coords = np.array(files['2017']['coordinates'])
    data = [None for _ in range(n_turbines)]
    data_coords = [None for _ in range(n_turbines)]
    data_means = np.zeros(n_turbines)
    
    for m, coord in enumerate(coords[:, ::-1]):
        # Bestie
        progress_bar(m + 1, len(coords))
        
        print()
        print('Coord_calc', m)
        dist_mat = grid_coords - coord
        dist_mat = dist_mat**2
        dist_sq = dist_mat[:, 0] + dist_mat[:, 1]
        # Index of our location of intrest
        idx_min = dist_sq.argmin()
        
        print('Speed lookup')
        # Store the time series values we need
        wind_speeds = [None for x in files.keys()]
        
        global k
        for n, wind_file in enumerate(files.values()):
            wind_speeds[n] = return_data_recur(wind_file, 'windspeed_100m',
                                               idx_min, k)
        print('Speed edit')
        # Combine and edit for vals we need for turbine
        wind_speeds = np.concatenate(wind_speeds)
        wind_speeds[wind_speeds < speed_range['min']] = 0
        wind_speeds[wind_speeds >= speed_range['cut-out']] = 0
        wind_speeds[wind_speeds > speed_range['max']] = speed_range['max']
        
        print('Store speed')
        # Calc mean and compare to stored values
        mean_speed = wind_speeds.mean()
        if mean_speed > data_means.min():
            idx = data_means.argmin()
            # Store the faster wind guy!
            data[idx] = wind_speeds
            data_means[idx] = mean_speed
            data_coords[idx] = coord
    
    return data, data_coords
    


if __name__ == '__main__':
    wind_bounds = get_bounds('raw_data/hawaii_bounds/' +
                             'tl_2019_us_coastline.zip', buffer=15)
    # Get the number of samples needed
    length = float(wind_bounds.to_crs('EPSG:3857').length)
    n_sample = int(length / turb_props['space'])
    # Extract coordinates
    coords = extract_coords(wind_bounds, n_sample)
    
    # Collect files to query
    wind_files = file_query()
    
    # Get the fastest wind locations!
    wind_speeds, coords = wind_series(coords, wind_files, n_turbines)

