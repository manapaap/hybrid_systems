#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOAL: extract wind data for Naveen

Will use similar structire as naveen_data as pulling similar
data
"""

import h5pyd
import pandas as pd
import numpy as np
from pyproj import Proj
import dateutil
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from copy import deepcopy


locs = {'Oahu_N': { 'lat': 21.76, 'lon': -157.9},
        'Oahu_N_call': {'lat': 21.7, 'lon': -158.5},
        'Oahu_S_call': {'lat': 21.16, 'lon': -158}}
# Heights of intrest, in m
eval_vals = [87.6, 161]


def slicer_vectorized(a, start, end):
    """
    Vectorized approach to slicing numpy array of strings
    """
    b = a.view((str, 1)).reshape(len(a), -1)[:,start:end]
    
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))


def indices_for_coord(f, lat_lon_dict):
    """
    Getting coords at minimum distance from query loc
    calculating straight line distance matrix
    
    Haversine distance probably better here
    but flat is good enough for immediate purpose
    as the recorded loc should be very close to
    the one we are querying
    """
    dset_coords = np.array(return_data(f, 'coordinates'))
    loc = np.array([lat_lon_dict['lat'],
                    lat_lon_dict['lon']])
    
    dist_mat =  dset_coords - loc
    dist_mat = dist_mat **2
    dist = dist_mat[:, 0] + dist_mat[:, 1]
    
    idx_min = dist.argmin()
    coords = dset_coords[idx_min]
    
    return coords, idx_min


def return_query_data(f, time_arr, space_idx, field='windspeed_100m', k=0):
    """
    Returns data across the time period specified and at the
    specified location on the array
    
    Also recursive to try to find the problem
    """
    try:
        data = return_data(f, field)
        year_series = data[:, space_idx]
        time_series_rel = year_series[time_arr]
    except:
        if k > 15:
            raise IOError('WeewooWeewoo')
        else:
            k += 1
            return return_query_data(f, time_arr, space_idx, field, k)
    
    return time_series_rel
 

def wind_points(f, heights, time_arr, space_idx, field='windspeed'):
    """
    Extracts all the wind data and stacks it into a 2D array
    for wind at each height for all time time * height
    
    For purposes of interpolation
    """
    heights_str = [field + '_' + str(x) + 'm' for x in heights] 
    
    arrays = [None for _ in heights]
    
    n = 0
    global height
    for height in heights_str:
        data = return_query_data(f, time_arr, space_idx, height)
        # Everyting is 100 times larger!?
        data = data / 100
        arrays[n] = data
        n += 1
    
    arrays = np.array(arrays)
    
    return arrays.T


def fit_func(heights, array, eval_vals):
    """
    Fits a spline along the one axis of the numpy array,
    and evaluates it along the eval_vals
    """
    spline = CubicSpline(heights, array, axis=1)
    
    return spline(eval_vals)
        

def return_data(f, field, m=0):
    """
    Uses the h5pyd API to query data, but as that occassionally
    fails, this recursive function will ensure that 
    it finds the data when we know the path is correct
    
    Still raises the error after unreasonable number of iters
    """
    try:
        return f[field]
    except:
        if m > 15:
            raise OSError('Error rerieving data: Real this time')
        else:
            return return_data(f, field, m)


def return_date_idx(f, fpath, p=0,):
    """
    REcursively fetches the date index of the file
    """
    try:
        str_idx = np.array(return_data(f, 'time_index'), dtype=str)
        str_idx = slicer_vectorized(str_idx, 0, 19)
        date_idx = np.array(str_idx, dtype=np.datetime64)
        return date_idx, f
    except:
        if p > 15:
            raise OSError('Nooooo')
        else:
            p += 1
            f = h5pyd.File(fpath, 'r',
                           bucket="nrel-pds-hsds")
            return return_date_idx(f, fpath, p)


def data_to_df(f, filter_year, dates, locs_dict):
    """
    Extracts the data we need at heights needed by successively querying
    the file in question in the correct order, into a pandas df
    
    does the bulk of computation in this file
    """
    # Relevant elevations to query data
    wind_heights = np.arange(20, 220, 20)
    pres_heights = np.arange(0, 300, 100)
    
    # result dict
    result_dict = {key: {} for key in locs_dict.keys()}
    
    for loc, value in locs_dict.items():
        print(loc, value)
        global wind_arr, speed_vals, test
        # Get the elev vs. time arrays for all locs
        wind_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='windspeed')
        temp_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='temperature')
        dir_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='winddirection')
        pres_arr = wind_points(f, pres_heights, filter_year, 
                               space_idx, field='pressure')
        pres_arr = pres_arr * 10 # Convert to mB
        
        # Now, let's fit the splines and evaluate at the values
        speed_vals = fit_func(wind_heights, wind_arr, eval_vals)
        temp_vals = fit_func(wind_heights, temp_arr, eval_vals)
        dir_vals = fit_func(wind_heights, dir_arr, eval_vals)
        pres_vals = fit_func(pres_heights, pres_arr, eval_vals)
        
        # Let's store these arrays, separated by value
        # Tried to autmate but manual is just easier
        result_dict[loc]['wind_87.6'] = speed_vals[:, 0]
        result_dict[loc]['wind_161'] = speed_vals[:, 1]
        result_dict[loc]['temp_87.6'] = temp_vals[:, 0]
        result_dict[loc]['temp_161'] = temp_vals[:, 1]
        result_dict[loc]['dir_87.6'] = dir_vals[:, 0]
        result_dict[loc]['dir_161'] = dir_vals[:, 1]
        result_dict[loc]['pres_87.6'] = pres_vals[:, 0]
        result_dict[loc]['pres_161'] = pres_vals[:, 1]
        
        # Turn into Pandas dataframe
        test = result_dict
        result_dict[loc] = deepcopy(pd.DataFrame(oahu_2018[loc]))
        result_dict[loc].index = dates
        
    return result_dict
    
    


'''
f = h5pyd.File("/nrel/wtk/hawaii/Hawaii_2018.h5", 'r',
               bucket="nrel-pds-hsds")


# Get the time index
str_idx = np.array(return_data(f, 'time_index'), dtype=str)
str_idx = slicer_vectorized(str_idx, 0, 19)
date_idx = np.array(str_idx, dtype=np.datetime64)

# Indices for our period of intrest
rel_2018 = date_idx >= np.datetime64('2018-06-01T00:00:00')
dates_rel = date_idx[rel_2018]

real_coords, space_idx = indices_for_coord(f, locs['Oahu_N'])

wind_speed_heights = np.arange(20, 220, 20)
wind_range = wind_points(f, wind_speed_heights, rel_2018, space_idx, 
                         field='windspeed')

# What naveen asked for
eval_vals = [87.6, 161]
# Fit a curve
wind_speed_vals = fit_func(wind_speed_heights, wind_range, eval_vals)

# Same for temperature
temp_range = wind_points(f, wind_speed_heights, rel_2018, space_idx,
                         field='temperature')
temp_vals = fit_func(wind_speed_heights, temp_range, eval_vals)

# Same for direction
dir_range = wind_points(f, wind_speed_heights, rel_2018, space_idx,
                         field='winddirection')
dir_vals =  fit_func(wind_speed_heights, dir_range, eval_vals)

# Pressure needs a new array but it will work
pres_vals = np.array([0, 100, 200])
pres_range = wind_points(f, pres_vals, rel_2018, space_idx,
                         field='pressure')
# Renormalize to get a value in mB
pres_range = pres_range * 10
pres_vals = fit_func(wind_speed_heights, temp_range, eval_vals)

'''

# TODO: Stack observations together for temp, pres, wind direction
# and wind speed 


# Let's loop over all the locations of intrest for 2018
# then determine the 2019 workflow

if __name__ == '__main__':
    # Query our files
    f_2018 = h5pyd.File("/nrel/wtk/hawaii/Hawaii_2018.h5", 'r',
                   bucket="nrel-pds-hsds")
    f_2019 = h5pyd.File("/nrel/wtk/hawaii/Hawaii_2019.h5", 'r',
                   bucket="nrel-pds-hsds")
    
    # Get the time index for 2018
    date_idx_2018, f_2018 = return_date_idx(f_2018, "/nrel/wtk/hawaii/Hawaii_2018.h5")
    # Indices for 2018
    rel_2018 = date_idx_2018 >= np.datetime64('2018-06-01T00:00:00')
    dates_rel_2018 = date_idx_2018[rel_2018]
    
    # Same for 2019
    date_idx_2019, f_2019 = return_date_idx(f_2019, "/nrel/wtk/hawaii/Hawaii_2019.h5")
    # Get indices
    rel_2019 = date_idx_2019 <= np.datetime64('2019-06-01T00:00:00')
    dates_rel_2019 = date_idx_2019[rel_2019]
    
    # Create our dict to hold location information
    locs_real = {key: dict() for key in locs.keys()}
    
    # Identify the coordinates of intrest for each, with index
    for key, value in locs.items():
        real_coords, space_idx = indices_for_coord(f_2018, value)
        real_coords_2, space_idx_2 = indices_for_coord(f_2019, value)
        
        # Sanity check between years
        if (real_coords == real_coords_2).all() and space_idx == space_idx_2:
           locs_real[key]['coords'] = real_coords
           locs_real[key]['idx'] = space_idx
        else:
            raise Exception('Coordinate grid broke')
    
    # Combine our functions above to get the dataframe
    # we want
    oahu_2018 = data_to_df(f_2018, rel_2018, dates_rel_2018,
                              locs_real)
    oahu_2019 = data_to_df(f_2019, rel_2019, dates_rel_2019,
                              locs_real)
    
    # Let's combine these arrays
    oahu_all = {key: {} for key in locs_real.keys()}
    
    for loc in oahu_all.keys():
        oahu_all[loc] = pd.concat([oahu_2018, oahu_2019])
    
        
            
        
        
    

















