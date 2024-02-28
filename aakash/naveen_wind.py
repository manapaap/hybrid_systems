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
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from os import chdir


chdir('/Users/amanapat/Documents/hybrid_systems/')


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
    dset_coords = return_data(f, 'coordinates', arr=True)
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
        

def return_data(f, field, m=0, arr=False):
    """
    Uses the h5pyd API to query data, but as that occassionally
    fails, this recursive function will ensure that 
    it finds the data when we know the path is correct
    
    Still raises the error after unreasonable number of iters
    """
    try:
        if arr:
            return np.array(f[field])
        else:
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
    
    result_dict = {key: {} for key in locs_dict.keys()}
    
    for loc, value in locs_dict.items():
        # Get the elev vs. time arrays for all locs
        wind_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='windspeed')
        temp_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='temperature')
        dir_arr = wind_points(f, wind_heights, filter_year, 
                               value['idx'], field='winddirection')
        pres_arr = wind_points(f, pres_heights, filter_year, 
                               value['idx'], field='pressure')
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
        result_dict[loc] = pd.DataFrame(result_dict[loc])
        result_dict[loc].index = dates
        
    return result_dict
 

def format_df(dict_df):
    """
    Breaks up pandas datetime64 index into year, month, day...
    columns to reindex
    
    renames other columns 
    
    Does it by the whole dict
    """
    for key in dict_df.keys():
        dict_df[key] = dict_df[key].rename(columns={
            'wind_87.6': 'wind speed', 'wind_161': 'wind speed',
            'temp_87.6': 'temperature', 'temp_161': 'temperature',
            'pres_87.6': 'pressure', 'pres_161': 'pressure',
            'dir_87.6': 'degrees true', 'dir_161': 'degrees true'})
        # I tried to do this intelligently but it didn't matter
        # so its a manual mess
        dict_df[key]['Year'] = dict_df[key].index.year
        dict_df[key]['Month'] = dict_df[key].index.month
        dict_df[key]['Day'] = dict_df[key].index.day
        dict_df[key]['Hour'] = dict_df[key].index.hour
        dict_df[key]['Minute'] = dict_df[key].index.minute
        # Remove the original index
        dict_df[key].reset_index(inplace=True, drop=True)
    
    return dict_df


def loc_docstr(loc_dict):
    """
    Creates the docstring for the csv containing the info
    that we want
    
    loc_dict is of form locs_real['Oahu_N']
    """
    # Add in the intro string that he wants
    line_1 = 'Source,Location ID,Jurisdiction,Latitude,Longitude,' +\
             'Time Zone,Local Time Zone,Distance to Shore,' +\
             'Wind Direction,Windspeed,' +\
             'Pressure,Temperature,' +\
             'Version\n'
             
    # Coords formatted for each location
    line_2 = 'Wind Integration National Toolkit,-,Hawaii,' +\
             f'{loc_dict["coords"][0]},' +\
             f'{loc_dict["coords"][1]},0,-10,-,deg,m/s,' +\
             'mB,°C,v1.00\n{}'
                    
    return line_1 + line_2


def write_csv_doc(df, loc_dict, fname):
    """
    Writes the wind information to a CSV along with the docstring
    containing information about the same
    """
    template = loc_docstr(loc_dict)
    
    with open('mid_data/' +  fname + '.csv', 'w') as fp:
        fp.write(template.format(df.to_csv(index=False)))


def main():
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
        oahu_all[loc] = pd.concat([oahu_2018[loc],
                                   oahu_2019[loc]])
        
    # Separate this by the elevation
    oahu_87 = {key: pd.DataFrame() for key in oahu_all.keys()}
    oahu_161 = {key: pd.DataFrame() for key in oahu_all.keys()}
    for loc, value in oahu_all.items():
        for col in value.columns:
            if '87.6' in col:
                oahu_87[loc][col] = oahu_all[loc][col]
            elif '161' in col:
                oahu_161[loc][col] = oahu_all[loc][col]
    
    # Rename to get into formatting we need
    oahu_87 = format_df(oahu_87)
    oahu_161 = format_df(oahu_161)
    
    # Output our files!
    for key, value in oahu_87.items():
        fname = 'wind_resource_' + key + '_87.6m'
        write_csv_doc(value, locs_real[key], fname)
        
    for key, value in oahu_161.items():
        fname = 'wind_resource_' + key + '_161m'
        write_csv_doc(value, locs_real[key], fname)
    

if __name__ == '__main__':
    main()
