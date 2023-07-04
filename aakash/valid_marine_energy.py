#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the loading of elevation netCDF files

Let's also incorporate the oahu region, crop to that boundary, and then
investigate

Using the parameter values from levi's 2016 wave energy assessment
"""

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from os import chdir
import numpy as np
import scipy.ndimage
from shapely import MultiPoint
import geopandas as gpd
import alphashape


chdir('/Users/amanapat/Documents/hybrid_systems/')



def crop_depth_data(rast, dims, min_depth=-400, max_depth=-5):
    """
    Crops the raster to the region defined by max long/lat, and filters
    for values between the maximum and minimum acceptable depths
    
    Returns values filterd for depths between min and max, and also values
    just above max to represent "inland" for calculation purposes
    """
    # Crop by location
    rast = rast.loc[dict(y=slice(dims['min_lat'], dims['max_lat']))]
    rast = rast.loc[dict(x=slice(dims['min_lon'], dims['max_lon']))]
    # Range of acceptable wave collector depths
    rast = rast.where(rast.z > min_depth)
    test_rast = rast.copy()
    rast = rast.where(rast.z < max_depth)
    
    return rast, test_rast


def coords_to_points(pairs):
    """
    Converts the N*2 numpy array of coordinates into a list of points
    """
    points = [tuple(pos) for pos in list(pairs)]
    return points


def calc_concave_hull(points, alpha):
    """
    Calculates the concave hull of the points provided and returns the line
    bounding the same in the form of a geoseries object
    """
    shape = alphashape.alphashape(points, alpha)
    shape = shape.exterior
    
    return gpd.GeoSeries(shape)


def calc_boundary(inland_rast):
    """
    Calculates the boundary pixel locations around our inland elevation raster
    and creates a shapely linestring to go around this boundary, packed into
    a geopandas geodataframe because I felt like it
    """
    elev_arr = np.flip(np.array(inland_rast.z), axis=0)
    lat_arr = np.flip(np.array(inland_rast.y))
    lon_arr = np.array(inland_rast.x)

    # Let's stack these to obtain the shape we need
    lon_len = len(lat_arr)
    lat_arr = np.column_stack([lat_arr for _ in range(len(lon_arr))])
    lon_arr = np.row_stack([lon_arr for _ in range(lon_len)])

    # Mask with the values from the elevation data array - process described below:
    # 1) Add an arbitrarily large number to remove issue of zero elevation inland
    filter_elevs = elev_arr + 10e6
    filter_elevs = np.nan_to_num(filter_elevs)
    # 2) we now have an array with 0 and large number, let's get this into binary mask
    filter_elevs = np.array(filter_elevs != 0, dtype=np.int64)
    # 3) Let's now do a binary erosion to remove one edge pixel row
    filter_erode = scipy.ndimage.binary_erosion(filter_elevs)
    # Logical xor with the original array to obtain the pixels on the boundary
    filter_elevs = np.logical_xor(filter_elevs, filter_erode)
    filter_elevs = np.array(filter_elevs, dtype=np.int64)

    # 4) Let's now filter our coordinates of intrest
    edge_lats = lat_arr[filter_elevs == 1]
    edge_lons = lon_arr[filter_elevs == 1]

    # 5) Assemble this into coordinate pairs and create our linestring
    pairs = (np.vstack([edge_lons, edge_lats])).T
    
    points = coords_to_points(pairs)
    bounds = calc_concave_hull(points, alpha=21)
    
    """
    bounds_test = MultiPoint(pairs)
    bounds_test = gpd.GeoSeries(bounds)
    """
    
    return bounds


def oahu_bounds(path_to_file):
    """
    Simple function that puts together the calc_boundary and crop_data
    functions to allow for simple access to the boundary linestring file
    in subsequent programs
    
    Returns the boundary around Oahu that defines the perimeter for obtainable
    wave energy by virtue of depth
    """
    elevation = xr.open_dataset('raw_data/coast_elevation/crm_vol10.nc')

    oahu_dims = {'min_lat': 21.185,
                 'max_lat': 21.8,
                 'min_lon': -158.5,
                 'max_lon': -157.52}
    
    _, inland = crop_depth_data(elevation, dims=oahu_dims)
    
    bounds = calc_boundary(inland)
    
    # setting CRS based on the original NOAA crs
    bounds = bounds.set_crs('EPSG:4269')
    
    return bounds
    
    
def main():
    """
    This file is primarily for importing the calc boundary function, but
    I wanted to retain some plotting in this file to double check the effects
    of changing depth etc. at a later point
    """
    elevation = xr.open_dataset('raw_data/coast_elevation/crm_vol10.nc')
    
    elevation, _ = crop_depth_data(elevation, dims={'min_lat': 21.185,
                                             'max_lat': 21.8,
                                             'min_lon': -158.5,
                                             'max_lon': -157.52})
    bounds = oahu_bounds('raw_data/coast_elevation/crm_vol10.nc')
    
    
    levels = [-400, -300, -200, -100, -5]
    
    fig = plt.figure(2, figsize=(5, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plot = plt.contourf(elevation.x, elevation.y,
                 elevation.z, transform=ccrs.PlateCarree(),
                 levels=levels)
    
    proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
             pc in plot.collections]
    
    bounds.plot(ax=ax, color='red', markersize=0.03)
    
    ax.coastlines()
    ax.legend(proxy, ['400-300 m', '300-200 m', '200-100 m',
                      '100-5 m'], loc='lower left')
    ax.set_title('Oahu Coastal Relief for Marine Energy')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      alpha=0.5, linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    plt.show()


if __name__ == '__main__':
    main()
