#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that defines a function that gives us a linestring around Oahu defining 
two boundaries- the nearshore boundary (depth > 400 m) and the far shore 
boundary (< 10 nautical miles from shore)
"""

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from os import chdir
import numpy as np
import scipy.ndimage
import geopandas as gpd
import alphashape
from shapely import Polygon
from shapely.ops import polygonize
import sys


chdir('C:\\Users\\Aakas\\Documents\\Grad School\\NREL\\')
sys.path.append("aakash\\")


oahu_dims = {'min_lon': -158.5,
             'max_lon': -157.52,
             'min_lat': 21.185,
             'max_lat': 21.8}


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
    
    # bounds_test = MultiPoint(pairs)
    # bounds_test = gpd.GeoSeries(bounds)
    
    return bounds


def oahu_bounds_near_shore(path_to_file, min_depth=-400):
    """
    Simple function that puts together the calc_boundary and crop_data
    functions to allow for simple access to the boundary linestring file
    in subsequent programs
    
    Parameter: raster of coastal relief around oahu
    
    Returns the boundary around Oahu that defines the perimeter for obtainable
    wave energy by virtue of depth
    """
    elevation = xr.open_dataset('raw_data/coast_elevation/crm_vol10.nc')

    oahu_dims = {'min_lat': 21.185,
                 'max_lat': 21.8,
                 'min_lon': -158.5,
                 'max_lon': -157.52}
    
    _, inland = crop_depth_data(elevation, dims=oahu_dims, min_depth=min_depth)
    
    bounds = calc_boundary(inland)
    
    # setting CRS based on the original NOAA crs
    bounds = bounds.set_crs('EPSG:4269')
    
    return bounds
    

def load_oahu_shape(path_to_file):
    """
    Loads the Hawaii shape polygon and truncates to Oahu
    
    Parameter: coastlines shapefile
    
    Does mean this code needs an internet connection to work but oh well
    """
    usa_bounds = gpd.read_file(path_to_file)
    
    crop_bounds = Polygon([(oahu_dims['min_lon'], oahu_dims['min_lat']),
                          (oahu_dims['min_lon'], oahu_dims['max_lat']),
                          (oahu_dims['max_lon'], oahu_dims['max_lat']),
                          (oahu_dims['max_lon'], oahu_dims['min_lat'])])
    
    # This is much better than the US module but it includes a bunch of
    # tiny islands around Oahu that we don't care about and fuck up the buffer
    # Let's polygonize this and hard code selecting the right polygon
    # to prevent this issue. Inelegant but it doesn't matter anyway
    
    oahu_all = usa_bounds.clip(crop_bounds)
    oahu_list = gpd.GeoSeries(polygonize(list(oahu_all['geometry'])))
    oahu_poly = gpd.GeoSeries(oahu_list[8])
    oahu_bounds = oahu_poly.exterior
    
    return oahu_bounds.set_crs('EPSG:4269')


def oahu_shape_simple(oahu_bounds, alpha=10):
    """
    Draws the concave hull around oahu for a simpler representation
    of the shoreline
    """
    point_coords = np.array(oahu_bounds[0].coords)
    shape = alphashape.alphashape(point_coords, alpha)
    shape = gpd.GeoSeries(shape.exterior)
    
    return shape.set_crs('EPSG:4269')
    

def oahu_bounds_far_shore(path_to_file, buffer=10):
    """
    Takes the state bounds, buffers it by 10 nautical miles, returns crs
    to NAD83, and returns the buffered object to be used for wave analysys
    
    Parameter: coastlines shapefile
    """
    oahu = load_oahu_shape(path_to_file)
    
    oahu = oahu.to_crs('EPSG:3857')
    
    # Converting the nautical mile value to meters
    oahu = oahu.buffer(distance=(buffer * 1852)).exterior
    
    return oahu.to_crs('EPSG:4269')
    

def main():
    """
    This file is primarily for importing the calc boundary function, but
    I wanted to retain some plotting in this file to double check the effects
    of changing depth etc. at a later point
    """
    
    # First plot for near-shore calculations
    
    elevation = xr.open_dataset('raw_data/coast_elevation/crm_vol10.nc')
    
    elevation, _ = crop_depth_data(elevation, dims={'min_lat': 21.185,
                                             'max_lat': 21.8,
                                             'min_lon': -158.5,
                                             'max_lon': -157.52})
    bounds = oahu_bounds_near_shore('raw_data/coast_elevation/crm_vol10.nc',
                                    min_depth=-400)
    
    
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


    # Second plot for far shore calculations
    
    far_bounds = oahu_bounds_far_shore('raw_data/hawaii_bounds/' +
                                       'tl_2019_us_coastline.zip',
                                       buffer=10)
    
    fig = plt.figure(3, figsize=(5, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    far_bounds.plot(ax=ax, color='red', markersize=0.03)
    ax.coastlines()

    ax.set_title('Far Shore Marine Energy, 10 nautical mile buffer')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      alpha=0.5, linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    plt.show()
    
    
if __name__ == '__main__':
    main()
