#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defining function to extract total wave energy flux around a point
along a circle defined
in the function call (radius, lat/long)
"""


import numpy as np
import xarray as xr
import geopandas as gpd
from polycircles import polycircles
from shapely import LineString



# To add-
# construct a better mapping around Oahu which filters for distance of <20 km
# from the shore and greater than 400 m depth

# in energy calculation- add a minimum energy and maximum energy extractable 



def energy_flux(raster, time, geopd_shape):
    """
    Extracts raster cell values in a circle around the defined lat/long
    at the specified time
    
    Returns the net energy flux in kW along the wave front
    
    What this entrire file builds towards!
    """
    rel_raster = raster.loc[{'step': time}]
    
    # TODO: Focus only on waves with a direction towards the shore (cos similarity?)
    # TODO: Buffer to 10 nautical miles for far-shore resources
    flux_per_m = extract_vals(rel_raster, geopd_shape)
    
    # Coaxing from NAD84 so we can get an accurate length value
    length = float(geopd_shape.to_crs('EPSG:3857').length)
    
    return length * flux_per_m


def draw_circle(latitude, longitude, radius):
    """
    Defines a Shapely circle nested in a geopandas linestring
    around our point in questions
    
    Returns the geopandas linestring
    """
    circle_coords = polycircles.Polycircle(latitude=latitude, 
                                           longitude=longitude,
                                           radius=radius).to_lat_lon()
    circle_coords = np.array(circle_coords)
    # Need to fix the coordinate pairs to get in correct orientation for circle
    temp = circle_coords[:, 0].copy()
    circle_coords[:, 0] = circle_coords[:, 1]
    circle_coords[:, 1] = temp

    # Shapely into geopandas!
    circle = LineString(circle_coords)
    circle = gpd.GeoSeries(circle)    
    
    return circle


def extract_vals(raster_array, line, n_samples=1000, power_range=[5, 30]):
    """
    Extracts raster along a defined shapely object
    Raster must be extracted to the set time
    
    Sets null values to zero for ease of calculation
    
    Returns the mean of the raster values along the profile
    """
    global profile
    profile = np.zeros(n_samples)
    shapely_circle = line.loc[0]
    
    for n in range(n_samples):
        # Get next point in the line
        point = shapely_circle.interpolate(n / n_samples - 1, normalized=True)
        # Access nearest xarray pixel
        value = raster_array.sel(latitude=point.y, longitude=point.x,
                                 method='nearest').data
        profile[n] = float(value)
    profile = np.nan_to_num(profile)
    
    # Restrict the range of power to between provided values
    profile[profile < power_range[0]] = 0
    profile[profile > power_range[1]] = power_range[1]
    
    return np.mean(profile)
