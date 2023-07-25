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
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from math import sin, cos


n_samples = 100


# In the future I should add a try/except line to check for buffer's pickle
# presence load to reduce time on startup



def energy_flux(nrg_rast, geopd_shape, shore_vect, wave_vect):
    """
    Extracts raster cell values in a shape around the defined lat/long
    
    Returns the net energy flux in kW along the wave front
    
    What this entrire file builds towards!
    """    
    # Extract energy values from the raster
    flux_arr = extract_vals(nrg_rast, geopd_shape)
    # Get the corresponding cosine similarity
    cos_same = cos_similar(shore_vect, wave_vect)
    
    # Multiply to get the inward pointing wave energy
    flux_arr = flux_arr * cos_same
    # Set the max/min obtainable energy- hard coded
    flux_arr[flux_arr < 5] = 0
    flux_arr[flux_arr > 30] = 30
    
    flux_per_m = np.mean(flux_arr)
    
    # Coaxing from NAD83 so we can get an accurate length value
    length = float(geopd_shape.to_crs('EPSG:3857').length)
    
    # Retutning a number in kW!
    return flux_per_m * length


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


def vect_to_shore(oahu_bounds, exterior_bounds, n_samples=n_samples):
    """
    Draws a vector pointing from the boundary linestring to the shoreline
    """
    # Let's create the same number of vector points as samples taken
    # to extract values, hopefully the arrays will match up nicely
    vectors = [None for _ in range(n_samples)]
    iter_nums = [n / (n_samples - 1) for n in range(n_samples)]
    
    perimeter = exterior_bounds.loc[0]
    state_bound = oahu_bounds[0]
    
    for count, value in enumerate(iter_nums):
        # Get next point in line
        point = perimeter.interpolate(value, normalized=True)
        # Get the nearest point
        point_2 = nearest_points(state_bound, point)[0]
        # Create the line joining the two and save it
        line = LineString([point, point_2])
        vectors[count] = line
        
    vectors = gpd.GeoSeries(vectors)
    
    return vectors.set_crs('EPSG:4269')

        
def plot_shore_vects(oahu_bounds, exte_bounds, vectors):
    """
    Plots the vectors connecting our bounds to the shore
    
    Mostly for my own reference to double-check my own work
    """
    fig = plt.figure(3, figsize=(5, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.coastlines()
    vectors.plot(ax=ax, color='red', markersize=0.03)
    oahu_bounds.plot(ax=ax, color='blue', markersize=0.5)
    exte_bounds.plot(ax=ax, color='darkblue', markersize=0.5)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, 
                      y_inline=False, alpha=0.5, linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    plt.show()


def north_vects(vect_list):
    """
    Defines a new set of vectors with the same starting point but
    now just point upwards towards the north pole
    
    Will use to get our new angles
    """
    vectors = [None for _ in range(len(vect_list))]
    
    for n, line in enumerate(vect_list):
        point_1 = line.coords[0]
        point_2 = point_1[0], point_1[1] + 0.1
    
        north = LineString([point_1, point_2])
        vectors[n] = north
    
    vectors = gpd.GeoSeries(vectors)
    
    return vectors.set_crs('EPSG:4269')
    

def vect_angle(vects_1, vects_2):
    """
    Calculates the angle between two sets of vectors, intended to be the 
    ones pointing to shore and the ones pointing north
    """
    angles = np.zeros(vects_1.size)
    
    for n, vec_1, vec_2 in zip(range(vects_1.size), vects_1, vects_2):
        # We need to set the intersection point as our origin
        # otherwise we are looking between vectors starting near africa
        vec_1 = np.array(vec_1.coords)
        vec_2 = np.array(vec_2.coords)
        
        origin = vec_1[0]        
        real_vec_1 = vec_1[1] - origin
        real_vec_2 = vec_2[1] - origin
        
        # Dot product and normalize!
        product = real_vec_1 * real_vec_2
        norm = np.linalg.norm(real_vec_1) * np.linalg.norm(real_vec_2)
        
        angles[n] = np.sum(product) / norm
        
    return np.arccos(angles)


def wave_vector(wave_angle, shore_vect):
    """
    Defines a new vector pointing in the wave direction along the 
    shore vectors
    """
    vectors = [None for _ in range(wave_angle.size)]
    wave_angle = np.radians(wave_angle)
    
    for n, angle, line in zip(range(wave_angle.size), wave_angle, shore_vect):
        point_1 = line.coords[0]
        point_2 = [point_1[0] + (0.1 * sin(angle)), 
                   point_1[1] + (0.1 * cos(angle))] 
    
        north = LineString([point_1, point_2])
        vectors[n] = north
    
    vectors = gpd.GeoSeries(vectors)
    
    return vectors.set_crs('EPSG:4269')
    

def cos_similar(shore_vect, wave_vect):
    """
    Calculates the cosime similarity between the shore vector and 
    the wave vector
    """
    similarity = np.zeros(shore_vect.size)
    
    for n, vect1, vect2 in zip(range(shore_vect.size), shore_vect, wave_vect):
        # Get coordinate points
        vect1 = np.array(vect1.coords)
        vect2 = np.array(vect2.coords)
        # Move to origin
        origin = vect1[0]
        vect1 = vect1[1] - origin
        vect2 = vect2[1] - origin
        # calculate the cosime simularity
        prod = vect1.dot(vect2)
        norm = np.linalg.norm(vect1) * np.linalg.norm(vect2)
        similarity[n] = prod / norm
    
    return np.abs(similarity)


def extract_vals(raster_array, line, n_samples=n_samples):
    """
    Extracts raster along a defined shapely object
    Raster must be extracted to the set time
    
    Sets null values to zero for ease of calculation
    
    Original: Returns the mean of the raster values along the profile
    
    Now: returns the profile itself so I can do some calculations
    """
    profile = np.zeros(n_samples)
    shapely_circle = line.loc[0]
    
    for n in range(n_samples):
        # Get next point in the line
        point = shapely_circle.interpolate(n / (n_samples - 1), normalized=True)
        # Access nearest xarray pixel
        value = raster_array.sel(latitude=point.y, longitude=point.x,
                                 method='nearest').data
        profile[n] = float(value)
    profile = np.nan_to_num(profile)
    
    return profile
    