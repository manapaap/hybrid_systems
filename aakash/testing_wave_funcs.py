# -*- coding: utf-8 -*-
"""
Testing the wave files
"""

from os import chdir
import numpy as np
from polycircles import polycircles
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
import rioxarray as rio
import cartopy.crs as ccrs
import geopandas as gpd
from shapely import LineString


chdir('/Users/amanapat/Documents/hybrid_systems/')


min_lat = 17
max_lat = 24

min_lon = -162
max_lon = -153



beep = xr.open_dataset('raw_data/wave_data_swan/sig_height/' +
                        '/multi_1.wc_4m.hs.201905.grb2.txt',
                        engine='cfgrib')

# Fixing the weird way the coordinates are read to correct values
coords_right = np.array(beep.longitude)[::-1] * -1 + 79
beep = beep.drop_vars('longitude')
beep = beep.assign_coords({'longitude': coords_right})

# To a useable format
boop = beep.to_array().squeeze()

# Crop to just Hawaii
boop = boop.loc[dict(latitude=slice(max_lat, min_lat))]
boop = boop.loc[dict(longitude=slice(min_lon, max_lon))]


# If I want to visualize a single slice of time

selection = np.array(boop.step)[0]
time_slice = boop.loc[{'step': selection}]

levels = [0, 0.5, 1, 1.5, 2, 2.5, 3]

fig = plt.figure(1, figsize=(5, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = plt.contourf(time_slice.longitude, time_slice.latitude,
             time_slice.variable, transform=ccrs.PlateCarree(),
             levels=levels)

proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
         pc in plot.collections]

ax.coastlines()
ax.legend(proxy, ['0-0.5 m', '0.5-1 m', '1-1.5 m',
                  '1.5-2 m', '2-2.5 m', '2.5-3 m'], loc='lower left')
ax.set_title('Significant Wave and Swell Heights, June 2019')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  alpha=0.5, linestyle='--')

gl.top_labels = False
gl.right_labels = False
plt.show()


# Let's do the same for the mean period

beep2 = xr.open_dataset('raw_data/wave_data_swan/sig_period/' +
                        '/multi_1.wc_4m.tp.201905.grb2',
                        engine='cfgrib')

# Fixing the weird way the coordinates are read to correct values
coords_right = np.array(beep2.longitude)[::-1] * -1 + 79
beep2 = beep2.drop_vars('longitude')
beep2 = beep2.assign_coords({'longitude': coords_right})

# To a useable format
boop2 = beep2.to_array().squeeze()

# Crop to just Hawaii
boop2 = boop2.loc[dict(latitude=slice(max_lat, min_lat))]
boop2 = boop2.loc[dict(longitude=slice(min_lon, max_lon))]


# If I want to visualize a single slice of time

selection = np.array(boop2.step)[0]
time_slice2 = boop2.loc[{'step': selection}]

levels = [0, 3, 6, 9, 12, 15, 18]

fig = plt.figure(2, figsize=(5, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = plt.contourf(time_slice2.longitude, time_slice2.latitude,
             time_slice2.variable, transform=ccrs.PlateCarree(),
             levels=levels)

proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
         pc in plot.collections]

ax.coastlines()
ax.legend(proxy, ['0-3s', '3-6 s', '6-9 s',
                  '9-12 s', '12-15 s', '15-18 s'], loc='lower left')
ax.set_title('Mean Wave Period, June 2019')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  alpha=0.5, linestyle='--')

gl.top_labels = False
gl.right_labels = False
plt.show()


# Let's now get the energy flux per unit wave crest

flux_arr = 0.491 * (boop**2) * boop2

# Add a circle around Oahu to capture the raster cell values

circle_coords = polycircles.Polycircle(latitude=21.43, 
                                       longitude=-158,
                                       radius=50000).to_lat_lon()
circle_coords = np.array(circle_coords)
# Need to fix the coordinate pairs to get in correct orientation for circle
temp = circle_coords[:, 0].copy()
circle_coords[:, 0] = circle_coords[:, 1]
circle_coords[:, 1] = temp

# Shapely into geopandas!
circle = LineString(circle_coords)
circle = gpd.GeoSeries(circle)


# Visualize this to compare to the marine energy atlas and check circle
# dimentions


selection = np.array(flux_arr.step)[0]
time_slice3 = flux_arr.loc[{'step': selection}]

levels = [0, 10, 20, 30, 40, 50, 60]

fig = plt.figure(2, figsize=(5, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = plt.contourf(time_slice3.longitude, time_slice3.latitude,
             time_slice3.variable, transform=ccrs.PlateCarree(),
             levels=levels)

proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
         pc in plot.collections]

circle.plot(ax=ax, color='red')

ax.coastlines()
ax.legend(proxy, ['0-10 kW/m', '10-20 kW/m', '20-30 kW/m',
                  '30-40 kW/m', '40-50 kW/m', '50-60 kW/m'], loc='lower left')
ax.set_title('Wave Energy Flux, June 2019')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  alpha=0.5, linestyle='--')

gl.top_labels = False
gl.right_labels = False
plt.show()




