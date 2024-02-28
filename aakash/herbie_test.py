#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Herbie
"""


from herbie import Herbie
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

H = Herbie(
    "2023-06-10",
    model="ecmwf",
    product='oper',
)


oahu_dims = {'max_lon': -151.5,
             'min_lon': -162.5,
             'min_lat': 16.5,
             'max_lat': 23.5}

def crop_raster(rast, dims):
    """
    Crops raster to provided dims
    """
    rast = rast.loc[dict(latitude=slice(dims['max_lat'], dims['min_lat']))]
    rast = rast.loc[dict(longitude=slice(dims['min_lon'], dims['max_lon']))]
    
    return rast

def fix_coords(df):
    """
    Converts between native gfs degrees east to normal lat/long
    """
    df.coords['longitude'] = (df.coords['longitude'] + 180) % 360 - 180
    df = df.sortby(df.longitude)
    
    return df
    

# Download global data
u_winds_80m = H.xarray(":10u:" , remove_grib=True)
v_winds_80m = H.xarray(":10v:",  remove_grib=True)

temp_80m = H.xarray(":2t:", remove_grib=True)

# Fix coords
u_winds_80m = fix_coords(u_winds_80m)
v_winds_80m = fix_coords(v_winds_80m)
temp_80m = fix_coords(temp_80m)


# Crop to Oahu
u_winds_80m_oahu = crop_raster(u_winds_80m, oahu_dims)
v_winds_80m_oahu = crop_raster(v_winds_80m, oahu_dims)

temp_80m_oahu = crop_raster(temp_80m, oahu_dims)

# Plot temp

levels = [15, 18, 21, 24, 27, 30]


fig = plt.figure(2, figsize=(5, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = plt.contourf(temp_80m_oahu.longitude, temp_80m_oahu.latitude,
             temp_80m_oahu.t2m - 273.15, transform=ccrs.PlateCarree(), 
             levels=levels)
ax.coastlines()

proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_facecolor()[0]) for
         pc in plot.collections]
ax.legend(proxy, ['15-18 C', '18-21 C', '21-24 C',
                  '24-27 C', '27-30 C'], loc='lower left')
ax.set_title('2 meter Temp Over Hawaii, ECMWF')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  alpha=0.5, linestyle='--')

gl.top_labels = False
gl.right_labels = False
plt.show()
