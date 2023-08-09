#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots ana general analysis of the time series wave data
"""


import pandas as pd
import numpy as np
from os import chdir
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from scipy.interpolate import make_interp_spline, BSpline, pchip_interpolate


chdir('/Users/amanapat/Documents/hybrid_systems/')

# TODO: daily averages given specific months


def load_data():
    """
    Loads in our time series wave data
    """
    full_year_data = pd.read_csv('mid_data/wave_nrg_ALL.csv')
    five_year_data = pd.read_csv('mid_data/wave_nrg_5_year.csv')
    
    full_year_data['time'] = pd.to_datetime(full_year_data['time'])
    five_year_data['time'] = pd.to_datetime(five_year_data['time'])
    
    full_year_data = full_year_data.set_index('time')
    five_year_data = five_year_data.set_index('time')
    
    full_year_data.drop(full_year_data.columns[0], axis=1, inplace=True)
    five_year_data.drop(five_year_data.columns[0], axis=1, inplace=True)
    
    el_nino_long = pd.read_excel('raw_data/el_nino_all.xlsx')
    el_nino = format_ONI(el_nino_long)
    
    # Truncating the full data due to weird zero values for first two days
    return full_year_data[13:], five_year_data, el_nino


def plot_daily_avg(daily_avg, days, fig=1):
    """
    Plots the annual trend for average days across the time period
    """
    base = np.datetime64('2020-01-01')
    date_list = [base + np.timedelta64(x, 'D') for x in 
                 range(0, days.size)]
    
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    
    plt.figure(fig)
    plt.title('Daily Average of Wave Energy Flux, 2005-2019')
    plt.grid()
    plt.plot(date_list, daily_avg)
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel('Month')
    plt.ylabel('Wave Energy Potential (kW)')
    plt.show()
    

def format_ONI(el_nino_long):
    """
    Formats the el nino data from wide to long
    """
    data_size = (el_nino_long.shape[1] - 1) * (el_nino_long.shape[0] - 1)
    values = [None for _ in range(el_nino_long.shape[0])]
    
    for n in range(len(values)):
        # Extract the rows by index and store in time series
        values[n] = np.array(el_nino_long.iloc[n][1:])
    
    values = np.concatenate(values)
    values[values == -9.99e+01] = np.nan
    
    base = np.datetime64('1950-01')
    date_list = [base + np.timedelta64(x, 'M') for x in range(values.size)]
    
    el_nino = pd.DataFrame({'time': date_list,
                            'oni': values})
    el_nino.set_index('time', inplace=True)
    return el_nino


def compare_oni_nrg(wave_nrg, oni):
    """
    Plots a comparision of the ONI and monthly average of wave energy
    """
    min_date = wave_nrg.index[0] - np.timedelta64(5, 'D')
    max_date = wave_nrg.index[-1]
    
    oni_rel = oni.loc[min_date:max_date]
    
    fig, ax1 = plt.subplots()
    ax1.plot(oni_rel.index, oni_rel.oni, label='ONI', color='darkblue')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Oceanic Nino Index')
    ax1.grid()
    
    wave_nrg = wave_nrg.groupby([wave_nrg.index.year,
                                 wave_nrg.index.month]).mean()
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Wave Energy Potential (kW)')
    ax2.plot(oni_rel.index, wave_nrg.far_nrg,
             label='Far Energy', color='red')
    plt.legend()
    
    wave_nrg['time'] = oni_rel.index
    wave_nrg.set_index('time', inplace=True)
    
    return oni_rel, wave_nrg


def daily_seasonal_avg(full_data):
    """
    Plots an averaged daily time series for all seasons
    """
    # Select the seasonal averages
    winter = pd.concat([full_data[(full_data.index.month == 12)],
                        full_data[(full_data.index.month == 1)],
                        full_data[(full_data.index.month == 2)]])
    spring = pd.concat([full_data[(full_data.index.month == 3)],
                        full_data[(full_data.index.month == 4)],
                        full_data[(full_data.index.month == 5)]])
    summer = pd.concat([full_data[(full_data.index.month == 6)],
                        full_data[(full_data.index.month == 7)],
                        full_data[(full_data.index.month == 8)]])
    fall = pd.concat([full_data[(full_data.index.month == 9)],
                      full_data[(full_data.index.month == 10)],
                      full_data[(full_data.index.month == 11)]])
    # Sort the messy resultant values
    winter.sort_values(by='time', inplace=True)
    spring.sort_values(by='time', inplace=True)
    summer.sort_values(by='time', inplace=True)
    fall.sort_values(by='time', inplace=True)
    # Calculate the daily averages 
    winter = winter.groupby(winter.index.hour).mean()
    spring = spring.groupby(spring.index.hour).mean()
    summer = summer.groupby(summer.index.hour).mean()
    fall = fall.groupby(fall.index.hour).mean()
    # Create our time axis
    base = np.datetime64('2020-01-01 00:00:00')
    time = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    new_time = [base + np.timedelta64(x, 'm') for x in range(0, 1440, 30)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    
    k_spline = 3
    
    # Initiate our subplots
    
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    fig.suptitle('Seasonal Daily Wave Energy Potential')
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, 'Wave Energy Potential (kW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    # Create a spline to fit the low res data
    smooth = pchip_interpolate(time,
                               winter.far_nrg.append(pd.Series(winter.far_nrg[0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 0].plot(time[:-1], winter.far_nrg)
    ax[0, 0].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
    """ retaining the old interpolation just to see
    spl = make_interp_spline(time, 
                             spring.far_nrg.append(pd.Series(spring.far_nrg[0])), 
                             k=k_spline)
    smooth = spl(new_time)
    """
    smooth = pchip_interpolate(time,
                               spring.far_nrg.append(pd.Series(spring.far_nrg[0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 1].plot(time[:-1], spring.far_nrg)
    ax[0, 1].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    smooth = pchip_interpolate(time,
                               summer.far_nrg.append(pd.Series(summer.far_nrg[0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 0].plot(time[:-1], summer.far_nrg, label='Energy')
    ax[1, 0].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8, label='Interpolated')
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    ax[1, 0].legend(loc='lower left')
    
    # PLOTTING Fall
    smooth = pchip_interpolate(time,
                               fall.far_nrg.append(pd.Series(fall.far_nrg[0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 1].plot(time[:-1], fall.far_nrg)
    ax[1, 1].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    
    return winter, spring, summer, fall
    

def main():
    full_data, five_data, el_nino = load_data()
    
    daily_avg_five = five_data.groupby([five_data.index.month, 
                                        five_data.index.day]).mean()
    daily_avg_full = full_data.groupby([full_data.index.month, 
                                        full_data.index.day]).mean()
    
    days = np.arange(1, 367, step=1)
    
    plot_daily_avg(daily_avg_full, days, fig=1)
    rel_oni, monthly_avg = compare_oni_nrg(full_data, el_nino)
    season_avg = daily_seasonal_avg(full_data)
    
    # let's create an index for the daily/monthly averages
    start = np.datetime64('2020-01-01')
    end = np.datetime64('2021-01-01')
    full_idx = np.arange(start, end, np.timedelta64(1, 'D'))
    
    daily_avg_five.index = full_idx
    daily_avg_full.index = full_idx
    
    
if __name__ == '__main__':
    main() 

