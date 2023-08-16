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
    
    wind_data = pd.read_csv('mid_data/wind_nrg_5_year.csv')
    wind_data['time'] = pd.to_datetime(wind_data['Unnamed: 0'])
    wind_data.set_index('time', inplace=True)
    wind_data.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Truncating the full data due to weird zero values for first two days
    return full_year_data[13:], five_year_data, el_nino, wind_data


def clean_load_data(oahu_loads):
    """
    Cleans the oahu data from wide to long format
    """
    start = np.datetime64('2016-01-01T00:00')
    end = np.datetime64('2045-12-31T23:00')
    time_arr = np.arange(start, end + np.timedelta64(1, 'h'),
                         np.timedelta64(1, 'h'))
    
    num_entries = 24 * len(oahu_loads)
    power_arr = np.zeros(num_entries)
    
    # Subset to the hourly columns
    rel_oahu = oahu_loads[[str(x) for x in range(1, 25)]]
    
    for n, n_month in enumerate(range(0, num_entries, 24)):
        power_arr[n_month:n_month + 24] = rel_oahu.loc[n]
        
    oahu_loads = pd.DataFrame({'nrg_use': power_arr})
    oahu_loads.index = time_arr
    
    return oahu_loads


def load_data_2():
    """
    Loads in some of the other data that was too long for the other func
    """
    oahu_loads = pd.read_csv('raw_data/hawaii_load_ref_long.csv')
    oahu_loads = clean_load_data(oahu_loads)
    
    return oahu_loads


def plot_daily_avg(daily_avg, days, title, fig=1, nrg='Wave'):
    """
    Plots the annual trend for average days across the time period
    """
    base = np.datetime64('2020-01-01')
    date_list = [base + np.timedelta64(x, 'D') for x in 
                 range(0, days.size)]
    
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    
    plt.figure(fig)
    plt.title(title)
    plt.grid()
    plt.plot(date_list, daily_avg)
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel('Month')
    plt.ylabel(f'{nrg} Energy Potential (kW)')
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


def daily_seasonal_avg(full_data, energy_type, nrg):
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
    fig.suptitle(f'Seasonal Daily {energy_type} Energy Potential')
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, '{energy_type} Energy Potential (kW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    # Create a spline to fit the low res data
    smooth = pchip_interpolate(time,
                               winter[nrg].append(pd.Series(winter[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 0].plot(time[:-1], winter[nrg])
    ax[0, 0].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
    """ retaining the old interpolation just to see
    spl = make_interp_spline(time, 
                             spring[nrg]append(pd.Series(spring[nrg][0])), 
                             k=k_spline)
    smooth = spl(new_time)
    """
    smooth = pchip_interpolate(time,
                               spring[nrg].append(pd.Series(spring[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 1].plot(time[:-1], spring[nrg])
    ax[0, 1].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    smooth = pchip_interpolate(time,
                               summer[nrg].append(pd.Series(summer[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 0].plot(time[:-1], summer[nrg], label='Energy')
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
                               fall[nrg].append(pd.Series(fall[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 1].plot(time[:-1], fall[nrg])
    ax[1, 1].plot(new_time, smooth, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    
    return winter, spring, summer, fall


def daily_seasonal_avg_no_interp(full_data, energy_type, nrg):
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
    time = [base + np.timedelta64(x, 'h') for x in range(0, 24, 1)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    
    # Initiate our subplots
    
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    fig.suptitle(f'Seasonal Daily {energy_type} Energy Potential')
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, f'{energy_type} Energy Potential (kW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    ax[0, 0].plot(time, winter[nrg])
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
      
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    ax[0, 1].plot(time, spring[nrg])
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    ax[1, 0].plot(time, summer[nrg], label='Energy')
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    
    # PLOTTING Fall
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    ax[1, 1].plot(time, fall[nrg])
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    
    return winter, spring, summer, fall


def plot_nrg_mix(renew_data):
    """
    Plots the energy mix along with the net energy consumption for Oahu
    for the available years of data
    """
    # Downsample to daily freq for readability 
    renew_data = renew_data.groupby([renew_data.index.month,
                                     renew_data.index.day]).mean()
    
    days = np.arange(1, 367, step=1)
    base = np.datetime64('2020-01-01')
    date_list = [base + np.timedelta64(x, 'D') for x in 
                 range(0, days.size)]
    
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    
    plt.figure(np.random.randint(0, 10000000))
    plt.plot(date_list, renew_data['nrg_use'],
             label='Energy Consumption', color='red')
    plt.plot(date_list, renew_data['wind_real'],
             label='Wind Energy', color='grey')
    plt.plot(date_list, renew_data['wave_real'] + renew_data['wind_real'],
             label='Wind plus Waves', color='blue')
    plt.legend(loc='lower center')
    plt.grid()
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel('Month')
    plt.ylabel('Power (MW)')
    plt.title('Energy Mix vs. Consumption in Oahu')
    

def plot_daily_season_avg(renew_mix):
    """
    Plots the daily seasonal averages for energy consumption
    """
    # Select the seasonal averages
    winter = pd.concat([renew_mix[(renew_mix.index.month == 12)],
                        renew_mix[(renew_mix.index.month == 1)],
                        renew_mix[(renew_mix.index.month == 2)]])
    spring = pd.concat([renew_mix[(renew_mix.index.month == 3)],
                        renew_mix[(renew_mix.index.month == 4)],
                        renew_mix[(renew_mix.index.month == 5)]])
    summer = pd.concat([renew_mix[(renew_mix.index.month == 6)],
                        renew_mix[(renew_mix.index.month == 7)],
                        renew_mix[(renew_mix.index.month == 8)]])
    fall = pd.concat([renew_mix[(renew_mix.index.month == 9)],
                      renew_mix[(renew_mix.index.month == 10)],
                      renew_mix[(renew_mix.index.month == 11)]])
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
    time = [base + np.timedelta64(x, 'h') for x in range(0, 24, 1)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    
    # Initiate our subplots
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 5), sharey=True)
    fig.suptitle('Seasonal Daily Average Energy Use')
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, 'Energy (MW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    ax[0, 0].plot(time, winter['nrg_use'], color='red')
    ax[0, 0].plot(time, winter['wind_real'], color='grey')
    ax[0, 0].plot(time, winter['wind_real'] + winter['wave_real'],
                  color='blue')
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    ax[0, 1].plot(time, spring['nrg_use'], color='red')
    ax[0, 1].plot(time, spring['wind_real'], color='grey')
    ax[0, 1].plot(time, spring['wind_real'] + spring['wave_real'],
                  color='blue')
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    ax[1, 0].plot(time, summer['nrg_use'], 
                  label='Energy Use', color='red')
    ax[1, 0].plot(time, summer['wind_real'], label='Wind Energy', color='grey')
    ax[1, 0].plot(time, summer['wind_real'] + summer['wave_real'],
                  label='Wind plus Waves', color='blue')
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    ax[1, 0].legend(loc='upper left')
    
    # PLOTTING Fall
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    ax[1, 1].plot(time, fall['nrg_use'], color='red')
    ax[1, 1].plot(time, fall['wind_real'], color='grey')
    ax[1, 1].plot(time, fall['wind_real'] + fall['wave_real'], color='blue')
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    

def plot_random_day(renew_anal, day=''):
    """
    Plots a random day's power cycle or allows for date selection
    """
    if day == '':
        day = renew_anal.index[np.random.randint(0, len(renew_anal))]
    
    plot_data = 'mew'



def print_info(renew_anal):
    """
    Prints some useful information/statistics about the energy production data
    """
    print()
    print()
    print('Energy overproduction occurs ' +
          f'{100 * renew_anal["overprod"].sum() / len(renew_anal):.2f}%' + 
          ' of time')
    print(f'Mean overproduction is {abs(renew_anal.query(f"overprod == {True}")["nrg_diff"].mean()):.2f} MW')
    print(f'Mean underproduction is {renew_anal.query(f"overprod == {False}")["nrg_diff"].mean():.2f} MW')
    print()
    print(f'Maximum underproduction is {renew_anal["nrg_diff"].max():.2f} MW on {renew_anal.index[renew_anal["nrg_diff"].argmax()]}')
    print(f'Maximum overproduction is {abs(renew_anal["nrg_diff"].min()):.2f} MW on {renew_anal.index[renew_anal["nrg_diff"].argmin()]}')
    
    # Calculate the duration of missing energy
    len_def = []
    len_surp = []
    d = 0
    s = 0
    
    for x, y in zip(renew_anal['overprod'], renew_anal['overprod'][1:]):
        if x is True and y is True:
            s += 1
        elif x is False and y is False:
            d += 1
        elif x is True and y is False:
            len_surp.append(s)
            s = 0
        elif x is False and y is True:
            len_def.append(d)
            d = 0
    
    print()
    print(f'Mean surplus duration: {np.mean(len_surp):.2f} hours')
    print(f'Mean deficit duration: {np.mean(len_def)} hours')
    print(f'Max surplus duration: {max(len_surp)} hours')
    print(f'Max deficit duration: {max(len_def)} hours')
        
    


def main():
    global renew_anal
    full_data, five_data, el_nino, wind_data = load_data()
    oahu_load = load_data_2()
    
    daily_avg_five = five_data.groupby([five_data.index.month, 
                                        five_data.index.day]).mean()
    daily_avg_full = full_data.groupby([full_data.index.month, 
                                        full_data.index.day]).mean()
    daily_avg_wind = wind_data.groupby([wind_data.index.month, 
                                        wind_data.index.day]).mean()
    
    # Daily averages
    days = np.arange(1, 367, step=1)
    plot_daily_avg(daily_avg_full, days, fig=1,
                   title='Daily Average of Net Wave Energy, 2005-2019')
    plot_daily_avg(daily_avg_wind, days, fig=2, nrg='Wind',
                   title='Daily Average of Net Wind Energy, 2014-2019')
    # Seasonal averages
    # rel_oni, monthly_avg = compare_oni_nrg(full_data, el_nino)
    
    season_avg_wave = daily_seasonal_avg(full_data, energy_type='Wave',
                                         nrg='far_nrg')
    season_avg_wind = daily_seasonal_avg_no_interp(wind_data, energy_type='Wind',
                                                   nrg='wind_nrg')
        
    # let's create an index for the daily/monthly averages
    start = np.datetime64('2020-01-01')
    end = np.datetime64('2021-01-01')
    full_idx = np.arange(start, end, np.timedelta64(1, 'D'))
    
    daily_avg_five.index = full_idx
    daily_avg_full.index = full_idx
    
    # Query the wind energy to be the same time range as waves, and power use
    renew_data = wind_data.loc[(wind_data.index >= f'{five_data.index[0]}') &\
                              (wind_data.index < f'{five_data.index[-1]}')]
    
    # Let's create an interpolated time series for wave energy, 
    # far shore because of "environmentalism", and append to the wind df
    renew_data['wave_nrg'] = pchip_interpolate(five_data.index, 
                                               five_data.far_nrg,
                                               renew_data.index)
    # Scale for realistically obtainable energy, and in MW for better axes
    renew_data['wind_real'] = renew_data['wind_nrg'] * 0.4 * 0.001
    renew_data['wave_real'] = renew_data['wave_nrg'] * 0.1 * 0.001
    
    
    # Scale for the three-year analysis comparing it to the load data
    renew_data = renew_data.merge(oahu_load, how='left', 
                                  left_index=True, right_index=True)
    renew_anal = renew_data.dropna(axis=0, how='any')
    renew_anal = renew_anal.loc[renew_anal.index <\
                                pd.to_datetime('2019-01-01 00:00:00')]
    
    # How often do we meet demand?
    renew_anal['nrg_diff'] = renew_anal['nrg_use'] - (renew_anal['wave_real'] +
                                                      renew_anal['wind_real'])
    renew_anal['overprod'] = renew_anal['nrg_diff'] < 0
    
    print_info(renew_anal)
    
    plot_nrg_mix(renew_anal)
    plot_daily_season_avg(renew_data.dropna(axis=0, how='any'))
    
if __name__ == '__main__':
    main() 

