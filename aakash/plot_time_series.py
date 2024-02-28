#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots and general analysis of the time series wave data
"""


import pandas as pd
import numpy as np
from os import chdir
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import pchip_interpolate, UnivariateSpline
from copy import deepcopy


chdir('C:\\Users\\Aakas\\Documents\\Grad School\\NREL\\')


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
    
    full_year_data.index = full_year_data.index.tz_localize(tz='Pacific/Honolulu')
    five_year_data.index = five_year_data.index.tz_localize(tz='Pacific/Honolulu')
    
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
    oahu_loads = oahu_loads.tz_localize(tz='Pacific/Honolulu')
    
    return oahu_loads


def load_data_2():
    """
    Loads in some of the other data that was too long for the other func
    """
    oahu_loads = pd.read_csv('raw_data/hawaii_load_ref_long.csv')
    oahu_loads = clean_load_data(oahu_loads)
    
    ghi_data = pd.read_csv('mid_data/ghi_5_year.csv')
    ghi_data['time'] = pd.to_datetime(ghi_data['time'])
    ghi_data.set_index('time', inplace=True)
    
    return oahu_loads, ghi_data


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
    
    # Initiate our subplots
    
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    fig.suptitle(f'Seasonal Daily {energy_type} Energy Potential')
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, f'{energy_type} Energy Potential (MW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    # Create a spline to fit the low res data
    smooth = pchip_interpolate(time,
                               winter[nrg]._append(pd.Series(winter[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 0].plot(time[:-1], winter[nrg] / 1000)
    ax[0, 0].plot(new_time, smooth / 1000, color='red',
                  linestyle='dashed', alpha=0.8)
    ax[0, 0].set_ylim(4000, 4100)
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
                               spring[nrg]._append(pd.Series(spring[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    # We added an extra time for the spline, but remove this for actual plot
    ax[0, 1].plot(time[:-1], spring[nrg] / 1000)
    ax[0, 1].plot(new_time, smooth / 1000, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], new_time[-1]))
    ax[0, 1].set_ylim(3000, 3100)
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    smooth = pchip_interpolate(time,
                               summer[nrg]._append(pd.Series(summer[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 0].plot(time[:-1], summer[nrg] / 1000, label='Energy')
    ax[1, 0].plot(new_time, smooth / 1000, color='red',
                  linestyle='dashed', alpha=0.8, label='Interpolated')
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    ax[1, 0].legend(loc='lower left')
    ax[1, 0].set_ylim(1800, 1900)
    
    # PLOTTING Fall
    smooth = pchip_interpolate(time,
                               fall[nrg]._append(pd.Series(fall[nrg][0])),
                               new_time)
    
    # Testing for one point, the will subplot
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    # We added an extra time for the spline, but remove this for actual plot
    ax[1, 1].plot(time[:-1], fall[nrg] / 1000)
    ax[1, 1].plot(new_time, smooth / 1000, color='red',
                  linestyle='dashed', alpha=0.8)
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], new_time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 1].set_ylim(2450, 2550)
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
    fig.text(0.04, 0.5, f'{energy_type} Energy Potential (MW)',
             va='center', rotation='vertical')
    
    # PLOTTING WINTER
    ax[0, 0].grid()
    ax[0, 0].set_title('DJF')
    ax[0, 0].plot(time, winter[nrg] / 1000)
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
      
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    ax[0, 1].plot(time, spring[nrg] / 1000)
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    ax[1, 0].plot(time, summer[nrg] / 1000, label='Energy')
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    
    # PLOTTING Fall
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    ax[1, 1].plot(time, fall[nrg] / 1000)
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
             label='Energy Consumption', color='red', alpha=0.8)
    plt.plot(date_list, renew_data['wind_real'],
             label='Wind Energy', color='grey', alpha=0.6)
    plt.plot(date_list, renew_data['wave_real'],
             label='Wave Energy', color='blue', alpha=0.6)
    plt.plot(date_list, renew_data['solar_real'],
             label='Solar Energy', color='orange', alpha=0.6)
    plt.plot(date_list, renew_data['solar_real'] +\
                        renew_data['wind_real'] + renew_data['wave_real'],
                        label='Combined', color='violet')
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.grid()
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel('Month')
    plt.ylabel('Power (MW)')
    plt.title('Energy Mix vs. Consumption in Oahu')


def prob_no_power(renew_mix):
    """
    Plots the probability of not generating sufficient electricity
    at a given time for a given day
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
    # Create our time axis
    base = np.datetime64('2020-01-01 00:00:00')
    time = [base + np.timedelta64(x, 'h') for x in range(0, 24, 1)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    
    seasons = [winter, spring, summer, fall]
    seasons_str = ['winter', 'spring', 'summer', 'fall']
    prob_no_power = pd.DataFrame({x: np.zeros(24) for x in seasons_str})
    prob_season = {x: None for x in seasons_str}
    
    for season, string in zip(seasons, seasons_str):
        prob_season[string] = 1 - season['overprod'].sum() / len(season)
        for hour in np.arange(0, 24):
            szn_hour = season.iloc[season.index.hour == hour]
            prob_no_power[string][hour] = 1 - szn_hour['overprod'].sum() /\
                                              len(szn_hour)    
     
    plt.figure()
    plt.title('Probability of not Producting Sufficient Energy') 
    plt.plot(time, prob_no_power['winter'], label=f'Winter: {prob_season["winter"]:.2f}')
    plt.plot(time, prob_no_power['summer'], label=f'Summer: {prob_season["summer"]:.2f}')
    plt.plot(time, prob_no_power['spring'], label=f'Spring: {prob_season["spring"]:.2f}')
    plt.plot(time, prob_no_power['fall'], label=f'Fall: {prob_season["fall"]:.2f}')
    plt.legend()
    plt.grid()
    
    X = plt.gca().xaxis
    plt.xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    plt.xlabel('Time of Day')
    plt.ylabel('Prob. of insufficient power')
    
    return prob_no_power
    


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
    ax[0, 0].plot(time, winter['nrg_use'], color='red', alpha=0.8)
    ax[0, 0].plot(time, winter['wind_real'], color='grey', alpha=0.6)
    ax[0, 0].plot(time, winter['wave_real'], color='blue', alpha=0.6)
    ax[0, 0].plot(time, winter['solar_real'], color='orange', alpha=0.6)
    ax[0, 0].plot(time, winter['solar_real'] + winter['wave_real'] +\
                  winter['wind_real'], 
                  color='violet', alpha=1)
    X = ax[0, 0].xaxis
    ax[0, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING SPRING
    ax[0, 1].grid()
    ax[0, 1].set_title('MAM')
    ax[0, 1].plot(time, spring['nrg_use'], color='red', alpha=0.8)
    ax[0, 1].plot(time, spring['wind_real'], color='grey', alpha=0.6)
    ax[0, 1].plot(time, spring['wave_real'], color='blue', alpha=0.6)
    ax[0, 1].plot(time, spring['solar_real'], color='orange', alpha=0.6)
    ax[0, 1].plot(time, spring['solar_real'] + spring['wave_real'] +\
                  spring['wind_real'], 
                  color='violet', alpha=1)
    X = ax[0, 1].xaxis
    ax[0, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    
    # PLOTTING Summer
    ax[1, 0].grid()
    ax[1, 0].set_title('JJA')
    ax[1, 0].plot(time, summer['nrg_use'], 
                  label='Energy Use', color='red', alpha=0.8)
    ax[1, 0].plot(time, summer['wind_real'], label='Wind Energy', color='grey',
                  alpha=0.6)
    ax[1, 0].plot(time, summer['wave_real'], alpha=0.6,
                  label='Wave Energy', color='blue')
    ax[1, 0].plot(time, summer['solar_real'], alpha=0.6,
                  label='Solar Energy', color='orange')
    ax[1, 0].plot(time, summer['solar_real'] + summer['wave_real'] +\
                  summer['wind_real'], label='Combined',
                  color='violet', alpha=1)
    X = ax[1, 0].xaxis
    ax[1, 0].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    fig.legend(loc='upper right')
    
    # PLOTTING Fall
    ax[1, 1].grid()
    ax[1, 1].set_title('SON')
    ax[1, 1].plot(time, fall['nrg_use'], color='red', alpha=0.8)
    ax[1, 1].plot(time, fall['wind_real'], color='grey', alpha=0.6)
    ax[1, 1].plot(time, fall['wave_real'], color='blue', alpha=0.6)
    ax[1, 1].plot(time, fall['solar_real'], color='orange', alpha=0.6)
    ax[1, 1].plot(time, fall['solar_real'] + fall['wave_real'] +\
                  fall['wind_real'], 
                  color='violet', alpha=1)
    X = ax[1, 1].xaxis
    ax[1, 1].set_xlim((time[0], time[-1]))
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    ax[1, 0].set_xticks(ticks)
    

def plot_random_day(renew_anal, date=''):
    """
    Plots a random day's power cycle or allows for date selection
    
    date in format YEAR-MM-DD
    """
    if date:
        rel_data = renew_anal
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:])
        
        rel_data = renew_anal[renew_anal.index.year == year]
        rel_data = rel_data[rel_data.index.month == month]
        rel_data = rel_data[rel_data.index.day == day]
    else:
        # Must select for day with month in mind
        year = np.random.randint(2016, 2019)
        month = np.random.randint(1, 13)
        if month == 2 and year == 2016:
            day = np.random.randint(0, 30)
        elif month == 2:
            day = np.random.randint(0, 29)
        elif month in [1, 3, 5, 7, 8, 10, 12]:
            day = np.random.randint(0, 32)
        else:
            day = np.random.randint(0, 31)
        date = str(year) + '-' + str(month) + '-' + str(day)
        
        rel_data = renew_anal[renew_anal.index.year == year]
        rel_data = rel_data[rel_data.index.month == month]
        rel_data = rel_data[rel_data.index.day == day]
            
        
    plt.figure(np.random.randint(0, 10000000))
    
    # Create our time axis
    base = np.datetime64('2020-01-01 00:00:00')
    time = [base + np.timedelta64(x, 'h') for x in range(0, 24, 1)]
    ticks = [base + np.timedelta64(x, 'h') for x in range(0, 27, 3)]
    
    locator = mdates.HourLocator()
    fmt = mdates.DateFormatter('%H')
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xticks(ticks)
    
    plt.title(f'Energy mix for {date}')
    plt.xlabel('Hour')
    plt.ylabel('Energy (MW)')
    
    plt.plot(time, rel_data['nrg_use'], 
                  label='Energy Use', color='red', alpha=0.8)
    plt.plot(time, rel_data['wind_real'], label='Wind Energy', color='grey',
                  alpha=0.6)
    plt.plot(time, rel_data['wave_real'], alpha=0.6,
                  label='Wave Energy', color='blue')
    plt.plot(time, rel_data['solar_real'], alpha=0.6,
                  label='Solar Energy', color='orange')
    plt.plot(time, rel_data['solar_real'] + rel_data['wave_real'] +\
                  rel_data['wind_real'], label='Combined',
                  color='violet', alpha=1)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.grid()
    

def print_info(renew_anal):
    """
    Prints some useful information/statistics about the energy production data
    """
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
    global len_def, len_surp
    len_def = []
    len_surp = []
    d = 0
    s = 0
    
    for x, y, z in zip(renew_anal['overprod'], renew_anal['overprod'][1:], renew_anal.index):
        if x is True and y is True:
            s += 1
        elif x is False and y is False:
            d += 1
        elif x is True and y is False:
            len_surp.append(s)
            # if s > 250:
            #    print(f'SURPLUS AT {z}')
            s = 0
        elif x is False and y is True:
            len_def.append(d)
            # if d > 20:
            #  print(f'DEFICIT AT {z}')
            d = 0
    
    print()
    print(f'Mean surplus duration: {np.mean(len_surp):.2f} hours')
    print(f'Mean deficit duration: {np.mean(len_def):.2f} hours')
    print(f'Max surplus duration: {max(len_surp)} hours')
    print(f'Max deficit duration: {max(len_def)} hours')


def calc_sol_area(renew_anal, ghi_data, goal=1.5):
    """
    Calculates a representative area of solar panels needed to 
    make up the deficit in production, and creates
    the time series for the same
    """
    mean_deficit = renew_anal.query(f"overprod == {False}")["nrg_diff"].mean()
    # Let's aim for the mean deficit times some factor
    goal_energy = mean_deficit * goal
    # Mean GHI during production hours as a "representative" ghi
    # with a 20% efficiency, and convert to MW
    typ_ghi = float(ghi_data.query('ghi > 0.0').mean() * 0.2) * 10**-6
    # area of panels required
    area = goal_energy / typ_ghi
    print()
    print()
    print(f'Need {area / 4047:.2f} acres of panels')
    # Lets's specify area a la google project sunroof
    area = 4047 * 4000
    # actual energy time series, and back into MW, with 20% efficiency
    ghi_data['solar_real'] = area * ghi_data['ghi'] * 10**-6 * 0.2
    
    return ghi_data


def solar_coverage(renew_anal):
    """
    Calculates percentage of time overproduction occurs as a function
    of area of solar panels
    """
    areas = np.arange(0, 8000, 5) # acres
    perc_overprod = np.zeros(areas.shape)
    areas_m = areas * 4047 # convert to m2
    
    analyze = deepcopy(renew_anal)
    
    for n, area in enumerate(areas_m):
        analyze['solar_real'] = area * analyze['ghi'] * 10**-6 * 0.2
        analyze['nrg_diff'] = analyze['nrg_use'] -\
                                    (analyze['wave_real'] +
                                     analyze['wind_real'] + 
                                     analyze['solar_real'])
        analyze['overprod'] = analyze['nrg_diff'] < 0
        
        perc_overprod[n] = 100 * analyze["overprod"].sum() / len(analyze)
    
    # Fit a polynomial for now
    perc_prime = np.gradient(perc_overprod, 5)
    smooth = np.polyfit(areas, perc_prime, 6)
    smooth_func = np.poly1d(smooth)
    smooth = smooth_func(areas)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(areas, perc_overprod)
    ax1.set_ylabel('Energy Overprod. %')
    ax1.set_title('Solar Deployment impact on Energy')
    ax1.grid()
    
    ax2.plot(areas, perc_prime, alpha=0.8)
    ax2.plot(areas, smooth, color='red', linestyle='dashed')
    ax2.grid()
    ax2.set_xlabel('Area of Panels (acres)')
    ax2.set_ylabel('dE/dA')
    

def plot_load_curve(renew_data):
    """
    Plots the load curve, averaged over the data available
    """
    data = renew_data.groupby([renew_data.index.month,
                                     renew_data.index.day]).mean()
    plt.figure()
    plt.plot(np.sort(data['nrg_use'])[::-1], color='red')
    plt.xlabel('Days')
    plt.ylabel('Energy Consumption (MW)')
    plt.title('Load Curve for Oahu')
    plt.grid()


def main():
    global renew_anal
    full_data, five_data, el_nino, wind_data = load_data()
    oahu_load, ghi_data = load_data_2()
    
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
    
    daily_seasonal_avg(full_data, energy_type='Wave',
                                         nrg='far_nrg')
    daily_seasonal_avg_no_interp(wind_data, energy_type='Wind',
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
    
    renew_data = renew_data.tz_convert(tz='Pacific/Honolulu')
    # Let's create an interpolated time series for wave energy, 
    # far shore because of "environmentalism", and append to the wind df
    renew_data['wave_nrg'] = pchip_interpolate(np.asarray(five_data.index, 
                                                          dtype=float), 
                                               five_data.far_nrg,
                                               np.asarray(renew_data.index, 
                                                          dtype=float))
    # Scale for realistically obtainable energy, and in MW for better axes
    renew_data['wind_real'] = renew_data['wind_nrg'] * 0.4 * 0.001
    renew_data['wave_real'] = renew_data['wave_nrg'] * 0.1 * 0.001
    
    # Scale for the two-year analysis comparing it to the load data
    renew_data = renew_data.merge(oahu_load, how='left', 
                                  left_index=True, right_index=True)
    renew_anal = renew_data.dropna(axis=0, how='any')
    renew_anal = renew_anal.loc[renew_anal.index <\
                                pd.to_datetime('2019-01-01 00:00:00').\
                                    tz_localize(tz='Pacific/Honolulu')]
    
    # How often do we meet demand?
    renew_anal['nrg_diff'] = renew_anal['nrg_use'] - (renew_anal['wave_real'] +
                                                      renew_anal['wind_real'])
    renew_anal['overprod'] = renew_anal['nrg_diff'] < 0
    
    # Calculate solar energy requirements
    ghi_data = calc_sol_area(renew_anal, ghi_data, goal=1.5)
    
    # Let's do the seasonal solar flux from before for solar too 
    
    daily_avg_solar = (ghi_data['solar_real'] * 10**3).groupby([ghi_data.index.month, 
                                                 ghi_data.index.day]).mean()
    plot_daily_avg(daily_avg_solar, days, fig=3, nrg='Solar',
                   title='Daily Average of Needed Solar Energy, 2014-2019')
    
    daily_seasonal_avg_no_interp(ghi_data * 10**3, energy_type='Solar',
                                                    nrg='solar_real')
    
    # Merge calculated solar time series
    renew_anal = renew_anal.merge(ghi_data, how='left',
                                  left_index=True, right_index=True)
    renew_anal.set_index(pd.to_datetime(renew_anal.index), inplace=True)
    renew_anal = renew_anal.tz_convert(tz='Pacific/Honolulu')
    
    # Reduce energy diff by adding impact of solar
    renew_anal['nrg_diff'] = renew_anal['nrg_use'] - (renew_anal['wave_real'] +
                                                      renew_anal['wind_real'] + 
                                                      renew_anal['solar_real'])
    renew_anal['overprod'] = renew_anal['nrg_diff'] < 0
    
    # Get some plots and statistics
    print_info(renew_anal)
    
    plot_nrg_mix(renew_anal)
    plot_daily_season_avg(renew_anal)
    prob_no_power(renew_anal)
    
    solar_coverage(renew_anal)
    
    # Plot the load curve, just to have
    plot_load_curve(renew_anal)
    # Let's write the df to a csv and analyze characteristics
    # in a different file- this has gotten long and I can
    # always import plotting funcs when necessary
    renew_anal.to_csv('final_data/nrg_mix_2016-18.csv')


if __name__ == '__main__':
    main() 
