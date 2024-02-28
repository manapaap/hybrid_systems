#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOAL: Analysis of the combined renewables data to determine
energy storage requirements
"""


import pandas as pd
from os import chdir
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


chdir('C:\\Users\\Aakas\\Documents\\Grad School\\NREL\\')
pd.set_option('mode.chained_assignment', None)


def load_data():
    """
    Loads the time series data and formates it as needed
    """
    renew_mix = pd.read_csv('final_data/nrg_mix_2016-18.csv')
    renew_mix['time'] = pd.to_datetime(renew_mix['time'])
    renew_mix.set_index('time', inplace=True)
    
    return renew_mix


def simulate_battery(renew_mix, max_storage=565, num_batt=3, start=0):
    """
    Iterates along the renew mix array, trying to store
    and use energy as needed, to see if a net positive is possible
    
    May try to incorporate a "spin up" period where the battery
    starts at a set power level first, or intentionally start
    during the summer
    
    Battery units in MJ
    
    Max storage units in MWH, reworked into MJ
    """
    renew_mix['battery'] = 0
    # starting energy
    time_delta = 3600 # seconds
    max_storage = num_batt * max_storage * time_delta
    renew_mix.battery[0] = start * time_delta
    size = len(renew_mix)
    
    for n in range(1, size):
        # must multiply by negative since energy diff represents
        # underproduction
        energy_delta = -1 * renew_mix['nrg_diff'][n] * time_delta
        new_battery = renew_mix['battery'][n - 1] + energy_delta
        
        if new_battery < 0:
            # Fully drained battery 
            renew_mix.battery[n] = 0
        elif new_battery > max_storage:
            # Can't fill battery past maximum
            renew_mix.battery[n] = max_storage
        else:
            renew_mix.battery[n] = new_battery
    
    return renew_mix


def needed_cap_game(renew_mix):
    """
    Uses bisection to find required storage to ensure power
    doesn't go to zero
    """
    fail_meet = True
    min_bat = 565
    max_bat = 10_000_000_000
    old_bat = max_bat
    n = 0
    while fail_meet:
        curr_bat = (min_bat + max_bat) / 2
        renew_mix = simulate_battery(renew_mix, max_storage=curr_bat,
                                     num_batt=1, start=curr_bat)
        
        if renew_mix.battery.isin([0]).any():
            n += 1
            min_bat = curr_bat
            old_bat = curr_bat
        elif n > 100 or np.isclose(old_bat, curr_bat, atol=0.1):
            fail_meet = False
        else:
            n += 1
            max_bat = curr_bat
            old_bat = curr_bat

    return curr_bat


def moving_average(array, time, n=96, name='nrg_diff'):
    """
    Takes the moving average for the last n points and returns a df 
    with the new time and data arrays
    
    Pls n only divisible by the length of the array
    """
    start = time[0]
    end = time[-1]
    
    num = int(len(array) / n)
    out_arr = np.zeros(num)
    
    for k, m in enumerate(range(0, num * n, n)):
        out_arr[k] = np.mean(array[m:m + n])
    
    new_time = np.arange(start, end, np.timedelta64(n, 'h'))

    return pd.DataFrame({'time': new_time, name: out_arr}).set_index('time')


def plot_battery(renew_mix, batt, base=''):
    """
    Plots the time series of the battery capacity and of the power flux into
    the same, taking a rolling average to smooth the data
    """
    smooth_mix = moving_average(renew_mix['battery'], renew_mix.index, n=96,
                                name='battery')
    # smooth_nrg = moving_average(-renew_mix['nrg_diff'], renew_mix.index, n=0)
    smooth_nrg = -renew_mix['nrg_diff'][::48]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.plot(smooth_mix.index, 100 * smooth_mix['battery'] / 3600 / batt)
    ax1.vlines([smooth_mix.index[86], smooth_mix.index[194]], 0, 100,
               linestyle='dashed', color='red', alpha=0.4)
    if base:
        ax1.set_title(f'Simulated Energy for {batt / 1000:.1f} GWh battery' +
                      f' with {base:.1f} MW variable load')
    else:
        ax1.set_title(f'Simulated Energy for {batt / 1000:.2f} GWh battery')
    ax1.set_ylabel('Battery Capacity (%)')
    ax1.grid()
    ax2.grid()
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Energy Diff. (MW)')
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax2.xaxis.get_major_locator()))
    ax2.plot(smooth_nrg)


def simulate_load(renew_mix, batt, load):
    """
    Simulates the impact of base load on a given battery capacity
    
    load in MW batt in MWH
    """
    renew_mix.battery[0] = batt
    # starting energy
    time_delta = 3600 # seconds
    load *= time_delta # Get this into MJ from MW
    batt *= time_delta # Get this into MJ from MWH
    size = len(renew_mix)
    
    for n in range(1, size):
        # must multiply by negative since energy diff represents
        # underproduction
        energy_delta = -1 * renew_mix['nrg_diff'][n] * time_delta
        new_battery = renew_mix['battery'][n - 1] + energy_delta + load
        
        if new_battery < 0:
            # Sufficient base load to prevent draining
            renew_mix.battery[n] = 0
        elif new_battery > batt:
            # Can't fill battery past maximum
            renew_mix.battery[n] = batt
        else:
            renew_mix.battery[n] = new_battery
    
    return renew_mix
    

def needed_base_game(renew_mix, batt):
    """
    Uses bisection to find the required base load such that
    the battery doesn't go to zero
    """
    # Bounds on power, in MW
    min_load = 0
    max_load = 1000
    # Initial conditions
    old_load = min_load
    fail_needs = True
    n = 0
    
    while fail_needs:
        curr_load = (min_load + max_load) / 2
        
        renew_mix = simulate_load(renew_mix, batt, curr_load)
    
        if renew_mix.battery.isin([0]).any():
            n += 1
            min_load = curr_load
            old_load = curr_load
        elif n > 100 or np.isclose(old_load, curr_load, atol=0.1):
            fail_needs = False
        else:
            n += 1
            max_load = curr_load
            old_load = curr_load

    return curr_load


def main():
    global renew_mix
    renew_mix = load_data()
    
    # Assuming the current battery farm
    renew_mix = simulate_battery(renew_mix, max_storage=565,
                                 num_batt=5, start=565)
    plot_battery(renew_mix, 565)
    
    # Let's do a series of trials to determine the minimum necessary storage
    # requirements to achieve full renewables
    batt = needed_cap_game(renew_mix)
    
    renew_mix = simulate_battery(renew_mix, max_storage=batt,
                                 num_batt=1, start=batt)
    plot_battery(renew_mix, batt)
    
    # Let's now assess that energy storage requirement given some
    # variable base load
    # load = needed_base_game(renew_mix, 565 * 10)
    # 
    # renew_mix = simulate_load(renew_mix, batt=565 * 10,
    #                              load=load)
    # plot_battery(renew_mix, 565 * 10, load)
        
    # Yikes, that's less than ideal. What if we specify the load and
    # Assess the actual power outages
    load = 200 # MW
    batt = 565 * 5 # MWh
    
    renew_mix = simulate_load(renew_mix, batt, load)
    plot_battery(renew_mix, batt, load)
    print('\n\n\nPower needs met ' +
          f'{100 * len(renew_mix.query("battery > 0.0")) / len(renew_mix):.2f}' +
          '% of the time')


if __name__ == '__main__':
    main()
