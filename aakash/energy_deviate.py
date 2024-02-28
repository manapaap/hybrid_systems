# -*- coding: utf-8 -*-
"""
GOAL: Assessment of deviations of energy production from a defined
seasonal mean state, thus letting us see why we have dips when we do
"""

import pandas as pd
from os import chdir
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import sys as sys

chdir('C:\\Users\\Aakas\\Documents\\Grad School\\NREL\\')
sys.path.append("aakash\\")
from battery import load_data


def get_typ_prod(full_data):
    """
    Returns a df containing a "typical" production value for each energy type
    defined by season
    """
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
    winter_day = winter.groupby(winter.index.hour).mean()
    spring_day = spring.groupby(spring.index.hour).mean()
    summer_day = summer.groupby(summer.index.hour).mean()
    fall_day = fall.groupby(fall.index.hour).mean()
    
    # We now have 4 dataframes giving us a typical day for each
    # energy type. We need to subtract this from every day from the full
    # day to get the deviations from mean production
    for energy_type in ['wind_real', 'wave_real', 'solar_real']:
        winter[energy_type] = concat_prod(winter[energy_type],
                                          winter_day[energy_type])
        spring[energy_type] = concat_prod(spring[energy_type],
                                          spring_day[energy_type])
        summer[energy_type] = concat_prod(summer[energy_type],
                                          summer_day[energy_type])
        fall[energy_type] = concat_prod(fall[energy_type],
                                        fall_day[energy_type])
    # Re-combine the datasets
    deviations = pd.concat([winter, spring, summer, fall])
    deviations = deviations.loc[:, ['wind_real', 'wave_real', 'solar_real']]
    
    return deviations.sort_values(by='time')
    

def concat_prod(full_prod, day_prod):
    """
    Stacks the values of day_prod to the same length of the full_prod, and then
    returns the difference. This difference represents the anomalies in 
    production for the energy type, defined by season
    """
    stack = len(full_prod) // len(day_prod)
    stack_days = pd.concat(stack * [day_prod])    
    # Concert to numpy to avoid annoying time errors
    stack_days = np.asarray(stack_days)
    
    return full_prod - stack_days
    

def plot_deviations(devs, roll=24):
    """
    Plots the deviations of the energy production over the time period
    
    Averages the data over the extent of roll
    """
    roll = int(roll)
    devs = devs.rolling(roll).mean()[::roll]
    
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    
    fig.suptitle('Deviations from Typical Output')
    
    axs[0].plot(devs.index, devs['wave_real'], color='blue')
    axs[0].set_ylabel('Waves')
    axs[0].grid()
    
    axs[1].plot(devs.index, devs['wind_real'], color='grey')
    axs[1].set_ylabel('Wind')
    axs[1].grid()
    
    axs[2].plot(devs.index, devs['solar_real'], color='orange')
    axs[2].set_ylabel('Solar')
    axs[2].set_xlabel('Time')
    axs[2].grid()
    axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axs[2].xaxis.get_major_locator()))
    
    
    plt.show()


def main():
    renew_mix = load_data()
    
    devs = get_typ_prod(renew_mix)
    
    plot_deviations(devs, roll=24 * 8)
    

if __name__ == '__main__':
    main()
