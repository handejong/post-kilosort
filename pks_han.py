#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is just an example of how I work with PKS. I particularly like my
semi-automated clustering routines:

    - next()
    - auto_next()
    - han_super_auto()

In order to put the figures at specific locations on my desktop I use
a window manger called 'I3'.


Last Updated: Jun 27 16:40:34 2023

@autor: Han de Jong
"""

# Imports
import sys
import os
import seaborn as sns

# We need to find the filepath to PKS
this_file = __file__
pks_path = this_file[:-this_file[-1:0:-1].find('/')]
sys.path.append(pks_path + 'src/')

# Other imports:
from pks_processing import pks_dataset
import matplotlib.pyplot as plt

# Welcome message
def welcome_message():

    print(' ')
    print(f"{' Welcome to Post-Kilosort ':*^100}")
    print(' ')

# This function will focus the terminal
def go_back_to_terminal():
    string = r' i3-msg "[class=\"Gnome-terminal\"] focus " > /dev/null 2>&1'
    os.system(string);   


if __name__ == '__main__':

    # Setup interactive plotting
    plt.ion()

    # Grab the path from input argument
    path = sys.argv[1]

    # Make the dataset
    data = pks_dataset(path)

    # Optional (but recomended) sort the clusters in order of chanel (instead of the order
    # in which Kilosort found them.)
    data.clusters.sort_values('mainChannel', inplace=True)

    # Delete clusters that have less then X spikes
    X = 1
    print(f'Deleting units with less than {X} spikes.')
    for i, temp in data.clusters.iterrows():
        if temp.spikeCount < X:
            data.sort.delete_unit(i)

    # Let's have a look at the first 5 units of the "todo" dataframe
    temp = data.sort.todo().iloc[:5, :]
    units = temp.index.values
    s_chan = max([temp.mainChannel.min()-2, 0])
    channels = list(range(s_chan, s_chan+6))

    # Make some plots as an example
    # NOTE these plots are also all stored in data.linked_plots!
    waveform_plot = data.plot.waveform(units, channels)
    os.system("i3-msg floating toggle > /dev/null 2>&1")

    # NOTE: if you don't specify units or channels, they will be infered from the oldes open plot:
    amplitude_plot = data.plot.amplitude()
    os.system("i3-msg floating toggle > /dev/null 2>&1")

    # PCA plot plots' the first principal component. It's one of my favorites.
    pca_plot = data.plot.pca()
    os.system("i3-msg floating toggle > /dev/null 2>&1")

    # Plot a welcome message
    welcome_message()

    # Plot peri-event plots
    try:
        stamps = data.get_nidq(); stamps = stamps[stamps.Channel=='AI_2']
        peri_start = data.plot.peri_event(stamps = stamps.Start.values/1000)
        os.system("i3-msg floating toggle > /dev/null 2>&1")
        peri_stop =  data.plot.peri_event(stamps = stamps.Stop.values/1000)
        os.system("i3-msg floating toggle > /dev/null 2>&1")

        # Add timestamps tot he plots
        data.plot.add_timestamps(stamps.Start.values/1000)
    except:
        print('Unable to plot peri-event plots for NIDQ channel AI_2')
    
    # Make some shortlinks in the base workspace
    focus = data.plot.focus_unit
    todo = data.sort.todo
    delete = data.sort.delete_unit
    done = data.sort.mark_done
    add = data.plot.add_unit
    remove = data.plot.remove_unit

    # Put it all somewhere cool using I3

    # Waveforms
    string = r' i3-msg "[title=\"Figure 1\"] focus" > /dev/null 2>&1'
    os.system(string)
    for i in range(4):
        os.system("i3-msg move left > /dev/null 2>&1")

    # Peri_event start
    string = r' i3-msg "[title=\"Figure 4\"] focus" > /dev/null 2>&1'
    os.system(string)
    for i in range(5):
        os.system("i3-msg move left > /dev/null 2>&1")

    # Peri_event stop
    string = r' i3-msg "[title=\"Figure 5\"] focus" > /dev/null 2>&1'
    os.system(string)
    for i in range(5):
        os.system("i3-msg move left > /dev/null 2>&1")

    # Peri_event start again
    string = r' i3-msg "[title=\"Figure 4\"] focus" > /dev/null 2>&1'
    os.system(string)
    os.system("i3-msg move up > /dev/null 2>&1")
    
    # Waveforms again
    string = r' i3-msg "[title=\"Figure 1\"] focus" > /dev/null 2>&1'
    os.system(string)
    os.system("i3-msg move left > /dev/null 2>&1")
    os.system("i3-msg move left > /dev/null 2>&1")

    # Put the PCA plot up
    string = r' i3-msg "[title=\"Figure 3\"] focus" > /dev/null 2>&1'
    os.system(string)
    os.system("i3-msg move up > /dev/null 2>&1")

    # Put the amplitude plot to the right
    string = r' i3-msg "[title=\"Figure 2\"] focus" > /dev/null 2>&1'
    os.system(string)
    os.system("i3-msg move right > /dev/null 2>&1")

    # Resize the waveform plot
    string = r' i3-msg "[title=\"Figure 1\"] focus" > /dev/null 2>&1'
    os.system(string)
    os.system('i3-msg resize grow width 250px > /dev/null 2>&1')

    # Resize the PCA plot
    string = r' i3-msg "[title=\"Figure 3\"] focus" > /dev/null 2>&1'
    os.system(string)    
    os.system('i3-msg resize grow height 250px > /dev/null 2>&1')
    os.system('i3-msg resize grow width 250px > /dev/null 2>&1')

    # Finish up
    go_back_to_terminal()


################################# Han cluster helper #################################

# This function will force Matplotlib to update all included figures
def update_all_plots():
    for fig in [waveform_plot, pca_plot, amplitude_plot, peri_stop, peri_start]:
        plt.figure(fig.fig)
        plt.draw()

# This function will focus on the next unit I should work on and then compare it
# to all neighboring units.
def next():
    unit_id = int(data.sort.todo().index[0])
    data.plot.focus_unit(unit_id)
    neighbors = data.sort.neighbors(unit_id, 3)
    maybe = []
    for i in neighbors.index:
        if i == unit_id:
            continue
        add(i)
        update_all_plots()
        go_back_to_terminal()
        print(' ')
        print(neighbors.loc[[unit_id, i], :])
        answer = input("Is this the same unit? (y/n/m/d/c): ")

        # Does the user want to see the correlogram?
        if answer == 'c':
            fig = data.plot.correlogram()
            go_back_to_terminal()
            answer = input("Is this the same unit? (y/n/m/d): ")
            plt.close(fig)

        # Do we delete this unit
        if answer == 'd':
            delete(i)
            continue

        # Merge them?
        if answer == 'y':
            data.sort.merge_units(i, unit_id)
            continue

        # Not the same unit
        if answer == 'n':
            remove(i)
            continue

        # Maybe?
        if answer == 'm':
            remove(i)
            maybe.append(i)
            continue

        # If you get here, abort
        print(f'Unknown input: {answer}')
        return

    # Print an overview of the "maybe" units
    if len(maybe)>0:
        print(' ')
        print('Have a look at these units: ')
        print(neighbors.loc[maybe, :])

# This function basically allows me to cluster an entire dataset. The only thing
# that I have to do sometimes is drop out (keep pressing 'q') in order to translate
# or split units or do some more detailed inspection (e.g. look at raw waveform
# samples).
def auto_next():

    exit = False

    while not exit:

        found_good_unit = False
        while not found_good_unit:
            unit_id = int(data.sort.todo().index[0])
            focus(unit_id)
            fig = data.plot.ISI(unit_id)
            update_all_plots()
            go_back_to_terminal()
            answer = input('Should we delete this unig? yes/n: ')
            plt.close(fig)
            if answer == 'yes':
                delete(unit_id)
                continue
            found_good_unit = True

        # Compare this unit to all it's neighbors (yes all of them)
        next()
        update_all_plots()
        go_back_to_terminal()
        print(' ')
        print(f"{' Unit done? ':*^100}")
        answer = input('y/n/q/i: ')

        if answer == 'i':
            fig = data.plot.ISI(unit_id)
            print(f"{' Unit done? ':*^100}")
            go_back_to_terminal()
            answer = input('y/n/q: ')
            plt.close(fig)

        if answer == 'y':
            done(unit_id)
        if answer == 'q':
            exit = True

# This function will plot both the raw_unit sample and the ISI of a unit.
def details(unit):

    # Plot raw waveforms
    data.plot.raw_unit_sample(unit, 100)

    # Plot the ISI
    data.plot.ISI(unit)

# This function allows me to quickly get rid of a large number of "MUA"
# units.
def han_super_auto(data):

    # First plot overview of the entire dataset
    plt.figure()
    sns.scatterplot(x='Amplitude', y='mainChannel', size='spikeCount', hue='KSLabel', data=data.clusters)
    done = data.clusters.loc[data.clusters.done, :]
    sns.scatterplot(x='Amplitude', y='mainChannel', size='spikeCount', hue='KSLabel', 
        edgecolor='red', data=done, linewidth=2, legend=False)
    go_back_to_terminal()

    # Figure out a threshold below which we are killing all MUA units
    threshold = float(input("Delete al MUA units with amplitude < "))

    # Delete them all
    indexer = (data.clusters.KSLabel=='mua') & (data.clusters.Amplitude<threshold)
    if input(f'Deleting {sum(indexer)} units, ok? y/n') == 'y':
        indexer = data.clusters.index[indexer].values
        data.sort.delete_unit(indexer)

    # Figure out a threshold below which we want to inspect MUA units
    threshold = float(input("Focus on all 'mua' units with amplitude < "))
    indexer = (data.clusters.KSLabel=='mua') & (data.clusters.Amplitude<threshold)
    indexer = data.clusters.index[indexer].values
    for unit in indexer:
        if not unit in data.done:
            temp = data.plot.ISI(unit)
            data.plot.focus_unit(unit); plt.pause(0.001)

            # Update all other figures
            update_all_plots()
            go_back_to_terminal()
            
            check = input(f'Delete unit ({unit}/{data.clusters.index[-1]}, {int(data.clusters.loc[unit].spikeCount)} spikes)? (y/n/q) ')
            if check == 'y':
                data.sort.delete_unit(int(unit))
            elif check == 'q':
                return -1
            plt.close(temp)

    return  




