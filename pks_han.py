#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Last Updated: Nov 11 14:01:20 2022

@autor: Han de Jong
"""


"""
Find the path to this file and work with it.

I use this so I can keep all PKS files in a Github folder and use
a simple system alias to call is when I'm inside a folder with data.

For instance:
alias pks="ipython -i /home/han/Git/post-kilosort/pks.py"
"""
import sys
import os
import seaborn as sns

this_file = __file__
pks_path = this_file[:-this_file[-1:0:-1].find('/')]
sys.path.append(pks_path + 'src/')

# Other imports:
from pks_processing import pks_dataset
import matplotlib.pyplot as plt

def welcome_message():

    print(' ')
    print(f"{' Welcome to Post-Kilosort ':*^100}")
    print(' ')


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

    # Put it all somewhere cool
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


################################# Han cluster helper ##############################################
def update_all_plots():
    for fig in [waveform_plot, pca_plot, amplitude_plot, peri_stop, peri_start]:
        plt.figure(fig.fig)
        plt.draw()

def next():
    i = data.sort.todo().index[0]
    data.plot.focus_unit(i, show_neighbors=0.8)

def details(unit):

    # Plot raw waveforms
    data.plot.raw_unit_sample(unit, 100)

    # Plot the ISI
    data.plot.ISI(unit)

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
    return None    




