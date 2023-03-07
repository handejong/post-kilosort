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

    # NOTE: if you don't specify units or channels, they will be infered from the oldes open plot:
    amplitude_plot = data.plot.amplitude()

    # PCA plot plots' the first principal component. It's one of my favorites.
    pca_plot = data.plot.pca()

    # Plot a welcome message
    welcome_message()
