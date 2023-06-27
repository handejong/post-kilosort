#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
When run by itself, this file will:

    1. Load the selected dataset
    2. Plot some example plots

You can fully customize your requested plot or write your own clustering routine below.
For an example of how I work with PKS, see "pks_han.py" in the same repository.

Example:
--------
Navigate to the folder with your data
(this data needs to be previously clustered using Kilosort 3.0)
    
    $cd ~/data/folder

Run pks on that folder
(Suppose you downloaded pks to a folder named 'Git')

    $python -i ~/Git/PKS/pks.py .

The '.' indicates you want to run PKS on the current folder

TIP:
----
Add pks as an alias to you .bashrc like so:

>>> alias pks="ipython -i /home/han/Git/post-kilosort/pks.py"

Now you can run it from any folder by just typing "pks .""

Last Updated: Jun 27 16:24:20 2023
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

# Print a little welcome message
def welcome_message():

    print(' ')
    print(f"{' Welcome to Post-Kilosort ':*^100}")
    print(' ')

############################# BELOW IS WHERE YOU CUSTOMIZE PKS #############################
# Feel free to remove or add things
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
    waveform_plot = data.plot.waveform(units, channels)

    # NOTE: if you don't specify units or channels, they will be infered from the oldes open plot:
    amplitude_plot = data.plot.amplitude()

    # PCA plot plots' the first principal component. It's one of my favorites.
    pca_plot = data.plot.pca()

    # Print a welcome message
    welcome_message()

    # Maybe you have timestamps in a logAI file?
    if False:
        # Plot peri-event plots from AI_2 (or any other analog input)
        try:
            stamps = data.get_nidq(); stamps = stamps[stamps.Channel=='AI_2']
            peri_start = data.plot.peri_event(stamps = stamps.Start.values/1000)
        except:
            print('Unable to plot peri-event plots for NIDQ channel AI_2')
    
    # Make some shortlinks in the base workspace
    focus = data.plot.focus_unit
    todo = data.sort.todo
    delete = data.sort.delete_unit
    done = data.sort.mark_done
    add = data.plot.add_unit
    remove = data.plot.remove_unit

    



