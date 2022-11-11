#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Last Updated: Nov 11 14:01:20 2022

@autor: Han de Jong
"""

# Imports
import sys; sys.path.append('./src/')
from pks_processing import pks_dataset
import matplotlib.pyplot as plt

# Startup
plt.ion()




def welcome_message():

   	print(' ')
   	print(f"{' Welcome to Post-Kilosort ':*^100}")
   	print(' ')


if __name__=='__main__':

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
		if temp.spikeCount<X:
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