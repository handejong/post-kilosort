#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The functions below are all used for plotting and visualization of Kilosort
output data.

This functions provide a general overview of the dataset:
    - overview (not implemented yet)
    - probe_view (not implemented yet)
    
These function are used to inspect the unit waveforms. 
    - waveform: used to plot waveforms
    - pca: used to plot the first principal component of the waveform
    - amplitude: used to plot the amplitude in time
    - inspect: all three of the above plus the correlograms.
    - plot_3D: will plot 

Run them by calling:
    
    >>> function(self, list_of_units, list_of_channels)

This is a nice way to look at individual (raw) waveforms
    - raw_spike sample

For instance, this will plot 25 (uniformly selected) waveforms
of unit 4 on channel 3, 4 and 5.

    >>> dataset.plot.raw_spike_sample(4, [3, 4, 5], sample_n = 25)

Sometimes you want to see what a unit looks like on ALL channels
    - raw_unit_sample

This will plot either a single waveform of unit 4 or the average of
1000 waveforms on all channels

    >>> raw_unit_sample(4, sample_n = 1)
    >>> raw_unit_sample(4, sample_n = 1000)
    
for instance, this will plot the waveforms of unit 3 and 4 on electrode 0-5:
        
    >>> dataset.plot.waveform([3, 4], [0, 1, 2, 3, 4, 5]) 
    
There are functions to look at firing distribution: 
    - correlogram
    - ISI
    
Run them by calling:
    
    >> function(self, list_of_units)

This function deals with peri-even histograms
    - peri_event

For instance, this will make the PEH around where T=0 are the
timepoints in 'stamps' of unit 3 and unit 4.0

    >>> dataset.plot.peri_event([3, 4], stamps)

Finally there are functions that give an overview of the complete dataset
    - spike_raster
    - cluster_progress TODO

Some (most) plots allow you to add or remove units or to change channels
or add timestamps. You can do this to individual plots or you can do this
to all plots that are currently open simulateneously as follows:

    >>> dataset.plot.add_unit(5) # Will add unit 5 to all plots
    >>> dataset.plot.remove_unit(5) # Will remove unit 5 from all plots
    >>> dataset.plot.add_timestamps(stamps) # Will add timestamps
    >>> dataset.plot.change_channel(1, 3) # Will change the subplot at index 3 to channel 3

 This line will focus all plots on unit 5. It will pick the best channels and only show
 neighboring units if they have at least 0.9 similarity with unit 5.

    >>> dataset.plot.focus(5, show_neighbors=0.9)

Created: Fri Nov  4 12:21:50 2022
Last Updated: Jun 23 10:42:11 2023

@author: Han de Jong
"""

# Dependend libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams
import seaborn as sns
import pks_attribute_functions as atf
from mpl_toolkits import mplot3d
from functools import partial

# Settings
rcParams['toolbar'] = 'None'

class pks_plotting:

    def __init__(self, pks_dataset):

        # Store the datset itself
        self.data = pks_dataset

        # Store formatting parameters that may or may not be used
        self.plot_max = 250
        self.palette = "tab10"  # hls is unclear, but looks good.
        self.colormap = self._color_map(10)
        self.facecolor = 'k'
        self.axcolor='w'

    def waveform(self, units=None, channels=None):
        """

        Responsible for making a waveform plot.

        Parameters
        ----------
        units : list of units or a single int
            List (or NumPy Array) of the units you want to plot.
        channels : list of channels or a single int
            List (or NumPy Array) of the channels you are interested in.
        NOTE: if units or channels == None, it will be infered from currently
        open plots.

        Returns
        -------
        axs : waveform_plot object
            A waveform plot object that controls the behavior of a window that
            contains the plots of the waveforms.
        """

        return waveform_plot(self, units, channels)

    def pca(self, units=None, channels=None):
        """
        Responsible for plotting the first principal component attribute plot.
        Channel PCA's are fitted by the "channel_pca" method of the pks_dataset
        object. These are performed on a dataset containing a random sample of
        waveforms on the channel obtained when a spike occured on this channel 
        or it's neighboring channels according to Kilosort.

        Parameters
        ----------
        units : list of units or a single int
            List (or NumPy Array) of the units you want to plot.
        channels : list of channels or a single int
            List (or NumPy Array) of the channels you are interested in.
        NOTE: if units or channels == None, it will be infered from currently
        open plots.

        Returns
        -------
        axs : a pca_plot object
            A pca_plot object is just a attribute plot object with a slightly more
            complicated attribute function.
        """

        # Infer units and channels
        units, channels = self.data._infer_unit_channels(units, channels)

        # Do the channel PCAs
        pca = []
        for channel in channels:
            pca.append(self.data.channel_pca(channel))

        return pca_plot(self, units, channels, other_data=pca)

    def amplitude(self, units=None, channels=None):
        """

        Plots the amplitude of the units in 'units' on the channels in
        'channels' in time.

        Parameters
        ----------
        units : list of units or a single int
            List (or NumPy Array) of the units you want to plot.
        channels : list of channels or a single int
            List (or NumPy Array) of the channels you are interested in.
        NOTE: if units or channels == None, it will be infered from currently
        open plots.

        Returns
        -------
        axs : An atribute plot object
            This object controls the behavior of the attribute plot

        """
        return time_plot(self, units, channels, atf._calc_amplitude)

    def peri_event(self, units=None, stamps=None, peri_event_window=(-5.0, 5.0)):
        """

        Plot the spikes as events relative 

        Parameters
        ----------
        units : list of units or a single int
            List (or NumPy Array) of the units you want to plot.
        stamps: an array with timestamps in seconds
        NOTE: if units or channels == None, it will be infered from currently
        open plots.

        Returns
        -------
        axs : A peri-event plot
            This object controls the behavior of peri-event plot

        """
        return peri_event_plot(self, units=units, channels=[1, 2, 3], other_data=stamps,
            peri_event_window=peri_event_window)

    def custom_plot(self, units=None, channels=None, type='time_plot', 
        attribute_function=None):
        """
        You can use this function to plot any particular waveform attribute in two ways

            - time_plot: Where X is the time (in s) and Y the amplitude
            - attribute_plot: Where X and Y are attributes on different channels

        Have a look at the pks_attribute_functions to see how they should be formatted

        Example:

            >>> self.plot.custom_plot(units = [0, 1, 13], 
                                      channels = [1, 2, 13], 
                                      type='attribute_plot', 
                                      attribute_function=atf._calc_amplitude)

        This will plot a 3x3 grid of scatter plots with the amplitude of
        the waveforms of unit 0, 1 & 13 on channel 1, 2 13.

        """
        if type == 'time_plot':
            try:
                return time_plot(self, units, channels, attribute_function)
            except:
                raise Exception(
                    'Not a valid attribute function see help(custom_plot) for help.')

        if type == 'attribute_plot':
            try:
                return attribute_plot(self, units, channels, attribute_function)
            except:
                raise Exception(
                    'Not a valid attribute function see help(custom_plot) for help.')

        # TODO include 3D plot

        raise Exception(
            'Not a valid plot type, see help(custom_plot) for help.')

    def ISI(self, unit):
        """
        Plots the distribtion of the inter-spike-intervals

        Parameters
        ----------
        unit : int or str
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
            Handle to the figure

        """

        spikes = self.data.get_unit_spikes(unit)
        isi = 1000*np.diff(spikes)/float(self.data.metadata['imSampRate'])

        fig, ax = plt.subplots(1)
        ax.hist(isi, bins=np.linspace(0, 25, 25), density=True, range=(0, 25))
        ax.set_xlabel('Inter-spike-interval (ms)')
        ax.set_ylabel('Density')

        return fig

    def correlogram(self, units=None):

        # Infer units if necesary
        units, _ = self.data._infer_unit_channels(units, [0, 1, ])

        # Make figure
        fig, axs = plt.subplots(nrows=len(units), ncols=len(units),
                                tight_layout=True, figsize=[8, 8], sharex=True)

        # Colors
        colors = self._color_map(len(units))

        # Make the correlograms and plot them
        for X, unit_X in enumerate(units):
            for Y, unit_Y in enumerate(units):

                # Figure out color
                color = 'grey'
                if X == Y:
                    color = colors[X]

                # Calculate and plot
                interval = self._correlogram(unit_Y, unit_X)
                axs[Y, X].hist(interval, bins=np.linspace(0, 25, 25),
                               density=True, range=(0, 25), color=color)

                # Formatting
                if Y == len(units)-1:
                    axs[Y, X].set_xlabel('Time (ms)')
                if X == 0:
                    axs[Y, X].set_ylabel(f'Unit: {unit_Y} (density)')
                if Y == 0:
                    axs[Y, X].set_title(f'Unit: {unit_X}')

        # Finaly formatting
        plt.suptitle('Correlogram')

        return fig

    def raw_spike_sample(self, unit, channels, sample_n = 25, window = 3):
        """
        Plots a random sample of 'sample_n' waveforms. This is just to look
        at the whitened or unfiltered raw signal. To get an idea if your 
        spikes really belong to a unit or are part of the noise.

        Parameters
        ----------
        unit : int
            The unit that your are interested in
        channel: int
            The channel that will be plot
        sample_n: int
            The number of waveform
        window: float
            The window in ms (on both sides of the spike)

        Returns
        -------
        ax : Matplotlib axes
            Axes handle of the plot
        """

        # Convert channels
        if channels.__class__ == int:
            channels = [channels]

        # Select the spikes
        spikes = self.data.get_unit_spikes(unit)

        # Make the figure
        fig, axs = plt.subplots(1, len(channels), figsize = (6*len(channels), 10), 
            facecolor='k', tight_layout = True, sharex = True)
        if not axs.__class__ == np.ndarray:
            axs = [axs]

        # For every channel do this
        for i, chan in enumerate(channels):

            # Grab waveform
            waveforms = self.data.get_waveform(spikes, channel=chan, average=False, 
                sample=sample_n, window = window).transpose()
            to_plot = pd.DataFrame(waveforms)

            # Deal with the offset
            offset = -0.5 * to_plot.min().min()
            offseter = np.linspace(0, to_plot.shape[1]*offset, to_plot.shape[1])
            to_plot = to_plot + offseter

            # Update the axis
            correction = ((len(to_plot)/2)-1)/(len(to_plot)/2)
            to_plot.index = np.linspace(-window, correction*window, len(to_plot))

            # Plot
            to_plot.plot(linewidth=0.5, legend=False, color='w', ax=axs[i])
            axs[i].set_facecolor('k')
            axs[i].tick_params(axis='x', colors='w')
            axs[i].set_xlabel('Time (ms)', color='w')
            axs[i].set_title(f'Channel {chan}', color='w')
            axs[i].set_yticks([])

        # Final formatting
        axs[0].set_ylabel(f'Trials (n = {sample_n}, uniformly picked)', color='white')

        return fig, axs

    def raw_unit_sample(self, unit, sample_n = 1000, window=3):
        """
        Plots the average respond (averaged over sample_n spikes) on
        all channels.

        Parameters
        ----------
        unit : int
            The unit that your are interested in
        sample_n: int
            The number of spikes we'll average over to get the waveform
        window: float
            The window in ms (on both sides of the spike)

        Returns
        -------
        ax : Matplotlib axes
            Axes handle of the plot
        """

        # Select the spikes
        spikes = self.data.get_unit_spikes(unit)

        # Make the figure
        fig, axs = plt.subplots(1, 4, figsize = (20, 12), 
            facecolor='k', tight_layout = True, sharex = True, sharey=True)

        # For every channel do this
        for i in range(4):

            channels = [j for j in range(i, 383, 4)]
            to_plot = pd.DataFrame()
            for channel in channels:
                to_plot[channel] = self.data.get_waveform(spikes, 
                    channel = channel, average = True, sample = sample_n, 
                    window = window)

            # Deal with the offset
            offset = 0.5 * self.data.clusters.Amplitude.loc[unit]
            offseter = np.linspace(0, to_plot.shape[1]*offset, to_plot.shape[1])
            to_plot = to_plot + offseter

            # Update the axis
            correction = ((len(to_plot)/2)-1)/(len(to_plot)/2)
            to_plot.index = np.linspace(-window, correction*window, len(to_plot))

            # Plot
            to_plot.plot(linewidth=0.5, legend=False, color='w', ax=axs[i])
            axs[i].set_facecolor('k')
            axs[i].tick_params(axis='x', colors='w')
            axs[i].set_xlabel('Time (ms)', color='w')
            axs[i].set_yticks([])

        # Final formatting
        axs[0].set_ylabel('Channels', color='white')

        return fig, axs

    def plot_3D(self, units=None, channels=None,
                attribute_function=atf._calc_amplitude):
        """
        NOTE DONE
        """

        # Infer units if necesary
        units, channels = self.data._infer_unit_channels(units, channels)

        # Check that channels is 3
        if len(channels) > 3:
            channels = channels[:3]
            print(f'Using only channels: {channels} for 3D plot.')

        return plot_3D(self, units, channels, attribute_function)

    def spike_raster(self, sample = 100000):
        """
        Will plot a uniformly sampled selection of 'sample' spike from
        the dataset. The color refers to the spike amplitude.

        """
        # Grab the data
        spikeID = self.data.spikeID

        # Only interested in spikes that have not been drown out
        indexer = np.zeros(spikeID.shape).astype(bool)
        for i in self.data.clusters.index:
            indexer[spikeID==i] = True

        # Clean
        spikeID = spikeID[indexer]
        spikeTimes = self.data.spikeTimes[indexer]
        spikeAmp = self.data.spikeAmp[indexer]

        # Sample
        if sample<len(spikeID):
            spikeID = spikeID[::len(spikeID)//sample]
            spikeTimes = spikeTimes[::len(spikeTimes)//sample]
            spikeAmp = spikeAmp[::len(spikeAmp)//sample]
        spikeTimes = spikeTimes/float(self.data.metadata['imSampRate'])

        # Get the channel
        channel = spikeID.copy()
        for i in np.unique(spikeID):
            channel[spikeID==i] = self.data.clusters.loc[i].mainChannel

        # Make random permutations of the spikeID
        random = np.random.permutation(np.unique(spikeID))
        rand_spikeID = spikeID.copy()
        for i, val in enumerate(random):
            rand_spikeID[spikeID==val] = i

        # Plot
        fig, axs = plt.subplots(1, 4, figsize = [20, 10], tight_layout = True,
            sharex = True, sharey = True, facecolor = 'k')
        for i in range(1, 5):
            indexer = (channel-i)%4==0
            axs[i-1].scatter(x=spikeTimes[indexer], 
                y=channel[indexer], c=spikeAmp[indexer],
                vmin=spikeAmp.mean()-2*spikeAmp.std(), 
                vmax=spikeAmp.mean()+2*spikeAmp.std(), 
                marker = '|',
                alpha=0.8,
                cmap='cool')
            axs[i-1].set_facecolor('k')
            axs[i-1].tick_params(axis='x', colors='w')
            axs[i-1].set_xlabel('Time (s)', color='w')
        axs[0].tick_params(axis='y', colors='w')
        axs[0].set_ylabel('Channel #', color='w')

    def cluster_progress(self):
        """
        Quick overview of the dataset.

        """

        # First plot overview of the entire dataset
        plt.figure(figsize=(9, 9), tight_layout = True)
        sns.scatterplot(x='Amplitude', y='mainChannel', size='spikeCount', 
            hue='KSLabel', data=self.data.clusters)
        done = self.data.clusters.loc[self.data.clusters.done, :]
        sns.scatterplot(x='Amplitude', y='mainChannel', size='spikeCount', 
            hue='KSLabel', edgecolor='red', data=done, linewidth=2, 
            legend=False)

    def add_unit(self, unit):

        # Add the unit to each plot
        _ = [i.add_unit(unit) for i in self.data.linked_plots]

    def remove_unit(self, unit):

        # Remove the unit from each plot
        _ = [i.remove_unit(unit) for i in self.data.linked_plots]

    def roll(self, n=1):
        """
        Move up one channel for all plots

        """
        # Grab channels
        _ = [i.roll(n) for i in self.data.linked_plots]

    def change_channel(self, index, new_channel):

        # Change the channel in each plot
        _ = [i.change_channel(index, new_channel)
             for i in self.data.linked_plots]

    def add_timestamps(self, stamps, color='orange'):
        """
        Will add timestamps to the plots that allow this

        paramters
        ---------
        stamps: np.ndarray pd.DataFrame or list
            List of stamps (in sec)
            If you input a Pandas DataFrame, it will look for a column "Start""
        """

        # Check the timestamps
        if stamps.__class__ == pd.DataFrame:
            stamps = stamps.Start.values

        # Are they really in seconds (not ms)
        if max(stamps) > 100000:
            ans = input("Did you input stamps in ms instead of s? (y/n) ")
            if ans == 'y':
                stamps = stamps/1000

        _ = [i.add_timestamps(stamps, color) for i in self.data.linked_plots]

    def focus_unit(self, unit_id, show_neighbors = None):
        """
        Will update all plots to just show one unit on the most appropriate
        channels.

        parameters
        ----------
        unit_id: int
            The unit you are interested in
        show_neighbors : None or Float
            If not None, will plot neighbors with similarity > this number [0, 1]
        """

        # Are there any plots?
        if len(self.data.linked_plots)==0:
            print('There are not plots open')
            return

        # Grab all units in all plots and remove them
        all_units = []
        for plot in self.data.linked_plots:
            all_units.extend(plot.units)
        _ = [self.remove_unit(unit) for unit in set(all_units)]

        # Set the right channels
        channel = int(self.data.clusters.loc[unit_id].mainChannel)
        n_channels = len(self.data.linked_plots[0].channels)
        if n_channels%2 == 0:
            start = channel-int(n_channels/2-1)
            stop = channel + int(n_channels/2)+1
        else:
            start = channel - int(n_channels/2-0.5)
            stop = channel + int(n_channels/2+0.5)
        channels = [i for i in range(start, stop)]
        if channels[0]<0:
            channels = [i + -1*channels[0] for i in channels] 
        for i, channel in enumerate(channels):
            self.change_channel(i, channel)

        # Plot the unit
        self.add_unit(unit_id)

        # Display the neighbors
        neighbors = self.data.sort.neighbors(unit_id, 3)
        print('These neurons are close:')
        print(neighbors.sort_values('similarity', ascending = False))

        # Add neighbors tot he plot
        if not show_neighbors is None:
            indexer = neighbors[neighbors.similarity>show_neighbors].index.values
            self.add_unit(indexer)

    def _plot_waveform(self, data, ax=None, color='red'):
        """
        Responsible for plotting one or more waveforms

        Parameters
        ----------
        data : 1-D or 2-D NumPy array
            Contains the waveform(s) that should be plotted. If multiple 
            waveforms should be plotted every row should be a waveform.
        ax : Matplotlib axes object, optional
            The axes that the waveforms will be plotted on.
        color : str or RGB values, optional
            Color in which the waveforms will be plot

        Returns
        -------
        ax : Matplotlib axes
            The Matplotlib axes object of the waveform plot.

        """

        # Do we have to make new axes?
        if ax is None:
            fig, ax = plt.subplots(1)

        # Plot
        handles = ax.plot(data.transpose(), color=color, linewidth=0.1,
                          alpha=0.8)

        # Formatting
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        return handles

    def _color_map(self, n: int):
        """

        Parameters
        ----------
        n : int
            Number of colors to return

        Returns
        -------
        Seaborn color palette of n colors.

        """

        return sns.color_palette(self.palette, n)

    def _correlogram(self, unit_1, unit_2):

        unit_1 = self.data.get_unit_spikes(unit_1)
        unit_2 = self.data.get_unit_spikes(unit_2)

        # For unit 1, not interested in stamps at the end
        unit_1 = unit_1[unit_1 < unit_2[-1]]

        interval = np.array([unit_2[unit_2 > i][0]-i for i in unit_1])
        interval = 1000*interval/float(self.data.metadata['imSampRate'])

        return interval


class plot_object:
    """
    Main plot_object class, all other plot objects are derived from this class.
    """

    def __init__(self, plot_object, units, channels, attribute_function = None,
                 other_data = None, peri_event_window = (-5.0, 5.0)):

        # Store data and handles
        self.plotter = plot_object
        self.data = plot_object.data
        self.facecolor = plot_object.facecolor
        self.axcolor = plot_object.axcolor
        self.plot_handles = {}
        self.units = []
        self.colors = []
        self.other_data = other_data
        self.update_diagonal = True
        self.peri_event_window = peri_event_window

        # Figure out what attribute function to use
        if not attribute_function is None:
            self.attribute_function = attribute_function

        # Infer the units and the channels
        units, channels = self.data._infer_unit_channels(units, channels)
        self.channels = channels  # (Units are added below)

        # make figure
        self.fig = self.make_figure()
        self.is_closed = False
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('key_press_event', self._key_press)

        # Run the startup
        self.startup()

        # update units
        _ = [self.add_unit(unit) for unit in units]

        # Add yourself to the data object
        self.data.linked_plots.append(self)

    def startup(self):
        """
        Function runs at construction. Primarily used to set the plot handles.
        This is because the plot handles for waveform and scatter plots are
        differently organised.

        Returns
        -------
        None.

        """

        # Add empty handles for the plots
        for i, chan in enumerate(self.channels):
            self.plot_handles[i] = []

    def make_figure(self):
        """
        Should make the figure and return the figure handle.
        """

        pass

    def add_unit(self, units):
        """
        Responsible for adding one or more units to every axes in the plot.
        It sould:
            - Draw the unit on every axes
            - Include the unit in the object (self.units.append)

        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:

            # Check if unit is not allready in there
            if unit in self.units:
                continue

            # Grab spikes
            spikes = self.data.get_unit_spikes(unit)

            # Figure out color
            color = self._get_color()
            self.colors.append(color)

            # Draw every channel
            for i, channel in enumerate(self.channels):
                data = self._get_plot_data(spikes, channel)
                self._add_unit_to_axes(data, i, color)

            # Add the unit
            self.units.append(unit)

        # Rescale axis if required
        self._rescale_axes()

        # Add (update) the legend
        self._add_legend()

    def remove_unit(self, units):
        """
        Responsible for removing one or more units from all linked
        axes.
        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:
            # Check if unit in there
            if unit not in self.units:
                return None

            # Get index of unit
            index = self.units.index(unit)

            # Remove the units
            self.units.pop(index)
            self.colors.pop(index)

            # Remove the plots
            for i, channel in enumerate(self.channels):

                # Remove the unit
                self._remove_unit_from_axes(index, i)

                # Also remove the handles
                temp = self.plot_handles[i].pop(index)

        # Update the legend
        self._add_legend()

        # Rescale axis if required
        self._rescale_axes()

    def change_channel(self, index, new_channel):

        # Update the channel list
        self.channels[index] = new_channel

        # Replot all units on the axis with index=index
        for i, unit in enumerate(self.units):  # for all units

            # Remove unit from axis
            self._remove_unit_from_axes(i, index)

            # Add the new unit
            data = self._get_plot_data(unit, new_channel)
            self._add_unit_to_axes(data, index, self.colors[i], index=i)

        # Set the axes labels
        self._label_plot(index)

    def roll(self, n=1):
        """
        Will move every plot 'n' channels up.
        """

        channels = self.channels[n:]
        for i in range(n):
            channels.append(channels[-1]+1)

        for i, channel in enumerate(channels):
            self.change_channel(i, channel)

    def add_timestamps(self, stamps, color):
        """
        Some plots (e.g. peri-event histograms and time plots) will let you plot
        timestamps
        """
        pass

    def update(self, unit):
        """
        Primarily used for updating if a different object (such as the main
        PKS_dataset object) make changes to the dataset.

        Parameters
        ----------
        unit : int or str
            Unit identifyer. This is the unit that will be re-plotted.

        Returns
        -------
        None.

        """

        # Check if this unit is plotted anyway
        if not unit in self.units:
            return None

        # Only do this if the figure is open
        if not self.is_closed:
            self.remove_unit(unit)
            self.add_unit(unit)

    def _get_plot_data(self, spikes_or_unit, channel):

        pass

    def _add_unit_to_axes(self, data_X, data_Y, axes_X, axes_Y, color, index=None):
        # If the axes are one-dimensional, only use data_X and data_Y

        pass

    def _remove_unit_from_axes(self, index, axes_X, axes_Y=None):
        """
        Removes a unit from the axes. Axes_X and Axes_Y ares used for
        attribute (scatter) plots. Time plots use only axes_X.

        """

        # We want this function to work on both 2-D and 1-D axes
        if len(self.axs.shape) == 2:

            # Sometimes we don't want the diagonal to be updated
            if (not self.update_diagonal) & (axes_X == axes_Y):
                return None

            temp = self.plot_handles[axes_X][axes_Y][index]
            temp.remove()
            self.plot_handles[axes_X][axes_Y][index] = []
        else:
            temp = self.plot_handles[axes_X][index]
            if temp.__class__ == list:
                _ = [j.remove() for j in temp]
            else:
                temp.remove()
            self.plot_handles[axes_X][index] = []

    def _get_color(self):
        """
        Controls color selection. The colormap is set in the main pks_plotting
        class, it is a Seaborn map.

        Returns
        -------
        truple

        """

        # Get the colormap
        colormap = self.plotter.colormap
        for color in colormap:
            if color not in self.colors:
                return color

        # return random color
        return tuple(np.random.random(3))

    def _label_plot(self, axes_X):
        """
        Takes the axes at index 'axes_X' and labels the X and Y axes as well as the title.
        """

        pass

    def _add_legend(self):
        """
        Responsible for adding a legend to the axes in ax.

        Parameters
        ----------
        ax : matplotlib axes
            The axes that a legend will be added to.
        units : 1-D NumPy array
            List of units that are included in the plot.

        Returns
        -------
        None.

        """

        # Figure out the unit names
        units = self.units
        unit_names = []
        for unit in units:
            unit_name = f'Unit: {unit}'
            if unit > 10000:
                if str(unit)[:3] == '999':
                    unit_name = f'Unit: {unit-99900000} (tagged waveforms)'
            unit_names.append(unit_name)
        colors = self.colors
        elements = [Patch(facecolor=colors[i], edgecolor=colors[i],
                          label=unit_names[i])
                    for i in range(len(self.units))]
        self.axs.ravel()[-1].legend(handles=elements)

    def _on_close(self, event):
        """
        Runs after fiture close
        """
        self.is_closed = True
        self.data._check_linked_plots()

    def _key_press(self, event):
        """
        Runs after a keypress on the figure

        """

        self.event = event

    def _rescale_axes(self):
        pass

    def _default_formatting(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor(self.facecolor)
        ax.spines['left'].set_color(self.axcolor)
        ax.spines['bottom'].set_color(self.axcolor)
        ax.xaxis.label.set_color(self.axcolor)
        ax.yaxis.label.set_color(self.axcolor)
        ax.tick_params(colors=self.axcolor)
        ax.title.set_color(self.axcolor)

    def _onscroll(self, event, x_axis = False):
        # Will attempt to reset the Y axis only

        # Let's see if we can find the axes
        ax = event.inaxes

        xdata = event.xdata
        ydata = event.ydata
        if xdata is not None and ydata is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        xcenter = ((xlim[1] + xlim[0]) / 2 - xdata) / (xlim[1] - xlim[0])
        ycenter = ((ylim[1] + ylim[0]) / 2 - ydata) / (ylim[1] - ylim[0])

        if event.button == 'up':
            ax.set_ylim(ylim[0] * 0.9 - ycenter * (ylim[1] - ylim[0]) * 0.1, ylim[1] * 0.9 - ycenter * (ylim[1] - ylim[0]) * 0.1)
            if x_axis:
                ax.set_xlim(xlim[0] * 0.9 - xcenter * (xlim[1] - xlim[0]) * 0.1, xlim[1] * 0.9 - xcenter * (xlim[1] - xlim[0]) * 0.1)
        else:
            ax.set_ylim(ylim[0] * 1.1 - ycenter * (ylim[1] - ylim[0]) * 0.1, ylim[1] * 1.1 - ycenter * (ylim[1] - ylim[0]) * 0.1)
            if x_axis:
                ax.set_xlim(xlim[0] * 1.1 - xcenter * (xlim[1] - xlim[0]) * 0.1, xlim[1] * 1.1 - xcenter * (xlim[1] - xlim[0]) * 0.1)


        plt.draw()


class plot_3D(plot_object):

    def startup(self):

        # There will be only one plot handle (for now)
        self.plot_handles = {0: []}

    def make_figure(self):

        # TODO allow for multiple axes
        fig = plt.figure(tight_layout = True, facecolor = self.facecolor)
        ax = plt.axes(projection = '3d', facecolor = self.facecolor)
        self._default_formatting(ax)

        # Label axes
        ax.set_xlabel(f'Channel: {self.channels[0]}')
        ax.set_ylabel(f'Channel: {self.channels[1]}')
        ax.set_zlabel(f'Channel: {self.channels[2]}')

        # Store the axes
        self.axs = np.array([ax])

        return fig

    def add_unit(self, units):
        """
        Responsible for adding a unit to a 3D plot
        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:

            # Check if unit is not allready in there
            if unit in self.units:
                return None

            # Grab spikes
            spikes = self.data.get_unit_spikes(unit)

            # Figure out color
            color = self._get_color()
            self.colors.append(color)

            # Get the data and plot
            x = self._get_plot_data(spikes, self.channels[0])
            y = self._get_plot_data(spikes, self.channels[1])
            z = self._get_plot_data(spikes, self.channels[2])
            self._add_unit_to_axes(x, y, z, color)

            # Add the unit
            self.units.append(unit)

        # Add (update) the legend
        self._add_legend()

    def remove_unit(self, units):
        """
        Responsible for removing one or more units from all linked
        axes.
        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == ndarray):
            units = [units]

        for unit in units:
            # Check if unit in there
            if unit not in self.units:
                return None

            # Get index of unit
            index = self.units.index(unit)

            # Remove the units
            self.units.pop(index)
            self.colors.pop(index)

            # Remove the plots
            # for i, channel in enumerate(self.channels):
            # TODO replace this line with something to go over all axes

            # Remove the unit
            self._remove_unit_from_axes(index, 0)

            # Also remove the handles
            temp = self.plot_handles[0].pop(index)

        # Update the legend
        self._add_legend()

    def _get_plot_data(self, spikes, channel):

        # Make the data
        temp = self.data.get_waveform(spikes, channel, average=False,
                                      sample=self.plotter.plot_max)

        return self.attribute_function(temp, channel)

    def _add_unit_to_axes(self, x, y, z, color):
        """
        Ads the units to the axes
        """

        handle = self.axs[0].scatter3D(x, y, z, color=color,
                                       s=10, marker='.', alpha=0.8)

        self.plot_handles[0].append(handle)


class attribute_plot(plot_object):

    def startup(self):

        self.plot_handles = {}
        for X, chan_X in enumerate(self.channels):
            self.plot_handles[X] = {}
            for Y, chan_Y in enumerate(self.channels):
                self.plot_handles[X][Y] = []

    def make_figure(self):

        # Make the figure
        n_channels = len(self.channels)
        fig, axs = plt.subplots(figsize=[10, 10], nrows=n_channels,
                                ncols=n_channels, tight_layout=True,
                                facecolor=self.facecolor)
        onscroll = partial(self._onscroll, x_axis=True)
        fig.canvas.mpl_connect('scroll_event', onscroll)

        # # Share X and Y, but not on the diagonal
        # for x in range(n_channels):
        #     for y in range(n_channels):
        #         if x == y:
        #             continue
        #         axs[x, y].sharex(axs[0, 1])
        #         axs[x, y].sharey(axs[0, 1])
        #         axs[x, y].set_facecolor(self.facecolor)

        # Formatting
        for i, ax in enumerate(axs[0, :]):
            ax.set_title(f'Channel: {self.channels[i]}')
        for i, ax in enumerate(axs[:, 0]):
            ax.set_ylabel(f'Channel: {self.channels[i]}')
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            self._default_formatting(ax)

        # Add keypresses to the axis?

        # Store the axes
        self.axs = axs

        return fig

    def add_unit(self, units):

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:
            # Check if unit is not allready in there
            if unit in self.units:
                continue

            # Figure out color
            color = self._get_color()
            self.colors.append(color)

            # Make data_X
            spikes = self.data.get_unit_spikes(unit)
            data = [self._get_plot_data(spikes, channel)
                    for channel in self.channels]

            # Draw every channel
            for X, chan_X in enumerate(self.channels):
                for Y, chan_Y in enumerate(self.channels):

                    # Plot X and Y data on the corrext axes and store the handle
                    self._add_unit_to_axes(data[X], data[Y], X, Y, color)

            # Add the unit
            self.units.append(unit)

            # Add a legend
            self._add_legend()

    def remove_unit(self, unit):
        """

        Removes unit from:
            - The unit list
            - The color list
            - Each axes in the figure
            - Removes the handles itself as well

        """

        # Check if unit in there
        if unit not in self.units:
            return None

        # Get index of unit
        index = self.units.index(unit)

        # Remove the units
        self.units.pop(index)
        self.colors.pop(index)

        # Remove the plots
        for X, chan_X in enumerate(self.channels):
            for Y, chan_Y in enumerate(self.channels):

                # Sometimes we don't want the diagonal to be updated
                if (not self.update_diagonal) & (X == Y):
                    continue

                self._remove_unit_from_axes(index, X, Y)

                # Pop the handles
                self.plot_handles[X][Y].pop(index)

        # Update the legend
        self._add_legend()

    def change_channel(self, index, new_channel):

        # Update the channel list
        self.channels[index] = new_channel

        # Replot all cells horizontally
        for i, unit in enumerate(self.units):  # for all units

            # Unfortunately have to get all the channels
            data = [self._get_plot_data(unit, channel)
                    for channel in self.channels]

            # Replace horizontally
            Y = index
            for X, ax in enumerate(self.axs[Y, :]):
                # remove the unit from the axis
                self._remove_unit_from_axes(i, X, Y)
                self._add_unit_to_axes(
                    data[X], data[Y], X, Y, self.colors[i], index=i)

            # Replace Vertically
            X = index
            for Y, ax in enumerate(self.axs[:, X]):
                # remove the unit from the axis
                self._remove_unit_from_axes(i, X, Y)
                self._add_unit_to_axes(
                    data[X], data[Y], X, Y, self.colors[i], index=i)

        # Set the axes labels
        self.axs[0, index].set_title(f'Channel: {self.channels[index]}')
        self.axs[index, 0].set_ylabel(f'Channel: {self.channels[index]}')

    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__ == int) | (spikes_or_unit.__class__ == str) | (spikes_or_unit.__class__ == np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit

        # Make the data
        temp = self.data.get_waveform(spikes, channel, average=False,
                                      sample=self.plotter.plot_max)

        return self.attribute_function(temp, channel)

    def _add_unit_to_axes(self, data_X, data_Y, axes_X, axes_Y, color, index=None):

        # Sometimes we don't want the diagonal to be updated
        if (not self.update_diagonal) & (axes_X == axes_Y):
            return None

        # Do the plotting
        handle = self.axs[axes_Y, axes_X].scatter(x=data_X, y=data_Y, marker='.',
                                                  color=color, s=10, alpha=0.8)

        if index is None:
            self.plot_handles[axes_X][axes_Y].append(handle)
        else:
            self.plot_handles[axes_X][axes_Y][index] = handle


class pca_plot(attribute_plot):
    """
    The PCA plot is a special case of the attribute plot. This is because
    the axis (PCs) have to be re-calculated when a channel is placed or
    changed. For standard attributed plots, they are fixed.

    """

    def startup(self):

        # Make sure the handles make sense
        self.plot_handles = {}
        for X, chan_X in enumerate(self.channels):
            self.plot_handles[X] = {}
            for Y, chan_Y in enumerate(self.channels):
                self.plot_handles[X][Y] = []

        # Overwrite the center plots, plot the component itself.
        for i in range(len(self.channels)):
            self.axs[i, i].clear()
            self.plot_handles[i, i] = self.axs[i, i].plot(
                self.other_data[i].components_.transpose())
            self.axs[i, i].set_xticks([])
            self.axs[i, i].set_yticks([])
            self.axs[i, i].set_facecolor(self.facecolor)
        self.axs[0, 0].set_title(f'Channel: {self.channels[0]}')
        self.axs[0, 0].set_ylabel(f'Channel: {self.channels[0]}')
        self._add_legend()

        # Prevent the diagonal plots from being overwritten
        self.update_diagonal = False

    def change_channel(self, index, new_channel):
        """

        The PCA change_channel method is different from the default one, because
        we have to update the PCA itself as well as the principal component plot.

        """

        # Update the channel list
        self.channels[index] = new_channel

        # Update the PCA
        self.other_data[index] = self.data.channel_pca(new_channel)

        # Replot all cells horizontally
        for i, unit in enumerate(self.units):  # for all units

            # Unfortunately have to get all the channels
            data = [self._get_plot_data(unit, channel)
                    for channel in self.channels]

            # Replace horizontally
            Y = index
            for X, ax in enumerate(self.axs[Y, :]):
                # remove the unit from the axis
                self._remove_unit_from_axes(i, X, Y)
                self._add_unit_to_axes(
                    data[X], data[Y], X, Y, self.colors[i], index=i)

            # Replace Vertically
            X = index
            for Y, ax in enumerate(self.axs[:, X]):
                # remove the unit from the axis
                self._remove_unit_from_axes(i, X, Y)
                self._add_unit_to_axes(
                    data[X], data[Y], X, Y, self.colors[i], index=i)

        # Set the axes labels
        self.axs[0, index].set_title(f'Channel: {self.channels[index]}')
        self.axs[index, 0].set_ylabel(f'Channel: {self.channels[index]}')

        # Update the plot
        self.plot_handles[index, index][0].remove()
        self.plot_handles[index, index] = self.axs[index, index].plot(
            self.other_data[index].components_.transpose())

    def attribute_function(self, data, channel):

        index = self.channels.index(channel)

        return self.other_data[index].transform(data)


class waveform_plot(plot_object):

    def make_figure(self):

        # Make a figure for the waveforms
        cols = 5
        while not ((len(self.channels) % cols == 0) or (cols == 1)):
            cols -= 1
        rows = int(len(self.channels)/cols)
        fig, axs = plt.subplots(figsize=[4*cols, 4*rows], nrows=rows,
                                ncols=cols, tight_layout=True, sharex=True,
                                sharey=False, facecolor=self.facecolor)
        axs = axs.ravel()
        fig.canvas.mpl_connect('scroll_event', self._onscroll)

        # Store the axes handles
        self.axs = axs
        for ax in axs:
            self._default_formatting(ax)

        # Add titles to the axe
        for i, chan in enumerate(self.channels):
            self._label_plot(i)

        return fig

    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__ == int) | (spikes_or_unit.__class__ == str) | (spikes_or_unit.__class__ == np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit

        return self.data.get_waveform(spikes, channel, average=False, sample=self.plotter.plot_max)

    def _add_unit_to_axes(self, data_X, axes_X, color, index=None):
        # If the axes are one-dimensional, only use data_X and data_Y

        # Do the plotting
        handle = self.plotter._plot_waveform(data_X[:, 30:60], self.axs[axes_X],
                                             color=color)

        if index is None:
            self.plot_handles[axes_X].append(handle)
        else:
            self.plot_handles[axes_X][index] = handle

    def _label_plot(self, axes_X):

        # Find the name of the channel
        chan = self.channels[axes_X]

        # Grab the ax
        ax = self.axs[axes_X]
        ax.set_title(f'Channel: {chan}')

    def _rescale_axes(self):

        # figure out the max amplitude
        max_amp = 0
        for i in self.units:
            amp = self.data.clusters.loc[i].Amplitude
            if amp>max_amp:
                max_amp = amp

        # Rescale
        #current_lim = self.axs[0].get_ylim()
        new_lim = (1.8 * -1*max_amp, 1.3*max_amp)
        self.axs[0].set_ylim(new_lim)


class time_plot(plot_object):
    """
    Time_plots plot data in time. Time is on the X-axis, while the parameter
    of interest is on the Y-axis. Different channels are represented in
    different plots, stacked vertically. 

    """

    def make_figure(self):

        # Make the figure
        n_channels = len(self.channels)
        fig, axs = plt.subplots(figsize=[10, 12], nrows=n_channels,
                                ncols=1, tight_layout=True, sharex=True,
                                sharey=False, facecolor=self.facecolor)
        axs = axs.ravel()
        axs[-1].set_xlabel('Time (s)')
        self.axs = axs
        fig.canvas.mpl_connect('scroll_event', self._onscroll)

        # Formatting
        for i, ax in enumerate(axs):
            self._label_plot(i)
            self._default_formatting(ax)

        return fig

    def add_timestamps(self, stamps, color='orange'):
        """
        Will add timestamps to the plot
        """

        for ax in self.axs:
            ylim = ax.get_ylim()
            ax.vlines(stamps, ylim[0], ylim[1], linestyle='--', color=color, 
                linewidth=0.3, alpha=1)

    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__ == int) | (spikes_or_unit.__class__ == str) | (spikes_or_unit.__class__ == np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit

        # Get the data
        data, X = self.data.get_waveform(spikes, channel,
                                         average=False, sample=self.plotter.plot_max,
                                         return_spikes=True)
        X = X/float(self.data.metadata['imSampRate'])
        X = X*self.data.sync_time_coefficients[0] + self.data.sync_time_coefficients[1]
        data = self.attribute_function(data)

        return data, X

    def _add_unit_to_axes(self, data_X, axes_X, color, index=None):
        # If the axes are one-dimensional, only use data_X and data_Y

        # Data_X is a tuple in this case
        data = data_X[0]
        X = data_X[1]

        # Do the plotting
        handle = self.axs[axes_X].scatter(x=X, y=data, marker='.',
                                          color=color, s=20, alpha=0.8)

        if index is None:
            self.plot_handles[axes_X].append(handle)
        else:
            self.plot_handles[axes_X][index] = handle

    def _label_plot(self, axes_X):

        # Grab the ax
        ax = self.axs[axes_X]
        ax.set_ylabel(f'Channel: {self.channels[axes_X]}')
        ax.set_yticks([])


class peri_event_plot(plot_object):
    """
    Peri_event_plots plot the unit stamps surounding an event.

    """

    def make_figure(self):

        # Make the figure
        fig, axs = plt.subplots(figsize=[10, 12], nrows=2,
                                ncols=1, tight_layout=True, sharex=True,
                                sharey=False, facecolor=self.facecolor)
        axs = axs.ravel()
       
        self.axs = axs

        # Formatting
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('p fires')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Trial')
        axs[0].axis('off')
        self._default_formatting(axs[1])

        plt.suptitle('Peri Event Plot')

        return fig

    def startup(self):

        # Add empty handles for the plots
        self.plot_handles = {'event_plot': [], 'histogram': []}

    def add_unit(self, units):
        """
        Responsible for adding one or more units to every axes in the plot.
        It sould:
            - Draw the unit on every axes
            - Include the unit in the object (self.units.append)

        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:

            # Check if unit is not allready in there
            if unit in self.units:
                continue

            # Grab spikes
            spikes = self.data.get_unit_spikes(unit, return_real_time = True)

            # Figure out color
            color = self._get_color()
            self.colors.append(color)

            # Draw the unit
            event_data, hist, bins = self._get_plot_data(spikes, self.other_data,
                self.peri_event_window)
            self._add_unit_to_axes(event_data, hist, bins, color)

            # Add the unit
            self.units.append(unit)

        # Add (update) the legend
        self._add_legend()

        # Rescale axis
        self._rescale_axes()

    def change_channel(self, *kwargs):

        pass

    def roll(self, *kwargs):

        pass

    def remove_unit(self, units):
        """
        Responsible for removing one or more units from all linked
        axes.
        """

        # Multiple units or just 1?
        if not (units.__class__ == list) | (units.__class__ == tuple) | (units.__class__ == np.ndarray):
            units = [units]

        for unit in units:
            # Check if unit in there
            if unit not in self.units:
                return None

            # Get index of unit
            index = self.units.index(unit)

            # Remove the units
            self.units.pop(index)
            self.colors.pop(index)

            # Remove the unit
            self._remove_unit_from_axes(index)

        # Rescale axis
        self._rescale_axes()

        # Update the legend
        self._add_legend()

    def _rescale_axes(self):
        """
        rescales the y-axis of the histogram to best fit the data
        """

        # Get the max value
        max_value = 0
        for i in self.plot_handles['histogram']:
            temp = i.datavalues.max()
            if temp>max_value:
                max_value = temp

        # Rescale
        self.axs[1].set_ylim((0, max_value+0.1))

    def _remove_unit_from_axes(self, index):
        """
        Removes a unit from both axes
        """

        # Grab the handles
        event_handles = self.plot_handles['event_plot'].pop(index)
        hist_handle = self.plot_handles['histogram'].pop(index)

        # Delete all event plots
        _ = [i.remove() for i in event_handles]

        # Delete the histogram
        hist_handle.remove()

    def _get_plot_data(self, spikes_or_unit, stamps, peri_event_window):

        # Grab spikes
        if (spikes_or_unit.__class__ == int) | (spikes_or_unit.__class__ == str) | (spikes_or_unit.__class__ == np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit

        # Figure out the window and bin width
        bin_size = 0.1
        if bin_size > 0.5 * (abs(peri_event_window[0]) + abs(peri_event_window[1])):
            bin_size = 0.001
        num_bins = int((peri_event_window[1] - peri_event_window[0]) / bin_size)
        
        # Grab the event data
        event_data = []
        for stamp in stamps:
            relative_window = np.array(peri_event_window) + stamp
            event_data.append(spikes[(relative_window[0] <= spikes) & (spikes <= relative_window[1])] - stamp)

        # Grab the histogram data
        hist, bins = np.histogram(spikes - stamps[:, np.newaxis], bins=num_bins,
                                      range=peri_event_window)
        hist = hist/len(stamps)

        return event_data, hist, bins

    def _add_unit_to_axes(self, event_data, hist, bins, color, store_handle = True):

        # Plot the event plot
        event_plot_handles = self.axs[0].eventplot(event_data, color=color, linewidths=2,
            alpha = 0.8)

        # Plot the histogram
        _, _, hist_handle = self.axs[1].hist(bins[:-1], bins, weights=hist, edgecolor=color,
                                facecolor = color, alpha=0.5)

        if store_handle:
            self.plot_handles['event_plot'].append(event_plot_handles)
            self.plot_handles['histogram'].append(hist_handle)
        else:
            return event_plot_handles, hist_handle
