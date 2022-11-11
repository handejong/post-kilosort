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

Run them by calling:
    
    >>> function(self, list_of_units, list_of_channels)
    
for instance, this will plot the waveforms of unit 3 and 4 on electrode 0-5:
        
    >>> dataset.plot.waveform([3, 4], [0, 1, 2, 3, 4, 5]) 
    
Finally there are functions to look at firing distribution:
    
    - correlogram
    - ISI
    
Run them by calling:
    
    >> function(self, list_of_units)

Created: Fri Nov  4 12:21:50 2022
Last Updated: 

@author: Han de Jong
"""

# Dependend libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pks_attribute_functions as atf
from pks_inspector import inspect_object


class pks_plotting:
    
    def __init__(self, pks_dataset):
        
        # Store the datset itself
        self.data = pks_dataset
        
        # Store parameters that may or may not be used
        self.plot_max = 250
        self.palette = "tab10" #hls is unclear, but looks good.
        self.colormap = self._color_map(10)
    
    
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
            pca.append(self.data.channel_pca(channel, None))
         
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


    def custom_plot(self, units, channels, type='time_plot', attribute_function=None):

        if type == 'time_plot':
            try:
                return time_plot(self, units, channels, attribute_function)
            except:
                raise Exception('Not a valid attribute function see help(custom_plot) for help.')

        if type =='attribute_plot':
            try:
                return attribute_plot(self, units, channels, attribute_function)
            except:
                raise Exception('Not a valid attribute function see help(custom_plot) for help.')

        raise Exception('Not a valid plot type, see help(custom_plot) for help.')

    
    def inspect(self, units, channels):
        
       
        # Correlograms
        #self.correlogram(units)
        
        # Print overview
        #print(self.data.clusters.loc[units, :])
        
        return inspect_object(self, units, channels)
        
        
    def ISI(self, unit):
        """
        Plots the distribtion of the inter-spike-intervals

        Parameters
        ----------
        unit : int or str
            DESCRIPTION.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        
        spikes = self.data.get_unit_spikes(unit)
        isi = 1000*np.diff(spikes)/self.data.params['sample_rate']
        
        fig, ax = plt.subplots(1)
        ax.hist(isi, bins=np.linspace(0, 25, 25), density=True, range=(0, 25))
        ax.set_xlabel('Inter-spike-interval (ms)')
        ax.set_ylabel('Density')
        
        return ax
    
    
    def correlogram(self, units=None):

        # Infer units if necesary
        units, _ = self.data._infer_unit_channels(units, [0, 1,])
        
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
                if X==Y:
                    color = colors[X]
                
                # Calculate and plot
                interval = self._correlogram(unit_Y, unit_X)
                axs[Y, X].hist(interval, bins=np.linspace(0, 25, 25), 
                        density=True, range=(0, 25), color=color)
                
                # Formatting
                if Y==len(units)-1:
                    axs[Y, X].set_xlabel('Time (ms)')
                if X == 0:
                    axs[Y, X].set_ylabel(f'Unit: {unit_Y} (density)')
                if Y==0:
                    axs[Y, X].set_title(f'Unit: {unit_X}')
                    
        # Finaly formatting
        plt.suptitle('Correlogram')

                
    def add_unit(self, unit):

        # Add the unit to each plot
        _ = [i.add_unit(unit) for i in self.data.linked_plots]


    def remove_unit(self, unit):

        # Remove the unit from each plot
        _ = [i.remove_unit(unit) for i in self.data.linked_plots]


    def change_channel(self, index, new_channel):

        # Change the channel in each plot
        _ = [i.change_channel(index, new_channel) for i in self.data.linked_plots]


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


    def _color_map(self, n:int):
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
        unit_1 = unit_1[unit_1<unit_2[-1]]
        
        interval = np.array([unit_2[unit_2>i][0]-i for i in unit_1])
        interval = 1000*interval/self.data.params['sample_rate']
        
        return interval



class plot_object:
    """
    Main plot_object class, all other plot objects are derived from this class.
    """
    
    def __init__(self, plot_object, units, channels, attribute_function=None,
                 other_data = None):
        
        # Store data and handles
        self.plotter = plot_object
        self.data = plot_object.data
        self.plot_handles = {}
        self.units = []
        self.colors = []
        self.other_data = other_data
        self.update_diagonal = True
        
        # Figure out what attribute function to use
        if not attribute_function is None:
            self.attribute_function = attribute_function

        # Infer the units and the channels
        units, channels = self.data._infer_unit_channels(units, channels)
        self.channels = channels #(Units are added below)
            
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

 
    def add_unit(self, unit):
        """
        Responsible for adding a unit to every axes in the plot.

        """

        # Check if unit is not allready in there
        if unit in self.units:
            return None
        
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
            
        # Add (update) the legend
        self._add_legend()
   
  
    def remove_unit(self, unit):
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
        

    def change_channel(self, index, new_channel):

        # Update the channel list
        self.channels[index] = new_channel

        # Replot all units on the axis with index=index
        for i, unit in enumerate(self.units): # for all units

            # Remove unit from axis
            self._remove_unit_from_axes(i, index)

            # Add the new unit
            data = self._get_plot_data(unit, new_channel)
            self._add_unit_to_axes(data, index, self.colors[i], index=i)

        # Set the axes labels
        self._label_plot(index)

    
    def show_only(self, units):
        """
        Should remove all units except the units in 'units'. However: it
        is currently BROKEN.

        Parameters
        ----------
        units : list or NumPy Array
            List of units that should stay visible.

        Returns
        -------
        None.

        """
        
        if not units.__class__ == list:
            units = [units]
        
        for unit in self.units:
            if unit not in units:
                self.remove_unit(unit)

                
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

    def _add_unit_to_axes(self, data_X, data_Y, axes_X, axes_Y, color, index = None):
        # If the axes are one-dimensional, only use data_X and data_Y

        pass

    def _remove_unit_from_axes(self, index, axes_X, axes_Y=None):
        """
        

        """

        # We want this function to work on both 2-D and 1-D axes
        if len(self.axs.shape)==2:

            # Sometimes we don't want the diagonal to be updated
            if (not self.update_diagonal) & (axes_X==axes_Y):
                return None

            temp = self.plot_handles[axes_X][axes_Y][index]
            temp.remove()
            self.plot_handles[axes_X][axes_Y][index] = []
        else:
            temp = self.plot_handles[axes_X][index]
            if temp.__class__==list:
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
        
        colors = self.colors
        elements = [Patch(facecolor=colors[i], edgecolor=colors[i],
                          label=f'Unit: {self.units[i]}') 
                    for i in range(len(self.units))]
        self.axs.ravel()[-1].legend(handles=elements)


    def _on_close(self, event):
        """
        Runs after fiture close
        """
        self.is_closed=True
        self.data._check_linked_plots()
    

    def _key_press(self, event):
        """
        Runs after a keypress on the figure

        """
        
        self.event = event
        
        

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
                                ncols=n_channels, tight_layout=True)
 
        # Formatting
        for i, ax in enumerate(axs[0, :]):
            ax.set_title(f'Channel: {self.channels[i]}')
        for i, ax in enumerate(axs[:, 0]):
            ax.set_ylabel(f'Channel: {self.channels[i]}')    
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
            
        # Add keypresses to the axis?
        
        # Store the axes
        self.axs = axs
        
        return fig

        
    def add_unit(self, unit):
        
        # Check if unit is not allready in there
        if unit in self.units:
            return None
        
        # Figure out color
        color = self._get_color()
        self.colors.append(color)

        # Make data_X
        spikes = self.data.get_unit_spikes(unit)
        data = [self._get_plot_data(spikes, channel) for channel in self.channels]
        
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
                if (not self.update_diagonal) & (X==Y):
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
        for i, unit in enumerate(self.units): # for all units

            # Unfortunately have to get all the channels
            data = [self._get_plot_data(unit, channel) for channel in self.channels]

            # Replace horizontally
            Y = index
            for X, ax in enumerate(self.axs[Y, :]):
                self._remove_unit_from_axes(i, X, Y) # remove the unit from the axis
                self._add_unit_to_axes(data[X], data[Y], X, Y, self.colors[i], index=i)

            # Replace Vertically
            X = index
            for Y, ax in enumerate(self.axs[: X]):
                self._remove_unit_from_axes(i, X, Y) # remove the unit from the axis
                self._add_unit_to_axes(data[X], data[Y], X, Y, self.colors[i], index=i)

        # Set the axes labels
        self.axs[0, index].set_title(f'Channel: {self.channels[index]}')
        self.axs[index, 0].set_ylabel(f'Channel: {self.channels[index]}')   
   

    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__==int) | (spikes_or_unit.__class__==str) | (spikes_or_unit.__class__==np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit    

        # Make the data
        temp = self.data.get_waveform(spikes, channel, average=False, 
            sample=self.plotter.plot_max)

        return self.attribute_function(temp, channel)


    def _add_unit_to_axes(self, data_X, data_Y, axes_X, axes_Y, color, index = None):

        # Sometimes we don't want the diagonal to be updated
        if (not self.update_diagonal) & (axes_X==axes_Y):
            return None

        # Do the plotting
        handle = self.axs[axes_Y, axes_X].scatter(x=data_X, y=data_Y, marker='.', 
                                      color=color, s=10, alpha=0.8)

        if index is None:
            self.plot_handles[axes_X][axes_Y].append(handle)
        else:
            self.plot_handles[axes_X][axes_Y][index] = handle         


class pca_plot(attribute_plot):
    
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
            self.plot_handles[i, i] = self.axs[i, i].plot(self.other_data[i].components_.transpose())
            self.axs[i, i].set_xticks([]); self.axs[i, i].set_yticks([])
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
        for i, unit in enumerate(self.units): # for all units

            # Unfortunately have to get all the channels
            data = [self._get_plot_data(unit, channel) for channel in self.channels]

            # Replace horizontally
            Y = index
            for X, ax in enumerate(self.axs[Y, :]):
                self._remove_unit_from_axes(i, X, Y) # remove the unit from the axis
                self._add_unit_to_axes(data[X], data[Y], X, Y, self.colors[i], index=i)

            # Replace Vertically
            X = index
            for Y, ax in enumerate(self.axs[: X]):
                self._remove_unit_from_axes(i, X, Y) # remove the unit from the axis
                self._add_unit_to_axes(data[X], data[Y], X, Y, self.colors[i], index=i)

        # Set the axes labels
        self.axs[0, index].set_title(f'Channel: {self.channels[index]}')
        self.axs[index, 0].set_ylabel(f'Channel: {self.channels[index]}')

        # Update the plot
        self.plot_handles[index, index][0].remove()
        self.plot_handles[index, index] = self.axs[index, index].plot(self.other_data[i].components_.transpose())

    
    def attribute_function(self, data, channel):

        index = self.channels.index(channel)
        
        return self.other_data[index].transform(data)
       
        
class waveform_plot(plot_object):
        
    def make_figure(self):
        
        # Make a figure for the waveforms
        cols = 5
        while not ((len(self.channels)%cols == 0) or (cols==1)):
            cols-= 1
        rows = int(len(self.channels)/cols)
        fig, axs = plt.subplots(figsize=[4*cols, 4*rows], nrows=rows, 
                                ncols=cols,tight_layout=True, sharex=True)
        axs = axs.ravel()
        
        # Store the axes handles    
        self.axs = axs

        # Add titles to the axe
        for i, chan in enumerate(self.channels):
            self._label_plot(i)   
        
        return fig


    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__==int) | (spikes_or_unit.__class__==str) | (spikes_or_unit.__class__==np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit    

        return self.data.get_waveform(spikes, channel, average=False,sample=self.plotter.plot_max)


    def _add_unit_to_axes(self, data_X, axes_X, color, index = None):
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


class time_plot(plot_object):
        
    def make_figure(self):
        
        # Make the figure
        n_channels = len(self.channels)
        fig, axs = plt.subplots(figsize=[10, 12], nrows=n_channels, 
                                ncols=1, tight_layout = True, sharex=True)
        axs = axs.ravel()
        axs[-1].set_xlabel('Time (s)')
        self.axs = axs

        # Formatting
        for i, ax in enumerate(axs):
            self._label_plot(i)
            
        plt.suptitle('Amplitude in time')
        
        return fig

    
    def _get_plot_data(self, spikes_or_unit, channel):

        # Grab spikes
        if (spikes_or_unit.__class__==int) | (spikes_or_unit.__class__==str) | (spikes_or_unit.__class__==np.int64):
            spikes = self.data.get_unit_spikes(spikes_or_unit)
        else:
            spikes = spikes_or_unit

        # Get the data
        data, X = self.data.get_waveform(spikes, channel,
             average=False, sample=self.plotter.plot_max, 
             return_spikes=True)    
        X = X/self.data.params['sample_rate']
        data = self.attribute_function(data)

        return data, X


    def _add_unit_to_axes(self, data_X, axes_X, color, index = None):
        # If the axes are one-dimensional, only use data_X and data_Y

        # Data_X is a tuple in this case
        data = data_X[0]; X = data_X[1]

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
 
