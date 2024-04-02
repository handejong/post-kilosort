#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A signal browser for raw and filtered (Kilosort) data.

Created: Tue Oct 3 16:35 2023

@author: Han de Jong
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class signal_explorer:

    def __init__(self, pks_dataset, stamps = []):

        # Store the datset itself
        self.data = pks_dataset

        # Store formatting parameters that may or may not be used
        self.palette = "tab10"  # hls is unclear, but looks good.
        self.colormap = self.data.plot._color_map(10)
        self.facecolor = 'k'
        self.axcolor='w'

        # Start time
        self.T = 100
        self.window = 1

        # Which signals to show
        self.channel_start = 0
        self.channel_end = 10

        # Should we plot templates?
        self.show_templates = False

        # Scaling of the filter signal
        self.clean_mean = 0
        self.clean_std = 0

        # Make memmaps to the data
        self.clean_signal = self._get_raw_signal(self.data.signal_path)
        self.raw_signal = self._get_raw_signal(self.data.raw_signal_path)

        # Get metadata
        self.sample_rate = float(self.data.metadata['imSampRate'])
        self.timeline_clean = np.linspace(0, len(self.clean_signal) / self.sample_rate, len(self.clean_signal))
        self.timeline_raw = np.linspace(0, len(self.raw_signal) / self.sample_rate, len(self.raw_signal))

        # Store the stamps
        self.stamp_counter = 0
        self.stamps = stamps
        if len(stamps)>0:
            self.T = self.stamps[0]

        # Figure handles
        self.template_handles = []

        # CORRECT FOR SYNC SETTINGS
        # TODO

    def get_plot_data(self, raw = True):
        """
        This is the main function that will give you the data used for plotting.
        """

        # Raw or clear signal
        if raw:
            timeline = self.timeline_raw
            signal = self.raw_signal
        else:
            timeline = self.timeline_clean
            signal = self.clean_signal

        # Make the indexer
        indexer = (timeline>self.T-self.window) & (timeline<self.T+self.window)

        # Select the data
        to_plot = pd.DataFrame(signal[indexer, self.channel_start:self.channel_end+1], index=timeline[indexer])

        # Scale and offset
        mean = np.mean(to_plot.values[:])
        std = np.std(to_plot.values[:])
        to_plot = self.scale_and_offset(to_plot, mean, std)

        # Store mean and std if this is clean signal
        if not raw:
            self.clean_mean = mean
            self.clean_std = std

        return to_plot

    def get_template(self, unit, time = None):
        """

        """

        if time is None:
            time = self.T

        # Get the template
        template = self.data.get_template(unit).loc[self.channel_start:self.channel_end, :].transpose()

        # Set the timeline
        index =  np.linspace(0, template.shape[0] / self.sample_rate, template.shape[0])
        center = index[int(len(index)/2)]
        index = index + time - center
        template.index = index

        return template
    
    def make_figure(self):
        """
        """
        fig, ax = plt.subplots(1, figsize = (11, 7), tight_layout=True, facecolor=self.facecolor)
        self.fig = fig
        self.ax = ax

        # Plot
        self.raw = ax.plot(self.get_plot_data(raw=True), color=self.axcolor, linewidth=0.5, alpha = 0.6)
        self.clean = ax.plot(self.get_plot_data(raw=False), color='green', linewidth=0.8)

        # Add the T-line
        ylim = self.ax.get_ylim()
        self.t_line = ax.vlines(self.T, ymin=ylim[0], ymax=ylim[1], color='r',
            linestyle='--', linewidth=1, alpha = 0.5)

        # Some formatting
        ax.set_xlabel('Time (s)')
        self._default_formatting(ax)

        # Add the callback
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def update_figure(self):
        """
        """

        # Check if figure exists

        # Update raw
        new_data = self.get_plot_data(raw = True)
        for i, col in new_data.items():
            self.raw[i].set_data(col.index.values, col.values)

        # Update clean
        new_data = self.get_plot_data(raw = False)
        for i, col in new_data.items():
            self.clean[i].set_data(col.index.values, col.values)

        # Update figure
        self.ax.set_xlim((self.T-self.window, self.T+self.window))

        # Update the line indicating T
        ylim = self.ax.get_ylim()
        segments = [[[self.T, ylim[0]], [self.T, ylim[1]]]]
        self.t_line.set_segments(segments)

        # Update the templates
        self.delete_templates()
        if self.show_templates:
            self.plot_templates()

        # Update
        plt.draw()

    def templates(self):
        """
        Toggle if we want to see templates
        """

        if self.show_templates:
            self.show_templates = False
        else:
            self.show_templates = True

        self.update_figure()

    def plot_templates(self, units = None):
        """

        """
        # Plot all units in this window if units is None
        if units is None:
            indexer = (self.data.clusters.mainChannel>=self.channel_start) & (self.data.clusters.mainChannel<=self.channel_end)
            units = self.data.clusters[indexer].index.values

        # For every unit in units
        counter = 0
        colors = ['pink', 'orange', 'blue', 'red', 'purple', 'pink', 'orange']
        for unit in units:

            # See if this unit has it's main channel here
            mainChannel = self.data.clusters.loc[unit, 'mainChannel']
            if not (mainChannel >= self.channel_start) & (mainChannel <= self.channel_end):
                continue

            # See if there are any spikes in the window
            spikes = self.data.get_unit_spikes(unit, return_real_time=True)
            spikes = spikes[(spikes>self.T-self.window) & (spikes<self.T+self.window)]
            if len(spikes)<1:
                continue

            # For every spike
            for spike in spikes:
                template = self.get_template(unit, time=spike)
                template = self.scale_and_offset(template, mean = np.mean(template.values[:]),
                        std = (1/0.68) * np.std(template.values[:]))
                template = self.ax.plot(template, color=colors[counter], linewidth = 0.7)
                self.template_handles.append(template)
            counter += 1

    def delete_templates(self):
        """

        """
        while len(self.template_handles)>0:
            temp = self.template_handles.pop()
            for i in temp:
                i.remove()

    def next(self, i=1):
        """
        """
        if (len(self.stamps)>self.stamp_counter+i) & (self.stamp_counter + i > 0):
            self.stamp_counter += i
            self.T = self.stamps[self.stamp_counter]
            self.ax.set_title(f'Spike nr: {self.stamp_counter}', color=self.axcolor)
            self.update_figure()


    def scale_and_offset(self, to_plot, mean = None, std = None):
        """
        Will scale(Z-score) all the signals and then offset them by 10 std.
        """

        if mean is None:
            mean = self.clean_mean
        if std is None:
            std = self.clean_std

        to_plot = (to_plot - mean)/std
        offseter = np.linspace(0, to_plot.shape[1]*10, to_plot.shape[1])
        return to_plot + offseter


    def _get_raw_signal(self, path):
        """
        Makes a memory pap linking to the signal. Currently supported signals. 
        The path to the signal of interest is stored in self.signal_path.
        Currently supported signals are:

            - temp_wh.dat (output of Kilosort, whitened and hp filtered)
            - continous.dat (output of open Ephys)
            - some file ending in .bin (output of SpikeGLX)

        Returns
        -------
        Memmap to the signal of interest

        """

        # NOTE, self.params['dtype'] has the Kilosort output data type. In the
        # future this might not work for Open Ephys and/or SpikeGLX data.

        # Make the memmap
        data = np.memmap(path, dtype=self.data.params['dtype'])

        # Whitened Kilosort output
        if path.endswith('temp_wh.dat'):
            #print('Loading Kilosort output signal.')
            data = data.reshape([-1, 383])
            return data

        # Raw Open Ephys output
        if path.endswith('continous.dat'):
            #print('Loading Open Ephys output signal.')
            data = data.reshape([-1, 384])

            # Todo Remove channel 190!

            return data

        # Spike GLX output
        if path.endswith('ap.bin'):

            data = data.reshape([-1, 385])
            # Todo Remove channel 190.
            # Shall we leave channel 384 (sync pulses)?

            return data

        # RAISE ERROR

    def _on_key(self, event):

        if event.key == 'n':
            self.next(1)
            return
        if event.key == 'p':
            self.next(-1)
            return
        if event.key == 'r':
            self.stamp_counter = round(random.random()*len(self.stamps))-1
            self.next(1)
            return
        if event.key == 't':
            self.templates()
            return
        
        print(f'No action specified for key: {event.key}')

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

        # Y Axis
        start = self.channel_start; end = self.channel_end; n = 1 + end - start
        labels = np.linspace(start, end, n).astype(int)
        label_locations = np.linspace(0, n*10, n)
        ax.set_yticks(label_locations)
        ax.set_yticklabels(labels)
