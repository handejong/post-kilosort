#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKS_prosessing contains the main functions for loading, processing and saving
data. The main class is the pks_dataset class, which contains most of the
important data. This will load a Kilosort output dataset:
    
    >>> path = /home/usr/data/kilsort_output_folder (example)
    >>> dataset = pks_dataset(path)

A dataframe with basic info about the Kilosort output is included:
    
    >>> dataset.clusters
    
For every spike in the dataset there is a timestamp and an ID. The ID is the
unit ID it was assigned to by kilosort.

    >>> dataset.spikeTimes
    >>> dataset.spikeID
    
There is also info on the channel positions allong the probe and some basic
recording parameters.

    >>> dataset.channel_positions
    >>> dataset.params

Finally, there is a DataFrame that includes the manimulations you have made on
the dataset.

    >>> dataset.changeSet
    
The CHANGE_SET is a major feature of PKS. It contains all manipulations you
have made and makes it possible for others to replicate your analysis. All
manipulations are stored in a file: 'changeSet.py', which you can edit with
any text editor.

IMPORTANT the first time you load a new dataset PKS will make a folder in the
Kilosort output folder where it will store it's data. Before you can start
plotting data, you will also have to build the waveforms:
    
    >>> dataset.build_waveforms() (see options and parameters below)
    
This function can take quite a long time. The major advantage however is that
you will now have a self-containing dataset on disk, which might be more
managable than the raw datset.

Created: Fri Nov  4 12:34:37 2022
Last Updated: Nov 11 14:01:20 2022

@author: Han de Jong
"""

# Imports
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
import sys
import os
from multiprocessing import Pool, cpu_count
from pks_plotting import pks_plotting
from pks_spike_sorting import sorter
from sklearn.decomposition import PCA
import time


class pks_dataset:
    """
    The main PKS dataclass
    """

    def __init__(self, path: str):

        # Error handeling

        # Store the path and other variables
        self.path = self._check_path(path)
        self._check_folder()

        # Load the data
        self._load_data()

        # Make or check folder structure
        self._check_folder() # TODO twice?!

        # Adding external methods
        self.plot = pks_plotting(self)
        self.sort = sorter(self)

        # Where we will put handles to linked plots
        self.linked_plots = []

        # Where we will store units that are finished
        self.done = []

        # Run the change set
        # NOTE: it is just a Python scripts. You can put anything you want in
        # there!
        self.save_data = False
        exec(open(self.path + 'pks_data/changeSet.py').read())
        self.save_data = True

    def build_waveforms(self):
        """

        Grabs the raw signal (defined in self.signal_path) and extracts all the
        waveforms. Extracted waveforms are saved as sparce matrixes, resulting
        in a significantly more managable dataset.

        Returns
        -------
        None.

        """

        # Start a pool and do all the work.
        with Pool() as pool:
            print(f'Extracting raw waveforms on {cpu_count()} cores')
            res = pool.map(self._build_waveform,
                           range(self.params['n_channels']))

    def get_waveform(self, spikes: np.array, channel: int, average: bool = True,
                     sample: int = None, return_spikes=False, window=1.38):
        """

        Will grab the waveforms at time 'spikes' and return either the
        average waveform or a large 2D numpy array with all waveforms.

        Parameters
        ----------
        spikes : np.array
            List of the spike times (data points numbers) that you are
            interested in.
        channel : int
            channel number
        average : bool, optional
            Wether you want the average waveform or all waveforms. 
            The default is True.
        sample : None or int
            Returns a max of 'n' waveforms. These waveforms are homogenously
            sampled throughout the entire recording.
        return_spikes: bool
            If true, will return the sampled spikes as well (making it possible
            to plot waveforms on a timeline.)
        window: float
            Window (on both sides of the spike). Will be aproximated in data-
            points (int).

        Returns
        -------
        waveform : np.array
            a 1-D array (average waveform) or a 2-D array (all waveforms)
        spikes : np.array
            a 1-D array with the spikesTimes of the waveforms

        """

        # Sanitize inputs
        spikes = spikes.astype(int)

        # Are we sampling?
        if sample is not None and len(spikes) > sample:
            spikes = spikes[::len(spikes) // sample]
            if len(spikes)>sample:
                spikes = spikes[:sample]

        # Grab the signal we want to work on
        data = self._get_raw_signal()

        # What is the window size?
        samp_rate = float(self.metadata['imSampRate'])/1000
        window = int(samp_rate*window)

        # Output data
        try:
            output = data[spikes[:, np.newaxis] + np.arange(-window, window), channel]
        except:
            print("Unable to grab waveforms")

        # Make the output
        if average:
            waveform = np.mean(output, axis=0)
        else:
            waveform = output

        if return_spikes:
            return waveform, spikes
        else:
            return waveform

    def get_template(self, unit: int, all_channels: bool = False):
        """
        Grabs the 2D template of the unit in 'unit'.

        NOTE: currently only possible for Kilosort units. We don't make new
        templates for pks-clustered units.

        Parameters
        ----------
        unit : int
            index of the target unit
        all_channels : bool
            Wether or not the user wants all channels or just the channels on
            which this unit is.

        Returns
        -------
        Pandas DataFrame with the unit template.

        """

        # Load the template
        all_templates = np.load(self.path+'templates.npy')
        template = pd.DataFrame(all_templates[unit, :, :].transpose())

        # Unless the caller wants all channels, ommit the channels without template
        if not all_channels:
            template = template.iloc[(template.sum(axis=1) != 0).values, :]

        return template

    def get_unit_spikes(self, unit):
        """
        Returns 

        Parameters
        ----------
        unit : int or str
            Identifyer of the unit of interest. Should match up with a value in
            self.spikeID.

        Returns
        -------
        Numpy array with a list of spikes (data point numbers) of this unit.

        """

        return self.spikeTimes[self.spikeID == unit]

    def channel_pca(self, channel: int, units=[], n_components: int = 1):
        """
        Does a principal component (PCA) analysis on one channel. The waveforms 
        that are used to do the PCA over are either the waveforms indicated
        by the spikes of the units in 'units', or defined defined by the
        spikes of all units that are within 2 channels above or below this
        channel.

        Note: not all data is loaded to do the PCA over, instead a sample of
        1000 waveforms is used.

        Parameters
        ----------
        channel : int
            The channel that we want to do the PCA on
        units : TYPE, optional
            A list of units, the spikeTimes of which will be used to select
            waveforms.
        n_components : int, optional
            The number of components

        Returns
        -------
        pca : sklearn.PCA
            A fitted PCA object.

        """

        # Essentially we only do PCA on waveforms that occured during an spike on
        # a neighboring channel. But if we can't find spikes, we keep increasing
        # the range of the channels we consider to be neighboring.
        spikes = []
        channel_range = 2
        while len(units)==0:
            channel_range += 1
            indexer = (self.clusters.mainChannel > channel-channel_range) &\
                (self.clusters.mainChannel < channel+channel_range)
            units = self.clusters[indexer].index
            print(f'PCA on channel {channel} with range {channel_range}')
    
        # Get the spikes for all those units
        for unit in units:
            spikes.extend(self.get_unit_spikes(unit))
        spikes = np.array(spikes)

        # Get the waveforms
        waveforms = self.get_waveform(spikes, channel, sample=1000,
                                      average=False)

        # Do the pca
        pca = PCA(n_components=n_components)
        pca.fit(waveforms)

        return pca

    def get_nidq(self):
        """
        Will load NI DAQ data if available.

        Currently will assume TTL pulses recorded at analog inputs. Pulses are
        extracted through thresholding, exact values are ignored.

        Returns:
        --------
        Pandas dataFrame were every pulse is a row.

        """
        # First check if NI data was recorded
        if self.nidg_path is None:
            print('No NI DAQ data recorded.')
            return None

        # Check if timestamps already exist
        save_path = self.path + 'nidq.csv'
        try:
            output = pd.read_csv(save_path).set_index('#')
            return output
        except FileNotFoundError:
            print('Deriving stamps from raw signal.')

        # First grab the meta-data
        nidq_meta = self._load_meta_file(self.nidg_path[:-3] + 'meta')

        # Grab data
        data = np.memmap(self.nidg_path, dtype='int16')
        data = data.reshape([-1, int(nidq_meta['nSavedChans'])])

        # Grab every channel
        output = pd.DataFrame()
        timeline = np.linspace(0, len(data) / (float(nidq_meta['niSampRate'])/1000), len(data))
        for channel in range(data.shape[1]):

            try:
                selection = (data[:, channel]>10000).astype(int)

                # Grab the pulses
                diff = np.diff(selection)
                starts = timeline[:-1][diff == 1]
                stops = timeline[:-1][diff == -1]

                # Filter starts and stops
                starts = starts[starts < stops[-1]]
                stops = stops[stops > starts[0]]

                duration = stops - starts

                # Create output DataFrame
                temp = pd.DataFrame()
                temp['Start'] = starts
                temp['Stop'] = stops
                temp['Duration'] = duration
                temp['Channel'] = f'AI_{channel}'
                temp['ITI'] = np.insert(np.diff(starts), 0, np.nan)

                # Concatenate
                output = pd.concat((output, temp), axis=0)
            except:
                print(f'No pulses on AI_{channel}')
        
        # Some formatting
        output.index = range(len(output)) 
        output.index.name = '#'
        output.loc[:, 'Duration'] = output.Duration.round(1)
        output.loc[:, 'ITI'] = output.ITI.round(1)
        output.loc[:, 'Start'] = output.Start.round(1)
        output.loc[:, 'Stop'] = output.Stop.round(1)

        # Save the file
        output.to_csv(save_path)

        return output

    def get_sync_pulses(self):
        """
        Will load the sync pulses

        Imec cards have one TTL input that can be used for sync pulses.

        What I hate about this function is that there is a lot of overlap with
        get_nidq.

        Returns:
        --------
        Pandas dataFrame were every pulse is a row.

        """
        # First check if NI daexitta was recorded

        # Check if timestamps already exist
        save_path = self.path + 'sync.csv'
        try:
            output = pd.read_csv(save_path).set_index('#')
            return output
        except FileNotFoundError:
            print('Deriving stamps from raw signal.')

        # First grab the meta-data and the path
        sync_meta = self._load_meta_file(self._find_files_with_extension(self.path,
            'lf.meta'))
        sync_path = self._find_files_with_extension(self.path, 'lf.bin')

        # Grab data
        data = np.memmap(sync_path, dtype='int16')
        data = data.reshape([-1, int(sync_meta['nSavedChans'])])

        # Grab only channel 384
        timeline = np.linspace(0, len(data) / (float(sync_meta['imSampRate'])/1000), len(data))
        selection = (data[:, 384]>10).astype(int)

        # Grab the pulses
        diff = np.diff(selection)
        starts = timeline[:-1][diff == 1]
        stops = timeline[:-1][diff == -1]

        # Filter starts and stops
        starts = starts[starts < stops[-1]]
        stops = stops[stops > starts[0]]

        duration = stops - starts

        # Create output DataFrame
        output = pd.DataFrame(index = range(len(starts)))
        output['Start'] = starts
        output['Stop'] = stops
        output['Duration'] = duration
        output['ITI'] = np.insert(np.diff(starts), 0, np.nan)

        # Some formatting
        output.index.name = '#'
        output.loc[:, 'Duration'] = output.Duration.round(1)
        output.loc[:, 'ITI'] = output.ITI.round(1)
        output.loc[:, 'Start'] = output.Start.round(1)
        output.loc[:, 'Stop'] = output.Stop.round(1)

        # Save the file
        output.to_csv(save_path)

        return output


    def _infer_unit_channels(self, units, channels):
        """
        In general, we want the units and the channels to be lists of length>0. It is
        possible for the untis to be str instead of int, but the channels should
        always be int. In addition, if the user does not specify units or channels
        (i.e. they are "None") then they should be infered from a plot that is
        currently open.

        Returns
        -------
        Sanitized unit and channel lists

        """

        if units is None:
            units = self.linked_plots[0].units[:]

        if channels is None:
            channels = self.linked_plots[0].channels[:]

        units = self._check_if_list(units)
        channels = self._check_if_list(channels)

        return units, channels

    def _check_if_list(self, input_value):
        """
        For a lot of things, we only want to work on lists, not ints or NumPy arrays. This
        method is responsible for converting everything else into a list.

        Raises
        ------
        Exception if the input type is not supported.

        Returns
        -------
        The input as a list of length >0.

        """

        if input_value.__class__ == list:
            return input_value

        if input_value.__class__ == int:
            return [input_value]

        if input_value.__class__ == np.ndarray:
            return [i for i in input_value]

        raise Exception(f'{input_value.__class__} is not a valid input type.')

    def _get_raw_signal(self):
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
        data = np.memmap(self.signal_path,
                         dtype=self.params['dtype'])

        # Whitened Kilosort output
        if self.signal_path.endswith('temp_wh.dat'):
            #print('Loading Kilosort output signal.')
            data = data.reshape([-1, 383])
            return data

        # Raw Open Ephys output
        if self.signal_path.endswith('continous.dat'):
            #print('Loading Open Ephys output signal.')
            data = data.reshape([-1, 384])

            # Todo Remove channel 190!

            return data

        # Spike GLX output
        if self.signal_path.endswith('ap.bin'):

            data = data.reshape([-1, 385])
            # Todo Remove channel 190.
            # Shall we leave channel 384 (sync pulses)?

            return

        # RAISE ERROR

    def _build_waveform(self, channel: int, chan_range=[-10, 10]):
        """

        Extracts waveforms from the raw signal.

        Parameters
        ----------
        channel : int
            The channel that we are working on.
        chan_range : list of two ints, optional
            If there is a spike on any of the channels in this range
            (relative to electrode), this waveform will be include. The default 
            is [-10, 10].

        Returns
        -------
        None.

        """

        # Update the user
        print(f'Working on channel: {channel}.')

        # Grab the spike times on channels close by
        _, spike_channel_list = self._build_channel_list()
        indexer = (spike_channel_list >= channel + chan_range[0]) & \
            (spike_channel_list <= channel + chan_range[1])
        spike_times_selection = self.spikeTimes[indexer]

        # Grab from the raw data
        waveforms = self.get_waveform(spike_times_selection,
                                      channel,
                                      average=False)

        # Output
        output = np.zeros((len(self.spikeTimes), waveforms.shape[1]))
        output[indexer, :] = waveforms

        # Make sparce and save
        output = csr_matrix(output.transpose())
        save_npz(self.path + f'pks_data/raw/channel_{channel}.npz', output)

    def _load_data(self):
        """
        Responsible for loading the data

        Returns
        -------
        None.

        """
        # load all tsv files first
        files = ['cluster_Amplitude.tsv', 'cluster_ContamPct.tsv',
                 'cluster_group.tsv']
        clusters = [self._load_tsv(self.path + file) for file in files]
        self.clusters = pd.concat(clusters, axis=1)

        # Load all spike times
        self.spikeTimes = np.load(
            self.path + 'spike_times.npy').ravel().astype(int)
        self.spikeID = np.load(
            self.path + 'spike_clusters.npy').ravel().astype(int)

        # Add the Spike count to the data
        for i in self.clusters.index:
            self.clusters.loc[i, 'spikeCount'] = (self.spikeID == i).sum()

        # Delete units with 0 spikes
        self.clusters = self.clusters[self.clusters.spikeCount > 0]

        # Add the channel list
        temp, _ = self._build_channel_list()
        self.clusters.loc[:, 'mainChannel'] = temp

        # Add channel_map
        temp = np.load(self.path + 'channel_positions.npy')
        self.channel_positions = pd.DataFrame(temp, columns=['X', 'Y'])

        # Recording and Kilosort paramters
        sys.path.append(self.path)
        from params import dat_path, n_channels_dat, dtype, offset, sample_rate, hp_filtered
        self.params = {}
        self.params['dat_path'] = dat_path
        self.params['n_channels'] = n_channels_dat
        self.params['dtype'] = dtype
        self.params[offset] = offset
        self.params['sample_rate'] = sample_rate
        self.params['hp_filtered'] = hp_filtered

        # Store metadata
        file_path = self.params['dat_path'].split('/')[-1].split('.')[0] + '.ap.meta'
        self.metadata = self._load_meta_file(self.ap_meta_path)

        # Finally, load the similarity matrix
        self.similarity_matrix = pd.DataFrame(data = np.load(self.path + 'similar_templates.npy'))

    def _check_folder(self):
        """

        Checks if this dataset has ever been run trough pks before and if not
        makes the data folder.

        """
        # Check if folder exists
        if not os.path.isdir(self.path + 'pks_data'):
            print('Creating PKS output folder.')
            os.mkdir(self.path + 'pks_data')

        # Check if the change_set file exists
        if not os.path.isfile(self.path + 'pks_data/changeSet.py'):
            print('Creating the changeSet file.')
            with open(self.path + 'pks_data/changeSet.py', 'a') as f:
                f.write('# This file will execute in PKS. It contains all')
                f.write('modifications of the Kilosort dataset\n')

        # Check if raw folder exists
        if not os.path.isdir(self.path + 'pks_data/raw'):
            os.mkdir(self.path + 'pks_data/raw')

        # Check if we allready have one or more extracted channels
        if not os.path.isfile(self.path + 'pks_data/raw/channel_0.npz'):
            print('To build the waveforms run:')
            print('>>> self.build_waveforms()')

    def _build_channel_list(self):
        """
        Build the list of the channels on which any particular unit (as
        identified by Kilosort) has the largest amplitude. It should only be
        necesary to run this this method once. Note that the output file
        contains the channel list that is valid for the raw Kilosort output
        no changes are saved. These are instead applied to the data in this
        object using the 'changeSet' file.                                                             

        Returns
        -------
        channel_list:numpy array
            Vector containing the channel on which the unit at index 'i' has
            the larges amplitude.
        """

        # Check if the file allready exists:
        if os.path.isfile(self.path + 'pks_data/unit_channel_list.npy'):
            unit_channel_list = np.load(
                self.path + 'pks_data/unit_channel_list.npy')
            spike_chanel_list = np.load(
                self.path + 'pks_data/spike_channel_list.npy')
            return unit_channel_list, spike_chanel_list

        # If not, we'll have to build it
        print('Building channel list (only necessary on first run).')

        unit_list = self.clusters.index.values
        channel_hash = {unit: self._unit_channel(unit) for unit in unit_list}
        unit_channel_list = np.array([channel_hash[i] for i in unit_list])

        # Save the channel list
        np.save(self.path + 'pks_data/unit_channel_list.npy', unit_channel_list)

        # Also build and save the spike_channel list
        spike_channel_list = np.array([channel_hash[i] for i in self.spikeID])
        np.save(self.path + 'pks_data/spike_channel_list.npy', spike_channel_list)

        return unit_channel_list, spike_channel_list

    def _unit_channel(self, unit):
        """
        Grab the template and figure out on which channel the amplitude is
        largest.
        """

        template = self.get_template(unit).abs()

        return template.max(axis=1).idxmax()

    def _load_tsv(self, file: str):
        """
        Loads tab-sepparated files into a Pandas DataFrame.

        Parameters
        ----------
        file : String
            A String containing a filepath

        Returns
        -------
        data : DataFrame
            DataFrame containing the data in the tsv file.

        """

        data = pd.read_csv(file, sep='\t')
        data.set_index('cluster_id', inplace=True)

        return data

    def _check_linked_plots(self):
        """
        Check if linked plots are still open and pop them if they are not

        """
        to_pop = [plot.is_closed for plot in self.linked_plots]
        for i, pop in enumerate(to_pop):
            if pop:
                self.linked_plots.pop(i)

    def _load_meta_file(self, file_path):
        """
        Will load .meta files that are output from SpikeGLX.

        Returns:
        --------
        Dict with all key-value pairs form the .meta file.
        """

        config = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key] = value
        return config


    def _check_path(self, path: str):
        """
        Figure out the file_paths of the associated data files

        Parameters
        ----------
        path : string
            path to Kilosort dataset

        Returns
        -------
        Nothing if correct, but trows an error if the path is not a real path and
        a different error if not all the files that Kilosort outputs are presents.

        """
        # Formatting
        if not path[-1] == '/':
            path = path + '/'

        # Check if a real path
        # TODO

        # Find the signal paths
        self.signal_path = self._find_files_with_extension(path, 'temp_wh.dat')
        self.raw_sigal_path = self._find_files_with_extension(path, 'ap.bin')
        self.ap_meta_path = self._find_files_with_extension(path, 'ap.meta')

        # If available get NI daq data
        self.nidg_path = self._find_files_with_extension(path + '../', 'nidq.bin')

        # Check if kilosort files available
        #   amplitudes.npy
        #   channel_positions.npy
        #   cluster_Amplitude.npy
        #   cluster_ContamPct.tsv
        #   cluster_group.tsv
        #   cluster_KSLabel.tsv
        #   params.py
        #   similar_templates.npy
        #   spike_clusters.npy
        #   spike_templates.npy

        # Only print warning if we can't find this one
        #   temp_wh.dat

        # Later will also theck for raw data
        #   the ap.bin file (SpikeGLX or a .dat file (open Ephys))

        return path

    def _find_files_with_extension(self, folder_path, extension):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(extension):
                return folder_path + file_name
        return None

