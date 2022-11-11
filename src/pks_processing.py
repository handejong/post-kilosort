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

class pks_dataset:
    """
    The main PKS dataclass
    """
    
    def __init__(self, path:str):
        
        # Error handeling
        
        
        # Store the path and other variables
        self.path = self._check_path(path)
        self._check_folder()
        
        # Load the data
        self._load_data()
        
        # The signal we'll use for analysis
        self.signal_path = self.params['dat_path'].split('/')[-1]
        
        # Make or check folder structure
        self._check_folder()

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


    def get_waveform(self, spikes:np.array, channel:int, average:bool=True,
                     sample:int=None, return_spikes=False):
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
        if not sample is None:
            if len(spikes)>sample:
                i = int(len(spikes)/sample)
                spikes = spikes [0::i]
                
        # Grab the signal we want to work on
        data = self._get_raw_signal()
        
        # Output data
        output = np.zeros([len(spikes), 82])
        
        # Grab the data
        for i, spike in enumerate(spikes):
            try:
                output[i, :] = data[spike-41:spike+41, channel]
            except:
                print(f'Unable to grab waveform {i}')
                
        # Make the output
        if average:
            waveform = np.mean(output, axis=0)
        else:
            waveform = output
        
        if return_spikes:
            return waveform, spikes
        else:
            return waveform
    
        
    def get_template(self, unit:int, all_channels:bool=False):
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
        template = pd.DataFrame(all_templates[unit, :,:].transpose())
        
        # Unless the caller wants all channels, ommit the channels without template
        if not all_channels:
            template = template.iloc[(template.sum(axis=1)!=0).values, :]
            
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
        
        return self.spikeTimes[self.spikeID==unit]
    
    
    def channel_pca(self, channel:int, units=None, n_components:int=1):
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
        
        spikes = []
        if units is None:
            indexer = (self.clusters.mainChannel>channel-3) &\
                (self.clusters.mainChannel<channel+3)
            units = self.clusters[indexer].index
            
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

    
    def _infer_unit_channels(self, units, channels):
        """
        In general, we want the units and the channels to be lists of length>0. It is
        possible for the untis to be str instead of int, but the channels should
        always be ints. In addition, if the user does not specify units or channels
        (i.e. they are "None") they they should be infered from a plot that is
        currently open.

        Returns
        -------
        Sanitized unit and channel lists

        """

        if units is None:
            units = self.linked_plots[0].units

        if channels is None:
            channels = self.linked_plots[0].channels

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
        data = np.memmap(self.path + self.signal_path, 
                         dtype= self.params['dtype'])
        
        # Whitened Kilosort output
        if self.signal_path == 'temp_wh.dat':
            #print('Loading Kilosort output signal.')
            data = data.reshape([-1, 383])     
            return data
        
        # Raw Open Ephys output
        if self.signal_path == 'continous.dat':
            #print('Loading Open Ephys output signal.')
            data = data.reshape([-1, 384])
            
            # Todo Remove channel 190!
            
            return data
        
        # Spike GLX output
        if self.signal_path[-4:] == '.bin':

            data = data.reshape([-1, 385])
            # Todo Remove channel 190.
            # Shall we leave channel 384 (sync pulses)?
                
            return 
        
        # RAISE ERROR
        
    
    def _build_waveform(self, channel:int, chan_range=[-10, 10]):
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
        indexer = (spike_channel_list>=channel + chan_range[0]) & \
            (spike_channel_list<=channel + chan_range[1])
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
        files = ['cluster_Amplitude.tsv', 'cluster_ContamPct.tsv', \
                  'cluster_group.tsv']
        clusters = [self._load_tsv(self.path + file) for file in files]
        self.clusters = pd.concat(clusters, axis=1)
                   
        # Load all spike times
        self.spikeTimes = np.load(self.path + 'spike_times.npy').ravel().astype(int)
        self.spikeID = np.load(self.path + 'spike_clusters.npy').ravel().astype(int)
        
        # Add the Spike count to the data
        for i in self.clusters.index:
            self.clusters.loc[i, 'spikeCount'] = (self.spikeID==i).sum()

        # Delete units with 0 spikes
        self.clusters = self.clusters[self.clusters.spikeCount>0]
            
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

        # Finally, load the similarity matrix
        self.similarity_matrix = np.load(self.path + 'similar_templates.npy')

        
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
            unit_channel_list = np.load(self.path + 'pks_data/unit_channel_list.npy')
            spike_chanel_list = np.load(self.path + 'pks_data/spike_channel_list.npy')
            return unit_channel_list, spike_chanel_list 
        
        # If not, we'll have to build it    
        print('Building channel list (only necesary on first run).')
        
        unit_list = self.clusters.index.values
        channel_hash = {unit:self._unit_channel(unit) for unit in unit_list}
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

    
    def _load_tsv(self, file:str):
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


    def _check_path(self, path:str):
        """
        Quick error handeling of the input parameter
    
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