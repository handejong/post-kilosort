#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions and methods used to verify if units are opto-tagged


Created: Fri Jul  20 11:40:25 2023
Last Updated: Jul  20 11:40:25 2023

@author: Han de Jong
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class opto_tagging:

    def __init__(self, pks_dataset):
        # Store the datset itself
        self.data = pks_dataset

        # Empty parameters
        light_pulses = np.array([])


    def set_parameters(self, stamps, window):
        """
        This method is used to set the opto-tagging parameters.

        Parameters:
        -----------
        stamps: np.array
        	List of timestamps (in s) at which light pulses occured
        window: list or tuple
        	A window (in ms!) at which we expec the light response

        Returns:
        None

        Example:
        	data = pks.dataset(path)
        	stamps = data.get_nidq()
        	stamps = stamps[(stamps.Channel=='AI_1') & (stamps.Duration==5)].Start
        	self.opto_tagging.set_parameters(stamps, (0, 5))

        You have now set the opto_tagging object to look at stamps that light
        pulses that were collected on AI_1 and lasted 5 ms. You will analyse
        action potentials that occured in the window from 0 to 5 ms after light
        pulse onset.
        """

        # Save the stamps
        if stamps.__class__ == pd.DataFrame:
        	self.stamps = stamps.Start.values
        elif stamps.__class__ == np.ndarray:
        	self.stamps = stamps
        else:
        	self.stamps = np.array(stamps)

        # Save the interval
        self.window = (window[0]*10**-3, window[1]*10**-3)



    def get_tagged_spikes(self, unit, return_real_time = False):
        """
        Get the spikes of a unit that occured in the window after light pulse
        onset.

        Parameters:
        -----------
        return_real_time: Bool
        	Return the data index or the timepoint (in sec)

        Returns:
        	np.array of spikeTimes

        """

        # Get the spikes for this unit
        spikes = self.data.get_unit_spikes(unit, return_real_time=True)
        mask = spikes[:, np.newaxis] - self.stamps
        mask = np.any((mask>self.window[0]) & (mask<self.window[1]), axis=1)

        if return_real_time:
            return spikes[mask]
        else:
            spikes = self.data.get_unit_spikes(unit)
            return spikes[mask]


    def add_tagged_spikes(self, unit):
        """
        Will add the tagged waveforms to the main PKS_dataset object.
        It does so by adding a (fake) unit to the dataset with the
        "99900" prefix.
        """

        # Grab the spikes
        spikes = self.get_tagged_spikes(unit, return_real_time = False)

        # Unit name
        unit_name = 99900000 + unit

        # Add the unit to the main object
        temp = [unit_name] * len(spikes)
        spikeID = np.append(self.data.spikeID, temp)
        self.data.spikeID = spikeID
        spikeTimes = np.append(self.data.spikeTimes, spikes)
        self.data.spikeTimes = spikeTimes

        # Add the "unit" to the clusters overview
        self.data.clusters.loc[unit_name, :] = self.data.clusters.loc[unit, :]
        self.data.clusters.loc[unit_name, 'spikeCount'] = len(spikes)

        # This is super anoying but necesary
        # Is it though?
        self.data.clusters.mainChannel = self.data.clusters.mainChannel.astype(int)
        self.data.clusters.spikeCount = self.data.clusters.spikeCount.astype(int)

        # Update the similarity matrix
        self.data.similarity_matrix.loc[unit_name, :] = self.data.similarity_matrix[unit]
        self.data.similarity_matrix.loc[:, unit_name] = self.data.similarity_matrix[unit]
        self.data.similarity_matrix.loc[unit_name, unit_name] = 1

        # If there are any plots open, plot the tagged waveforms
        if len(self.data.linked_plots)>0:
            self.data.plot.add_unit(unit_name)


    def collision_test(self, unit, window=(0, 10)):
        """
        Specifically for antidromic opto-tagging
        """

        # Convert window to s
        window = (window[0]*10**-3, window[1]*10**-3)

        # Get the spikes and make the mask
        spikes = self.data.get_unit_spikes(unit, return_real_time=True)
        mask = spikes[:, np.newaxis] - self.stamps
        mask = np.any((mask>window[0]) & (mask<window[1]), axis=0)

        # Look at non-collision trials first
        non_collision_stamps = self.stamps[~mask]
        non_collision_plot = self.data.plot.peri_event(units=unit, 
            stamps=non_collision_stamps, 
            peri_event_window=(-0.020, 0.020))
        plt.suptitle('Free trials', color=self.data.plot.axcolor)

        # Look at collision trials
        collision_stamps = self.stamps[mask]
        collision_plot = self.data.plot.peri_event(units=unit, 
            stamps=collision_stamps, 
            peri_event_window=(-0.020, 0.020))
        plt.suptitle('Collision trials', color=self.data.plot.axcolor)


    def remove_all_tagged(self):
        """
        Simply removes all units from the dataset that are just collections
        of tagged waveforms. Will not delete any units or spikes!
        """

        # Make sure manipulations not saved
        save_state = self.data.save_data
        self.data.save_data = False

        # Grab all tagged waveforms "units" and delete them
        for unit in self.data.clusters.index:
            if unit > 10000:
                if str(unit)[:3] == '999':
                    self.data.sort.delete_unit(unit)

        # Go back to the original save state
        self.data.save_data = save_state


    def light_artifact(self, window = 10, sample_n = 10, offset = None):
        """
        This function will give you an idea of the light artifact.

        """

        # Convert light pulses to indexes
        indexes = self.data.convert_time_to_index(self.stamps)
        
        # Plot them
        self.data.plot.raw_unit_sample(indexes, sample_n = sample_n, 
                                       window = window,
                                       offset = offset)
        plt.suptitle(f'Average of {sample_n} waveforms', color=self.data.plot.axcolor)

        # Also plot one particular waveform
        indexes = np.random.choice(indexes, 2)
        example_time = self.data.convert_index_to_time(np.random.choice(indexes, 2))[0]
        self.data.plot.raw_unit_sample(indexes, sample_n = 1, 
                                       window = window,
                                       offset = 1000)
        plt.suptitle(f'Example waveform at T={example_time}', color=self.data.plot.axcolor)
        
        return None
    
    def light_response(self, units=None, window=20):
            """
            Will give a simply peri-event plot for this unit relative to the light pulses

            """

            self.data.plot.peri_event(units=units, stamps=self.stamps, 
                                    peri_event_window=(-0.010, window/1000))
