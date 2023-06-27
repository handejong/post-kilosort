#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""




Created: Fri Nov  8 13:51:11 2022
Last Updated: Nov 11 14:01:20 2022

@author: Han de Jong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
import os

class sorter:

    def __init__(self, pks_dataset):

        # Store the datset itself
        self.data = pks_dataset

        # Do we print text about what we are doing?
        self.verbose = True

    def delete_unit(self, units):

        # Multiple units or just 1?
        if units.__class__ == int:
            units = [units]

        for unit in units:

            # Print what we are doing
            if self.verbose:
                print(f'Deleting unit {unit}')

            # Remove row from clusters
            self.data.clusters = self.data.clusters[self.data.clusters.index != unit]

            # Remove spikes
            self.data.spikeTimes[self.data.spikeID != unit]
            self.data.spikeID[self.data.spikeID != unit]

            # Save manipulations
            if self.data.save_data:
                with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                    f.write(
                        f'self.sort.delete_unit({unit}) #{self._timestamp()}\n')

            # Update any plots
            self.update_plots_remove(unit)

    def merge_units(self, unit_1, unit_2):

        if self.verbose:
            print(f'Merging unit {unit_1} into unit {unit_2}.')

        # Merge spikes
        self.data.spikeID[self.data.spikeID == unit_1] = unit_2

        # Remove row from clusters
        spikeCount = self.data.clusters.loc[unit_1].spikeCount
        self.data.clusters = self.data.clusters[self.data.clusters.index != unit_1]
        self.data.clusters.loc[unit_2, 'spikeCount'] += spikeCount

        # Save manipulations
        if self.data.save_data:
            with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                f.write(
                    f'self.sort.merge_units({unit_1}, {unit_2}) #{self._timestamp()}\n')

        # Update any plots
        self.update_plots_remove(unit_1)
        self.update_plots(unit_2)

    def split_unit(self, unit, channels = None, k = 2):
        """
        SPLIT_UNIT will split a unit using k-means clustering with k clusters.

        For now only k-means clustering is supported, but this will be updated
        in the future. The k-means clustering is done on PCA-transformed (first-
        PC) of the provided channels.

        Parameters:
        ----------
        unit: the unit that needs to be split
        channels: a list of channels that should be used for the clustering

        Returns:
        --------
        None

        Note: if self.save_data is True, the manipulations is automatically saved.

        """

        # Infer channels
        _, channels = self.data._infer_unit_channels(unit, channels)
    
        # Get the unit spikes
        spikes = self.data.get_unit_spikes(unit)

        # Convert to PCA space
        workdata = np.zeros((len(spikes), len(channels)))
        for i, channel in enumerate(channels):
            pca = self.data.channel_pca(channel) # PCA
            waveforms = self.data.get_waveform(spikes, channel=channel, 
                    average=False) # Waveforms (this is the painfull one)
            workdata[:, i] = pca.transform(waveforms).flatten()

        # Now we have to do some clustering
        kmeans = KMeans(k)
        kmeans.fit(workdata)
        labels = kmeans.labels_

        # Plot the result
        fig, axs = plt.subplots(len(channels), len(channels),
            figsize = [10, 10], 
            tight_layout = True)
        for i, channel_i in enumerate(channels):
            for j, channel_j in enumerate(channels):
                if i==j:
                    for new_unit in range(k):
                        waveform = self.data.get_waveform(spikes[labels==new_unit], 
                            sample = 1000, channel=channel_i)
                        axs[i, j].plot(waveform)
                else:
                    axs[i, j].scatter(x=workdata[:, j], y=workdata[:, i], c=labels, s=0.1)

                # Formatting
                if j==0:
                    axs[i, j].set_ylabel(channel_i)
                if i==0:
                    axs[i, j].set_title(channel_j)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        # Confirm with the user
        ans = input('Accept this split? (y/n) :')

        # Execute
        if ans == 'y':

            # Get the old index
            old_index = self.data.clusters.index

            # Save the split
            name = f'unit_{unit}_k-means_{k}_{self._timestamp()}.npy'
            self._save_split(labels, name)

            # Execute split
            self._execute_split(unit, name)

            # Get any new units
            new_units = [i for i in self.data.clusters.index if i not in old_index]

            # Update any open plots
            self.data.plot.remove_unit(unit)
            self.data.plot.add_unit(unit)
            for new_unit in new_units:
                self.data.plot.add_unit(new_unit)

        # Close the figure
        plt.close(fig)


    def translate(self, unit, offset):

        if self.verbose:
            print(f'Offseting unit {unit} by {offset} datapoints.')

        # Offset
        indexer = self.data.spikeID == unit
        self.data.spikeTimes[indexer] = self.data.spikeTimes[indexer] + offset

        # Save manipulations
        if self.data.save_data:
            with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                f.write(
                    f'self.sort.translate({unit}, {offset}) #{self._timestamp()}\n')

        # Update any plots
        self.update_plots(unit)

    def mark_done(self, units):

        # Multiple units or just 1?
        if units.__class__ == int:
            units = [units]

        for unit in units:
            if self.verbose:
                print(f'Marking unit {unit} as done.')

            # Mark
            self.data.done.append(unit)
            self.data.clusters.loc[unit, 'done'] = True

            # Save manipulations
            if self.data.save_data:
                with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                    f.write(
                        f'self.sort.mark_done({unit}) #{self._timestamp()}\n')

            # Update the plots
            self.update_plots_remove(unit)

    def todo(self, n=10):
        """
        Prints the next 'n' units that we user should work on.

        Parameters
        ----------
        n : int
            Number of units to show

        Returns
        -------
        None

        """

        # Grab the top 10 units that are not marked done
        indexer = [i not in self.data.done for i in self.data.clusters.index]
        clusters = self.data.clusters[indexer].iloc[:10, :]

        return clusters

    def neighbors(self, unit, o:int = 3, min_spikes:int = 0):
        """
        Shows the units (prints to terminal) the units that are on the channels
        close to "unit".

        Paramters
        ---------
        o : int
            How close the the unit (n channels) has to be
        minimum_spikes : int
            How many spikes the units should have to be included.
        """

        channel = self.data.clusters.loc[unit].mainChannel
        neighbors = self.data.clusters.query("mainChannel>@channel-@o & mainChannel<@channel+@o")
        neighbors = neighbors[neighbors.spikeCount>min_spikes]

        # Add the similarity and sort by channel and similarity
        neighbors['similarity'] = self.data.similarity_matrix.loc[neighbors.index, unit]
        neighbors = neighbors.sort_values(['mainChannel', 'similarity'], ascending=[True, False])

        return neighbors

    def update_plots_remove(self, unit):

        linked_plots = self.data.linked_plots
        _ = [i.remove_unit(unit) for i in linked_plots]

    def update_plots(self, unit):

        linked_plots = self.data.linked_plots
        _ = [i.update(unit) for i in linked_plots]


    def _execute_split(self, unit, name):

        # get the split
        split = np.load(self.data.path + 'pks_data/splits/' + name)

        # Execute
        indices = np.where(self.data.spikeID==unit)[0]

        # For every unique value in split
        for i in np.unique(split):
            if not i == 0:

                # Update the spikeID
                new_id = self.data.clusters.index.max()+1
                self.data.spikeID[indices[split==i]] = new_id

                # Add a line to the clusters dataframe
                self.data.clusters.loc[new_id, :] = self.data.clusters.loc[unit, :]
                self.data.clusters.loc[new_id, 'spikeCount'] = np.sum(split==i)
                self.data.clusters.mainChannel = self.data.clusters.mainChannel.astype(int) # Super annoying
                self.data.clusters = self.data.clusters.sort_values('mainChannel')

                # Update the similarity matrix
                self.data.similarity_matrix.loc[new_id, :] = self.data.similarity_matrix[unit]
                self.data.similarity_matrix.loc[:, new_id] = self.data.similarity_matrix[unit]
                self.data.similarity_matrix.loc[new_id, new_id] = 1

                # Print a note
                print('NOTE: for split units there is no Kilosort template')
                print('The similarity matrix and waveform paramters are just copied from the original unit')

            else:
                self.data.clusters.loc[unit, 'spikeCount'] = np.sum(split==i)

        # Save manipulations
        if self.data.save_data:
            with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                f.write(
                    f"self.sort._execute_split({unit}, '{name}') #{self._timestamp()}\n")
        

    def _save_split(self, split, name):

        # Check if the folder with splits exists
        if not os.path.isdir(self.data.path + 'pks_data/splits'):
            os.mkdir(self.data.path + 'pks_data/splits')

        # Save the split
        np.save(self.data.path + 'pks_data/splits/' + name, split)

    def _timestamp(self):

        temp = datetime.now()
        date = temp.date().isoformat()
        time = temp.time().isoformat()[:-7]

        return date + ' ' + time
