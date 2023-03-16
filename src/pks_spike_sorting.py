#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""




Created: Fri Nov  8 13:51:11 2022
Last Updated: Nov 11 14:01:20 2022

@author: Han de Jong
"""

import numpy as np
from datetime import datetime


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
        self.data.clusters = self.data.clusters[self.data.clusters.index != unit_1]

        # Save manipulations
        if self.data.save_data:
            with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
                f.write(
                    f'self.sort.merge_units({unit_1}, {unit_2}) #{self._timestamp()}\n')

        # Update any plots
        self.update_plots_remove(unit_1)
        self.update_plots(unit_2)

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

    def neighbors(self, unit, o:int = 3, min_spikes:int = 0 ):
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

        return neighbors

    def update_plots_remove(self, unit):

        linked_plots = self.data.linked_plots
        _ = [i.remove_unit(unit) for i in linked_plots]

    def update_plots(self, unit):

        linked_plots = self.data.linked_plots
        _ = [i.update(unit) for i in linked_plots]

    def _timestamp(self):

        temp = datetime.now()
        date = temp.date().isoformat()
        time = temp.time().isoformat()[:-7]

        return date + ' ' + time
