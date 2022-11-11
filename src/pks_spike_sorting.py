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


	def delete_unit(self, unit, verbose=True):

		# Print what we are doing
		if verbose:
			print(f'Deleting unit {unit}')

		# Remove row from clusters
		self.data.clusters = self.data.clusters[self.data.clusters.index != unit]

		# Remove spikes
		self.data.spikeTimes[self.data.spikeID != unit]
		self.data.spikeID[self.data.spikeID != unit]

		# Save manipulations
		if self.data.save_data:
			with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
				f.write(f'self.sort.delete_unit({unit}) #{self._timestamp()}\n')
		
		# Update any plots
		self.update_plots_remove(unit)


	def merge_units(self, unit_1, unit_2, verbose=True):

		if verbose:
			print(f'Merging unit {unit_1} into unit {unit_2}.')

		# Merge spikes
		self.data.spikeID[self.data.spikeID==unit_1] = unit_2

		# Remove row from clusters
		self.data.clusters = self.data.clusters[self.data.clusters.index != unit_1]

		# Save manipulations
		if self.data.save_data:
			with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
				f.write(f'self.sort.merge_units({unit_1}, {unit_2}) #{self._timestamp()}\n')

		# Update any plots
		self.update_plots_remove(unit_1)
		self.update_plots(unit_2)


	def translate(self, unit, offset, verbose=True):

		if verbose:
			print(f'Offseting unit {unit} by {offset} datapoints.')

		# Offset
		indexer = self.data.spikeID==unit
		self.data.spikeTimes[indexer] = self.data.spikeTimes[indexer] + offset

		# Save manipulations
		if self.data.save_data:
			with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
				f.write(f'self.sort.translate({unit}, {offset}) #{self._timestamp()}\n')

		# Update any plots
		self.update_plots(unit)


	def mark_done(self, unit, verbose=True):

		if verbose:
			print(f'Marking unit {unit} as done.')

		# Mark
		self.data.done.append(unit)

		# Save manipulations
		if self.data.save_data:
			with open(self.data.path + 'pks_data/changeSet.py', 'a') as f:
				f.write(f'self.sort.mark_done({unit}) #{self._timestamp()}\n')

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