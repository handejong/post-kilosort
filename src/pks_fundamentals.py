#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to inspect unit fundamentals such as baseline firing frequency and waveform width

Created: Mon Oct 2 14:35 2023

@author: Han de Jong
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

def waveform_width(waveform, plot = False, sample_rate = 30):
    # Responsible for calculating waveform height and width at half maximum
    
    # Grab the first details
    details = {}
    details['width'] = []

    X = np.linspace(0, len(waveform)-1, len(waveform))
    X_new = np.linspace(0, len(waveform)-1, int((len(waveform)/sample_rate)*1000))

    # find and subtract the baseline
    waveform = waveform - waveform[0:15].mean()

    # Invert the waveform is neccesary
    if waveform[int(0.4*len(waveform)):int(0.5*len(waveform))].sum(axis=0) > 0:
        waveform = waveform * -1

    # Interpolate to microsec scale
    waveform_micro = np.interp(X_new, X, waveform.astype(float))

    # If plot, plot the waveform
    if plot:
        plt.figure()
        plt.plot(X_new*(1000/sample_rate), waveform_micro)
        plt.ylabel('Arbitrary Units'); plt.xlabel('Time (µs)')
    
    # Method one (with at half peak)
    threshold = np.min(waveform)/2
    details['width'] = int(1 + np.sum(waveform_micro<=threshold))
    width = details['width']
    
    # Plot this one if requested
    if plot:
        indexer = waveform_micro<=threshold
        X_temp = X_new[indexer]*(1000/sample_rate)
        Y_temp = np.ones(X_temp.shape)*threshold
        plt.plot(X_temp, Y_temp, label=f'width ({width}µs)')
    
    # Alternative way
    threshold = np.std(waveform[0:30])
    start = int(X[np.abs(waveform)>3*threshold].min())
    end = int(np.argmax(waveform[int(len(waveform)/2):]) + int(len(waveform)/2))
    details['waveform_width'] = int((end - start) * (1000/sample_rate))
    width = details['waveform_width']
    end_time = end*(1000/sample_rate)
    
    # Plot this method as well
    if plot:
        start_time = start*(1000/sample_rate)
        plt.plot([start_time, end_time], [waveform[start],\
                      waveform[start]], label=f'waveform_width ({width}µs)')
    
    # Another way, from the bottom of the peak to the max
    start = X_new[np.argmin(waveform_micro)]
    start_time = start*(1000/sample_rate)
    details['width_method_3'] = abs(int(end_time - start_time))
    width = details['width_method_3']
    
    # Plot the last one as well
    if plot:
        Y_temp = [np.min(waveform_micro), np.min(waveform_micro)]
        plt.plot([start_time, end_time], Y_temp,\
                 label=f'width_method_3 ({width}µs)')
        plt.plot([end_time, end_time], [Y_temp[0], waveform[end]],\
                 color='k', linestyle='--', linewidth=0.5)
        plt.legend()
        
    return details
