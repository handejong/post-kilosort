#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""




Created: Fri Nov  8 13:51:11 2022
Last Updated: Nov 11 14:01:20 2022

@author: Han de Jong
"""

def _calc_amplitude(waveforms, other=None):
    """
    Calculate the amplitude of each waveform in waveforms
    Parameters
    ----------
    waveforms : 2-D NumPy Array
        Each row should contain a waveform

    Returns
    -------
    Numpy array
        The amplitude of each waveform in the input dataset.

    """
    
    return waveforms.max(axis=1) + -1*waveforms.min(axis=1)

    
def _calc_peak(waveforms, other=None):
    
    # Look only at index window
    
    # 
    
    pass


# Make the functions available
if not __name__ =='__main__':
    _calc_amplitude = _calc_amplitude
    _calc_peak = _calc_peak
