#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:29:36 2022

@author: hugomez
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
from scipy import signal


mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'] 

#plt.plot(ecg_one_lead)


f, Pxx_den = signal.welch(ecg_one_lead)
plt.semilogy(f, Pxx_den)
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()