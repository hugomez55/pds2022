# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Generador de señales senoidales

"""

import numpy as np
import matplotlib.pyplot as plt

#%%  Inicialización

#%%  Testbench: creamos una función que no recibe argumentos para asegurar que siempre encontraremos nuestro espacio de variables limpio.

def funcionSeno (vmax, dc, ff, ph, nn, fs):
    
    # Grilla de sampleo temporal
    tt = np.arange(0.0, nn/fs, 1/fs)
    # Calculo de los valores punto a punto de la funcion
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc
    
    return tt,xx
    
#%% Presentación gráfica de los resultados
# Inicialización de variables 
N=1000
fs=1000
ts=1/fs


# Invocamos a nuestro testbench exclusivamente: 

tt,xx = funcionSeno( vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs)


    
plt.figure(1)
plt.plot(tt, xx)
plt.title('Señal: Senoidal')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')
plt.grid(which='both', axis='both')
plt.show()
    
    
