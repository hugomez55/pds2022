# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Generador de señales senoidales y Transformada Discreta de Fourier

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

#   Funcion que realiza la transformada discreta de Fourier

def funcionDFT(xx):
    N=len(xx)
    
    n=np.arange(N)
    
    k=n.reshape((N,1))
    
    twiddle = np.exp(-2j*np.pi*k*n/N)
    
    X=np.dot(twiddle,xx)
    
    return X

#   Funcion para visualizar de 0 a Nyquist
def partePositiva(X):
    
    N_pos = N//2
    
    #Nuevo array de 0 hasta Nyquist
    ff_pos = ff [ : N_pos]
    
    #Normalizado
    X_pos = X [ : N_pos]/N_pos
   

    return ff_pos, X_pos
    
    
    
#%% Presentación gráfica de los resultados
# Inicialización de variables 
N=1000
fs=1000
ts=1/fs


# Invocamos a nuestro testbench exclusivamente: 
#Armo una señal generada por la suma de tres senoidales

tt,x1 = funcionSeno( vmax=.5, dc=0, ff=100, ph=0, nn=N, fs=fs)

tt,x2 = funcionSeno( vmax=1, dc=0, ff=300, ph=0, nn=N, fs=fs)

tt,x3 = funcionSeno( vmax=.2, dc=0, ff=450, ph=0, nn=N, fs=fs)

xx=x1+x2+x3

plt.figure(1)
plt.clf()
plt.plot(tt, xx)
plt.title('Señal: Senoidal')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')
plt.grid(which='both', axis='both')
#plt.show()

X = funcionDFT(xx)
n=np.arange(N)
T=N/fs
#array de N elementos separadas por la resolución espectral
ff=n/T

plt.figure(2)
plt.clf()
plt.plot(ff,abs(X))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo DFT')

ff_pos, X_pos = partePositiva(X) #Resultado de DFT

plt.figure(3)
plt.clf()
plt.plot(ff_pos, abs(X_pos))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo |X|')
    
plt.show()
    
    


