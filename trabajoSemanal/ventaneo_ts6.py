# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Tarea semanal N°6

"""
# Inclusión de librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Inicialización de variables
plt.close('all')
fs=1000
N = 1000
df = fs/N # resolución espectral
zero_padding = 10
#ff = np.linspace(0, (N-1)*df, N)
ff = np.linspace(0, (N-1)*(df), (zero_padding+1)*N) #zero_padding veces N + N de la señal

# Cálculo de ventanas
# winBartlett = np.bartlett(N)
winBartlett = scipy.signal.windows.bartlett(N)
winHann = scipy.signal.windows.hann(N)
winBlackman = scipy.signal.windows.blackman(N)
winFlatTop = scipy.signal.windows.flattop(N)

#Aplicamos el zero padding
winBartlett = np.append(winBartlett, np.zeros(zero_padding*N))
winHann = np.append(winHann, np.zeros(zero_padding*N))
winBlackman = np.append(winBlackman, np.zeros(zero_padding*N))
winFlatTop = np.append(winFlatTop, np.zeros(zero_padding*N))


#Transformada de Fourier de las ventanas
ft_winBartlett = np.fft.fft(winBartlett) / winBartlett.shape[0]
ft_winHann = np.fft.fft(winHann) / winHann.shape[0]
ft_winBlackman = np.fft.fft(winBlackman) / winBlackman.shape[0]
ft_winFlatTop = np.fft.fft(winFlatTop) / winFlatTop.shape[0]

bfrec = ff <= fs/2

# Visualización de amplitud respecto las muestras
plt.figure(1)
plt.plot(winBartlett, 'red',label='Bartlett')
plt.plot(winHann, 'green',label='Hann')
plt.plot(winBlackman, 'blue',label='Blackman')
plt.plot(winFlatTop, 'black',label='Flat-Top')
plt.grid()
plt.xlim([0,1000])
axes_hdl = plt.gca()
axes_hdl.legend()

# Visualización de la respuesta en frecuencia
plt.figure(2)
plt.plot(ff[bfrec], 20* np.log10(np.abs(ft_winBartlett[bfrec])/np.abs(ft_winBartlett[0])), 'red',label='Bartlett')
plt.plot(ff[bfrec], 20* np.log10(np.abs(ft_winHann[bfrec])/np.abs(ft_winHann[0])), 'green',label='Hann')
plt.plot(ff[bfrec], 20* np.log10(np.abs(ft_winBlackman[bfrec])/np.abs(ft_winBlackman[0])), 'blue',label='Blackman')
plt.plot(ff[bfrec], 20* np.log10(np.abs(ft_winFlatTop[bfrec])/np.abs(ft_winFlatTop[0])), 'black',label='Flat-Top')




plt.grid()
plt.xlim([0,10])
plt.ylim([-300,10])
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()