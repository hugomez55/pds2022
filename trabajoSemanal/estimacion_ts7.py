# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Tarea semanal N°7

"""
# Inclusión de librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Inicialización de variables
plt.close('all')
fs=1000
N = 1000
df = fs/N # resolución espectral
amp = 2
freq = fs/4
ph = 0
dc = 0
muestras = 200
omega_ini = np.pi/2
ts = 1/fs
Wbins = 1

#tt = np.linspace(0, (N-1)*ts, N)

noise = (np.random.rand(1,muestras)-0.5)*2

tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)

omega = (omega_ini + noise * (np.pi*2/N))*tt*fs

signal = amp*np.sin(omega)

#ft_signal = np.fft.fft(signal) / signal.shape[0]
ft_signal = np.fft.fft(signal, axis=0) / signal.shape[0]

ff = np.arange(0, fs, fs/N)
bfrec = ff<=(fs/2)

estimador1 = np.abs(ft_signal[250, :])*2 #[muestra 250, realizaciones todas]

densidadPotencia = 2*np.abs(ft_signal)**2
subMatriz = densidadPotencia[250-Wbins:250+Wbins+1, :]

potenciaEstimada = np.sum(subMatriz, axis=0)
amplitudEstimada = np.sqrt(2*potenciaEstimada)

estimadores = np.vstack([estimador1, amplitudEstimada]).transpose()

plt.figure(1)
#En veces
plt.plot(ff[bfrec],(2*np.abs(ft_signal[bfrec , :])**2))

#En dB
#plt.plot(ff[bfrec], 20*np.log10(2*np.abs(ft_signal[bfrec, :])**2))


#plt.figure(2)
fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.hist(estimador1)
ax.hist(estimadores)

mediana = np.median(estimadores, axis=0)
sesgo = np.median(estimadores, axis=0) - amp
varianza = np.mean((estimadores - mediana)**2, axis=0)


plt.show()


#%%
# # Cálculo de ventanas
# winBartlett = np.bartlett(N)
# winHann = np.hanning(N)
# winBlackman = np.blackman(N)
# winFlatTop = scipy.signal.windows.flattop(N)

# #Transformada de Fourier de las ventanas
# ft_winBartlett = np.fft.fft(winBartlett) / winBartlett.shape[0]
# ft_winHann = np.fft.fft(winHann) / winHann.shape[0]
# ft_winBlackman = np.fft.fft(winBlackman) / winBlackman.shape[0]
# ft_winFlatTop = np.fft.fft(winFlatTop) / winFlatTop.shape[0]


# # Visualización de amplitud respecto las muestras
# plt.figure(1)
# plt.plot(winBartlett, 'red',label='Bartlett')
# plt.plot(winHann, 'green',label='Hann')
# plt.plot(winBlackman, 'blue',label='Blackman')
# plt.plot(winFlatTop, 'black',label='Flat-Top')
# plt.grid()
# axes_hdl = plt.gca()
# axes_hdl.legend()

# # Visualización de la respuesta en frecuencia
# plt.figure(2)
# plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_winBartlett[bfrec])**2), 'red',label='Bartlett')
# plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_winHann[bfrec])**2), 'green',label='Hann')
# plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_winBlackman[bfrec])**2), 'blue',label='Blackman')
# plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_winFlatTop[bfrec])**2), 'black',label='Flat-Top')
# plt.grid()
# plt.xlim([0,10])
# plt.ylim([-400,10])
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.show()