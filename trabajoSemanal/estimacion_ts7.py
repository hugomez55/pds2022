# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Tarea semanal N°7

"""
#%% Inclusión de librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas
from pandas import DataFrame
from IPython.display import HTML

#%% Inicialización de variables
plt.clf()
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

#%% Ventanas

winBartlett = sig.windows.bartlett(N).reshape(N,1)
winHann = sig.windows.hann(N).reshape(N,1)
winBlackman = sig.windows.blackman(N).reshape(N,1)
winFlatTop = sig.windows.flattop(N).reshape(N,1)

noise = (np.random.rand(1,muestras)-0.5)*2

tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)

omega = (omega_ini + noise * (np.pi*2/N))*tt*fs

seno = amp*np.sin(omega)

bartlettSeno = seno*winBartlett
hannSeno = seno*winHann
blackmanSeno = seno*winBlackman
flatTopSeno = seno*winFlatTop

# FFT
senoFFT = np.fft.fft(seno, axis=0) / seno.shape[0]
bartlettFFT = np.fft.fft(bartlettSeno, axis = 0)/bartlettSeno.shape[0]
hannFFT = np.fft.fft(hannSeno, axis = 0)/hannSeno.shape[0]
blackmanFFT = np.fft.fft(blackmanSeno, axis = 0)/blackmanSeno.shape[0]
flatTopFFT = np.fft.fft(flatTopSeno, axis = 0)/flatTopSeno.shape[0]

ff = np.arange(0, fs, fs/N)
bfrec = ff<=(fs/2)

# plt.figure(1)
# plt.plot(ff[bfrec], 10*np.log10(2*np.abs(bartlettFFT[bfrec, :])**2))
# plt.title("Bartlett")

#%% Estimadores
# En clase
# estimador1 = np.abs(senoFFT[250, :])*2 #[muestra 250, realizaciones todas]
# densidadPotencia = 2*np.abs(senoFFT)**2
# subMatriz = densidadPotencia[250-Wbins:250+Wbins+1, :]

# potenciaEstimada = np.sum(subMatriz, axis=0)
# amplitudEstimada = np.sqrt(2*potenciaEstimada)
# estimadores = np.vstack([estimador1, amplitudEstimada]).transpose()

#Estimador 3A
bartlettEstAmp = np.abs(bartlettFFT[250, :])*2
hannEstAmp = np.abs(hannFFT[250, :])*2
blackmanEstAmp = np.abs(blackmanFFT[250, :])*2
flatTopEstAmp = np.abs(flatTopFFT[250, :])*2

#Estimador 3B
# Bartlett
bartlettDenPot = 2*np.abs(bartlettFFT)**2
bartlettSubMatriz = bartlettDenPot[250-Wbins:250+Wbins+1,:]
bartlettPotEst = np.sum(bartlettSubMatriz, axis=0)
bartlettAmpEst = np.sqrt(2*bartlettPotEst)
# Hann
hannDenPot = 2*np.abs(hannFFT)**2
hannSubMatriz = hannDenPot[250-Wbins:250+Wbins+1,:]
hannPotEst = np.sum(hannSubMatriz, axis=0)
hannAmpEst = np.sqrt(2*hannPotEst)
# Blackman
blackmanDenPot = 2*np.abs(blackmanFFT)**2
blackmanSubMatriz = blackmanDenPot[250-Wbins:250+Wbins+1,:]
blackmanPotEst = np.sum(blackmanSubMatriz, axis=0)
blackmanAmpEst = np.sqrt(2*blackmanPotEst)
# Flat-Top
flatTopDenPot = 2*np.abs(flatTopFFT)**2
flatTopSubMatriz = flatTopDenPot[250-Wbins:250+Wbins+1,:]
flatTopPotEst = np.sum(flatTopSubMatriz, axis=0)
flatTopAmpEst = np.sqrt(2*flatTopPotEst)

bartlettEstimadores = np.vstack([bartlettEstAmp, bartlettAmpEst]).transpose()
hannEstimadores = np.vstack([hannEstAmp, hannAmpEst]).transpose()
blackmanEstimadores = np.vstack([blackmanEstAmp, blackmanAmpEst]).transpose()
flatTopEstimadores = np.vstack([flatTopEstAmp, flatTopAmpEst]).transpose()

# Bartlett
bartlettMediana = np.median(bartlettEstimadores, axis=0)
bartlettSesgo = np.median(bartlettEstimadores, axis=0) - amp
bartlettVarianza = np.mean((bartlettEstimadores - bartlettMediana)**2, axis=0)
# Hann
hannMediana = np.median(hannEstimadores, axis=0)
hannSesgo = np.median(hannEstimadores, axis=0) - amp
hannVarianza = np.mean((hannEstimadores - hannMediana)**2, axis=0)
# Blackman
blackmanMediana = np.median(blackmanEstimadores, axis=0)
blackmanSesgo = np.median(blackmanEstimadores, axis=0) - amp
blackmanVarianza = np.mean((blackmanEstimadores - blackmanMediana)**2, axis=0)
# Flat-Top
flatTopMediana = np.median(flatTopEstimadores, axis=0)
flatTopSesgo = np.median(flatTopEstimadores, axis=0) - amp
flatTopVarianza = np.mean((flatTopEstimadores - flatTopMediana)**2, axis=0)

#Agrupo estimadores
estimador3A = np.vstack([bartlettEstAmp, hannEstAmp, blackmanEstAmp, flatTopEstAmp]).transpose()
estimador3B = np.vstack([bartlettAmpEst, hannAmpEst, blackmanAmpEst, flatTopAmpEst]).transpose()

plt.figure(1)
plt.title("Estimador 3A")
kwargs = dict(alpha=0.5, bins=10, density=False, stacked=True)
# kwargs2 = dict(alpha=0.5, bins=2, density=False, stacked=True)
plt.hist(estimador3A[:,0], **kwargs, label='Bartlett')
plt.hist(estimador3A[:,1], **kwargs, label='Hann')
plt.hist(estimador3A[:,2], **kwargs, label='Blackman')
plt.hist(estimador3A[:,3], **kwargs, label='Flat-Top')
plt.legend()
plt.show()


# resultados3A = [ 
#                    [bartlettSesgo[0], bartlettVarianza[0]], 
#                    [hannSesgo[0], hannVarianza[0]],
#                    [blackmanSesgo[0], blackmanVarianza[0]],
#                    [flatTopSesgo[0], flatTopVarianza[0]], 
#                  ]
# df3A = DataFrame(resultados3A, columns=['$s_a$', '$v_a$'],
#                index=[
#                        'Bartlett',
#                        'Hann',
#                        'Blackman',
#                        'Flat-Top'               
#                        ])
# HTML(df3A.to_html())

# resultados3B = [ 
#                    [bartlettSesgo[1], bartlettVarianza[1]], 
#                    [hannSesgo[1], hannVarianza[1]],
#                    [blackmanSesgo[1], blackmanVarianza[1]],
#                    [flatTopSesgo[1], flatTopVarianza[1]], 
#                  ]
# df3B = DataFrame(resultados3B, columns=['$s_a$', '$v_a$'],
#                index=[
#                        'Bartlett',
#                        'Hann',
#                        'Blackman',
#                        'Flat-Top'               
#                        ])
# HTML(df3B.to_html())


#%% Auxiliar
# plt.figure(2)
# #En veces
# plt.plot(ff[bfrec],(2*np.abs(senoFFT[bfrec , :])**2))
# plt.xlim([240,260])
# #En dB
# #plt.plot(ff[bfrec], 20*np.log10(2*np.abs(ft_signal[bfrec, :])**2))
# #plt.figure(2)
# fig, ax = plt.subplots(nrows=1, ncols=1)
# #ax.hist(estimador1)
# ax.hist(estimadores)
# mediana = np.median(estimadores, axis=0)
# sesgo = np.median(estimadores, axis=0) - amp
# varianza = np.mean((estimadores - mediana)**2, axis=0)





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