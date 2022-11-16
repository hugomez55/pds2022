# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Testing

"""
# Inclusión de librerías
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import pandas
from pandas import DataFrame
from IPython.display import HTML

# Simular para los siguientes tamaños de señal
N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=np.float64) #originalmente np.float

##########################################
# Acá podés generar los gráficos pedidos #
##########################################
#######################################
# Tu simulación que genere resultados #
#######################################

tus_resultados_per = [ 
                   ['2', '3'], # <-- acá debería haber numeritos :)
                   ['3', '4'], # <-- acá debería haber numeritos :)
                   ['a', 'd'], # <-- acá debería haber numeritos :)
                   ['65', '3'], # <-- acá debería haber numeritos :)
                   ['5', '4'], # <-- acá debería haber numeritos :)
                   ['asd', 'g'], # <-- acá debería haber numeritos :)
                   ['345', 'dsf'], # <-- acá debería haber numeritos :)
                 ]
df = DataFrame(tus_resultados_per, columns=['$s_P$', '$v_P$'],
               index=N)
HTML(df.to_html())

tus_resultados = [ 
                   ['3', '3'], # <-- acá debería haber numeritos :)
                   ['43', '4'], # <-- acá debería haber numeritos :)
                   ['43', '23'], # <-- acá debería haber numeritos :)
                   ['2', 'as'], # <-- acá debería haber numeritos :)
                   ['3', 'as'] # <-- acá debería haber numeritos :)
                 ]
df = DataFrame(tus_resultados, columns=['$f_1$ (#)', '$W_2$ (dB)'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])
HTML(df.to_html())


# Visualización de la respuesta en frecuencia
# Así tenía la visualización
plt.figure(2)
plt.plot(ff[bfrec], 20* np.log10(2*np.abs(ft_winBartlett[bfrec])**2), 'red',label='Bartlett')
plt.plot(ff[bfrec], 20* np.log10(2*np.abs(ft_winHann[bfrec])**2), 'green',label='Hann')
plt.plot(ff[bfrec], 20* np.log10(2*np.abs(ft_winBlackman[bfrec])**2), 'blue',label='Blackman')
plt.plot(ff[bfrec], 20* np.log10(2*np.abs(ft_winFlatTop[bfrec])**2), 'black',label='Flat-Top')



# # Inicialización de variables
# plt.close('all')
# fs=1000
# N = 1000
# df = fs/N # resolución espectral
# ff = np.linspace(0, (N-1)*df, N)

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

# bfrec = ff <= fs/2

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

