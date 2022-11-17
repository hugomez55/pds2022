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
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig




## Inicialización del Notebook del TP4

# import numpy as np
# from pandas import DataFrame
# from IPython.display import HTML
# from scipy import signal as sig

# Insertar aquí el código para inicializar tu notebook
########################################################

#%%  Inicialización de librerías
# Setup inline graphics: Esto lo hacemos para que el tamaño de la salida, 
# sea un poco más adecuada al tamaño del documento
mpl.rcParams['figure.figsize'] = (10,10)


#%% Esto tiene que ver con cuestiones de presentación de los gráficos,
# NO ES IMPORTANTE
fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16

plt.rcParams.update({'font.size':fig_font_size})
plt.rcParams.update({'font.family':fig_font_family})

def vertical_flaten(a):
    
    return a.reshape(a.shape[0],1)

##########################################
# Acá podés generar los gráficos pedidos #
##########################################

num = [1,1,1] 
den = [3,0,0]

#ww, hh = sig.freqz(np.array([1, 2, 3]), 1)
ww, hh = sig.freqz(num, den)
ww = ww / np.pi

plt.figure(1)

plt.plot(ww, 20 * np.log10(abs(hh)), label='ejemplo')

plt.title('FIR ejemplo')
plt.xlabel('Frecuencia normalizada [pi radianes]')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.figure(2)
angles = np.unwrap(np.angle(hh))
plt.plot(ww, angles, 'g')
plt.set_ylabel('Angle (radians)', color='g')
plt.xlabel('Frecuencia normalizada [pi radianes]')
plt.ylabel('Fase [radianes]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()





plt.show()
#%%
# Simular para los siguientes tamaños de señal
# N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=np.float64) #originalmente np.float

##########################################
# Acá podés generar los gráficos pedidos #
##########################################
#######################################
# Tu simulación que genere resultados #
#######################################

# tus_resultados_per = [ 
#                    ['2', '3'], # <-- acá debería haber numeritos :)
#                    ['3', '4'], # <-- acá debería haber numeritos :)
#                    ['a', 'd'], # <-- acá debería haber numeritos :)
#                    ['65', '3'], # <-- acá debería haber numeritos :)
#                    ['5', '4'], # <-- acá debería haber numeritos :)
#                    ['asd', 'g'], # <-- acá debería haber numeritos :)
#                    ['345', 'dsf'], # <-- acá debería haber numeritos :)
#                  ]
# df = DataFrame(tus_resultados_per, columns=['$s_P$', '$v_P$'],
#                index=N)
# HTML(df.to_html())



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

