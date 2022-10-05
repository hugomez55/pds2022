# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Simulador de ADC

En esta tarea semanal retomamos la consigna de la tarea anterior, donde simulamos el bloque de cuantización de un ADC de B bits 
en un rango de  ±VF Volts. Ahora vamos a completar la simulación del ADC incluyendo la capacidad de muestrear a fs Hertz.
Para ello se simulará el comportamiento del dispositivo al digitalizar una senoidal contaminada con un nivel predeterminado de ruido. 
Comenzaremos describiendo los parámetros a ajustar de la senoidal:

    frecuencia f0 arbitraria, por ejemplo f0=fS/N=Δf  
    energía normalizada, es decir energía (o varianza) unitaria

Con respecto a los parámetros de la secuencia de ruido, diremos que:

    será de carácter aditivo, es decir la señal que entra al ADC será sR=s+n
    Siendo n la secuencia que simula la interferencia, y s la senoidal descrita anteriormente.
    La potencia del ruido será Pn=kn.Pq W siendo el factor k una escala para la potencia del ruido de cuantización Pq=q^2/12.
    finalmente, n será incorrelado y Gaussiano.

El ADC que deseamos simular trabajará a una frecuencia de muestreo fS=1000 Hz y tendrá un rango analógico de ±VF=2 Volts.

Se pide:

a) Generar el siguiente resultado producto de la experimentación. B = 4 bits, kn=1
b) Analizar para una de las siguientes configuraciones B = ̣{4, 8 y 16} bits, kn={1/10,1,10}. 
   Discutir los resultados respecto a lo obtenido en a).



"""
#%%  Librerias
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%%  Inicialización

# Inicialización de variables 
# N=1000          #Cantidad de muestras
# fs=1000         #Frecuencia de muestreo
# ts=1/fs
# df=fs/N

# loadFactor = 0.9  #Coeficiente de carga del ADC
# B = (4,8,16)             #Bits del ADC
# Vf = 2            #Tensión del ADC

# plt.close('all')

#%%  Definición de funciones a utilizar

def funcionSeno (vmax, dc, ff, ph, nn, fs):
    
    # Grilla de sampleo temporal
    #tt = np.arange(0.0, nn/fs, 1/fs)
    tt = np.linspace(0, (nn-1)*(1/fs), nn)
    # Calculo de los valores punto a punto de la funcion
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc
    
    #return tt,xx
    return xx

# def cuantizador (B, Vf, xxADC):
    
#     q = (Vf*2) / ((2**B)-1)
    
#     xxADCq = q * np.round(xxADC/q)
   
#     error = xxADC - xxADCq
  
#     return xxADCq, error

def funcionDFT(xx):
    N=len(xx)
       
    n=np.arange(N)
    
    #casteo a complejo
    X=np.zeros(N)*(1j)
    
    for k in range (N):
            
        twiddle = np.exp(-2j*np.pi*k*n/N)      
        
        X[k] = np.dot(twiddle,xx)
        
    return X


#%%  Script de referencia para generación de gráficos
# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

T = N/fs

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = N*over_sampling
 
# Datos del ADC
B = 4 # bits
Vf = 2 # Volts
q = Vf/2**B # Volts
 
# datos del ruido
kn = 10
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)
 
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral
 
#######################################################################################################################
#%% Acá arranca la simulación
 
# ....

#Grillas temporales

tt = np.linspace(0, (N-1)*ts, N)
tt_os = np.linspace(0, (N-1)*ts, N_os)

#Grillas frecuenciales
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N-1)*df, N_os)

analog_sig = funcionSeno( vmax=1, dc=0, ff=1, ph=0, nn=N_os, fs=fs)

nn = np.random.normal(0, np.sqrt(pot_ruido), size=N_os)

#in ADC
sr = analog_sig + nn

#Muestreo la señal analógica 1 cada OS muestras
sr = sr[::over_sampling]

#out ADC cuantizo la señal muestreada
srq = q * np.round(sr/q)

# ruido de cuantización
nq = srq - sr

#Transformadas de Fourier
ft_Srq = funcionDFT(srq)
ft_As = funcionDFT(analog_sig)
ft_SR = funcionDFT(sr)
ft_Nn = funcionDFT(nn)
ft_Nq = funcionDFT(nq)

Nnq_mean = np.mean(nq)
nNn_mean = np.mean(nn)

#######################################################################################################################
#%% Presentación gráfica de los resultados
plt.close('all')
 
plt.figure(1)
plt.plot(tt, srq, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(tt, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
 
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()
 
 
plt.figure(2)
bfrec = ff <= fs/2
 
nNn_mean = np.mean(np.abs(ft_Nn)**2)
Nnq_mean = np.mean(np.abs(ft_Nq)**2)
 
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
 
 
plt.figure(3)
bins = 10
plt.hist(nq, bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))