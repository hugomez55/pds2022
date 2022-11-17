#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:29:36 2022

Title: TS9: Estimación espectral a prueba: Ancho de banda del Electrocardiograma

@author: hugomez
"""
#%% Inclusión de librerías

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio 
import scipy.signal as sig

#%% Variables a tener en cuenta
# ecg_lead: Registro de ECG muestreado a $fs=1$ KHz durante una prueba de esfuerzo
# qrs_pattern1: Complejo de ondas QRS normal
# heartbeat_pattern1: Latido normal
# heartbeat_pattern2: Latido de origen ventricular
# qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren los latidos
   
#%% Variables globales
fs = 1000
porcPotencia = 0.99
bloque = 1024 #760

#%% Comienzo
plt.clf()
mat_struct = sio.loadmat('ECG_TP4.mat')
# sio.whosmat('ECG_TP4.mat') # Me indica las variables que hay en el archivo

ecg_one_lead = mat_struct['ecg_lead'] 
ecg_one_lead = ecg_one_lead.flatten()
cant_muestras = len(ecg_one_lead)
# plt.plot(ecg_one_lead)

signalECG = ecg_one_lead[0:100000]
# signalECG = signalECG.astype(np.float64)
# signalECG -= np.mean(signalECG)

# Espectro de la ECG en reposo
#[f, Pxx_reposo] = sig.welch(signalECG, fs=fs, nperseg=bloque, average='median', axis=0)
[f, Pxx_reposo] = sig.welch(signalECG, fs=fs, nperseg=bloque, axis=0)
# plt.figure(1)
# plt.plot(f, 10*np.log10(2*np.abs(Pxx_den)))

energiaAcumulada = np.cumsum(Pxx_reposo)
index_energia = np.where(np.cumsum(Pxx_reposo)/energiaAcumulada[-1] > porcPotencia)[0]
Wcorte_reposo = f[index_energia[0]]

# En ejercicio
signalECG = ecg_one_lead[450000:550000]
#[f, Pxx_ejercicio] = sig.welch(signalECG, fs=fs, nperseg=bloque, average='median', axis=0)
[f, Pxx_ejercicio] = sig.welch(signalECG, fs=fs, nperseg=bloque, axis=0)

# Esfuerzo
signalECG = ecg_one_lead[750000:850000]
#[f, Pxx_esfuerzo] = sig.welch(signalECG, fs=fs, nperseg=bloque, average='median', axis=0)
[f, Pxx_esfuerzo] = sig.welch(signalECG, fs=fs, nperseg=bloque, axis=0)

plt.figure(1)
# Decibeles - Ancho de banda del ruido
# plt.plot(f, 10*np.log10(Pxx_reposo), label='Reposo')
# plt.plot(f, 10*np.log10(Pxx_ejercicio), label='Ejercicio')
# plt.plot(f, 10*np.log10(Pxx_esfuerzo), label='Esfuerzo')
# plt.legend()
# plt.grid()

# Densidad en veces - Ancho de banda del ECG
plt.plot(f, (Pxx_reposo), label='Reposo')
plt.plot(f, (Pxx_ejercicio), label='Ejercicio')
plt.plot(f, (Pxx_esfuerzo), label='Esfuerzo')
plt.legend()
plt.grid()

# Análisis de latidos de ancho fijo
# Lugares en el tiempo donde detecta los latidos
qrs_detections = mat_struct['qrs_detections']
inf = 250
sup = 350
latido = (ecg_one_lead[int(qrs_detections[0] - inf):int(qrs_detection[0] + sup)])

muestras = np.arange(len(qrs_detections))
i = 0
latidos = np.zeros([sup+inf, qrs_detections.shape[0]])

#for nn in muestras
#    latidos[:,nn] = (ecg_one_lead[int(qrs_detections[nn] - inf):int(qrs_detection[nn] + sup)])
#    latidos[:,nn] -= np.mean(latidos[:,nn])

estimadorAmplitud = latidos [242, :]

filtroNormal = estimadorAmplitud < 11500

#indexVentricular = np.where(estimadorAmplitud > 11500)[0]
#ventricular = latidos[:,indexVentricular]

ventricular = latidos[:,np.bitwise_not(filtroNormal)]
normal = latidos[:,filtroNormal]

plt.plot(ventricular, 'b', label = 'Ventricular')
plt.plot(normal, 'g', label = 'Normal')

plt.plot(np.mean(ventricular, axis=1), 'b', label = 'Ventricular')
plt.plot(np.mean(normal, axis=1), 'g', label = 'Normal')


plt.figure(1)
plt.plot(latido)
plt.grid()


plt.figure(2)
plt.hist(estimadorAmplitud)






# plt.semilogy(f, Pxx_den)
# plt.ylim([0.5e-3, 1])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()

#%%
# Definir plantilla del filtro
# fs0 = # fin de la banda de detención 0
# fc0 = # comienzo de la banda de paso
# fc1 = # fin de la banda de paso
# fs1 = # comienzo de la banda de detención 1 
