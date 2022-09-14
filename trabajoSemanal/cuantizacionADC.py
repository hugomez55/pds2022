# -*- coding: utf-8 -*-
"""

@author: Hugo Alejandro Gomez

Descripción:
------------
Simulador de ADC

"""

import numpy as np
import matplotlib.pyplot as plt

#%%  Inicialización

# Inicialización de variables 
N=1000          #Cantidad de muestras
fs=1000         #Frecuencia de muestreo
ts=1/fs
df=fs/N

loadFactor=0.9  #Coeficiente de carga del ADC
# B=4             #Bits del ADC
Vf=2            #Tensión del ADC

plt.close('all')

#%%  Definición de funciones a utilizar

def funcionSeno (vmax, dc, ff, ph, nn, fs):
    
    # Grilla de sampleo temporal
    tt = np.arange(0.0, nn/fs, 1/fs)
    # Calculo de los valores punto a punto de la funcion
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc
    
    return tt,xx
    

def cuantizador (B, Vf, xxADC):
    
    q = (Vf*2) / ((2**B)-1)
    
    xxADCq = q* np.round(xxADC/q)
   
    error = xxADC - xxADCq
  
    return xxADCq, error

#%%Cálculos del ADC    


# Creamos nuestra señal senoidal

tt,xx = funcionSeno( vmax=1, dc=0, ff=10, ph=0, nn=N, fs=fs)

xxADC = xx*loadFactor   #Coeficiente de carga para limitar la entrada del ADC 

xxADC4,error4 = cuantizador(4, Vf, xxADC)
xxADC8,error8 = cuantizador(8, Vf, xxADC)
xxADC16,error16 = cuantizador(16, Vf, xxADC)


#%% Presentación gráfica de los resultados
#Preparación de las figuras 
#En rojo la señal senoidal
#En azul la señal cuantizada

#Señal cuantizada en 4 bits
plt.figure(1)
plt.plot(tt,xxADC,color='red')
plt.plot(tt,xxADC4,color='blue')
plt.title('Comparación con 4 bits')

#Señal cuantizada en 8 bits
plt.figure(2)
plt.plot(tt,xxADC,color='red')
plt.plot(tt,xxADC8,color='blue')
plt.title('Comparación con 8 bits')

#Señal cuantizada en 16 bits
plt.figure(3)
plt.plot(tt,xxADC,color='red')
plt.plot(tt,xxADC16,color='blue')
plt.title('Comparación con 16 bits')


#Distribución de los errores
plt.figure(4)
plt.title('Distribución del error con B = 4bits')
plt.hist(error4)

plt.figure(5)
plt.title('Distribución del error con B = 8bits')
plt.hist(error8)

plt.figure(6)
plt.title('Distribución del error con B = 16bits')
plt.hist(error16)

plt.show()



    
    
