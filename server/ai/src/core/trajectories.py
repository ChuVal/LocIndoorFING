
import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from itertools import accumulate
from sklearn.preprocessing import normalize
from random import randint, uniform,random

class Trajectories():
	"""Clase - Creación de trayectorias"""
	def __init__(self,configs):
		self.start = None
        
	def crear_mapa (self,configs):   
		Vmax = configs['trajectory']['vmax']
		Dmax = configs['trajectory']['dmax']
		DeltaT = configs['trajectory']['delta_T']
		MapaD = self.Mapa_Distancias(configs)
		MapaP = self.Mapa_Probabilidades(MapaD,Vmax,Dmax,DeltaT)
		MapaCDF = self.Mapa_CDF(MapaP)
		return MapaCDF

	def dist(self, loc1, loc2):
		(x1,y1) = loc1[0], loc1[1]
		(x2,y2) = loc2[0], loc2[1]
		loss = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
		return loss

	def Max_dist_puntos(self, lista):
		max = 0
		for i in range (lista.shape[0]):
			for j in range (lista.shape[1]-1):
				aux = self.dist(lista[i,j],lista[i,j+1])
				if (max < aux):
					max = aux
		return max

	def coord (self, loc,ZonasFila,ZonasColumna,x):
		coordenada = ()
		if (loc < ZonasFila):
			coordenada = (1 + loc*2,1)
		elif (loc < (ZonasFila*2)):
			coordenada = (1 + (loc-ZonasFila)*2, 3)
		elif (loc < (ZonasFila*3)):
			coordenada = ( 1 + (loc-(ZonasFila*2))*2,5)
		elif (loc < (ZonasFila*4)):
			coordenada = (1 + (loc-(ZonasFila*3))*2,7)
		elif (loc < (ZonasFila*5)):
			coordenada = (1 + (loc-(ZonasFila*4))*2, 9)
		return coordenada
    
#Se crea un mapa de distancias euclideaneas de las PR
	def Mapa_Distancias(self, configs):
		CantZonas = configs['trajectory']['PR']
		ZonasFila = configs['trajectory']['PR_Fila']
		ZonasColumna = configs['trajectory']['PR_Columna']
		x = configs['trajectory']['Distancia']
		MapaD_df = pd.DataFrame()        
		for i in range(CantZonas):    
			loc ='loc'+ str(i)
			MapaD_df[loc] = 1
			MapaD_df.at[i,loc] = 0    

		for j in range(CantZonas):
			for k in range(CantZonas):        
				MapaD_df.iat[j,k] = self.dist(self.coord(j,ZonasFila,ZonasColumna,x),self.coord(k,ZonasFila,ZonasColumna,x))

		MapaD = MapaD_df.values.astype('float32')
		#MapaD_df.to_csv('MapaD.csv',index=False)
		return MapaD_df

# Función que aplica la formula de probabilidad
	def Probabilidad (self, loc, vmax,dmax,deltaT):
		omega = vmax*deltaT
		aux = 2*(omega**2)
		a = 1/((aux)*(1 - np.exp((dmax**2)/aux)))
		b = np.exp(-((loc**2)/aux))       
		return (abs(b*a))
    
#Función que crea el Mapa de Probabilidades
	def Mapa_Probabilidades (self, MapaD_df,vmax,dmax,deltaT):
		filas, columnas = MapaD_df.shape
		MapaP_df = pd.DataFrame(index=MapaD_df.index,columns=MapaD_df.columns)
		MapaD = MapaD_df.values.astype('float32')
		for i in range(filas):    
			for j in range(columnas):           
				MapaP_df.iat[i,j] = self.Probabilidad(MapaD[i,j],vmax,dmax,deltaT) 
		#MapaP_df.to_csv('MapaP.csv',index=False)
		MapaP_norm = normalize((MapaP_df), axis=1, norm='l1')
		MapaP_norm_df = pd.DataFrame(MapaP_norm, index=MapaP_df.index, columns=MapaP_df.index)
		return MapaP_norm_df

#Función que crea el Mapa de suma acumulativa
	def Mapa_CDF (self, MapaP):
		filas, columnas = MapaP.shape
		MapaCDF = np.array(MapaP)
		for i in range(filas):
			MapaCDF[i,:]=list(accumulate(MapaCDF[i,:]))

		MapaCDF_df = pd.DataFrame(MapaCDF, index=MapaP.index, columns=MapaP.index)        
		MapaCDF_norma = normalize((MapaCDF_df), axis=1, norm='max')
		#MapaCDF_df.to_csv('MapaCDF.csv',index=False)
		return MapaCDF_norma
    
#Función que busca en el array el valor más parecido a value y devuelve el indice
	def getnearpos(self,array,value):
		idx = (np.abs(array-value)).argmin()
		return idx

#Función que genera cantidad trayectorias aleatorias de tamaño T 
	def generacion_trayectorias(self,T,cantidad,MapaCDF):
		rows = []
		columns = []

		for i in range(T):    
			loc = 'Timestep' + str(i)
			col = loc
			columns.append(col)
    
		for j in range(cantidad):
			row = []
			for k in range(T):
				vacio = 0
				row.append(vacio)
			rows.append(row)

		Trayectorias = pd.DataFrame(rows, columns=columns)

		for fila in range(Trayectorias.shape[0]):
			L0 = randint(0,(MapaCDF.shape[0]-1))
			Trayectorias.iat[fila, 0] = L0
			pos_ant = L0
			for pos in range(T-1):
				R = random()       
				indice = self.getnearpos(MapaCDF[pos_ant,:], R)
				Trayectorias.iat[fila, pos+1] = indice
				pos_ant = indice
                        
		return Trayectorias
    
	def lista(self,pos,datos):    
		for i in range(datos.shape[0]):
			if pos == datos['zona'][i]:
				valor = datos.iloc[i,:]
		return valor 
    
	def trayectorias_con_una_huella (self,huellas,trayectorias):
		tra = trayectorias.values.astype('float32')
		Trayectorias_huellas = np.empty([tra.shape[0],tra.shape[1],huellas.shape[1]])

		for fila in range(tra.shape[0]):
			for columna in range(tra.shape[1]):
				huella = self.lista(tra[fila,columna],huellas)
				for profundo in range(huellas.shape[1]):
					Trayectorias_huellas[fila][columna][profundo] = huella[profundo]
		return Trayectorias_huellas
    
	def train_and_test (self, trayectorias, split):
		i_split = int((trayectorias.shape[0]) * split)
		data_train = trayectorias[:i_split]
		data_test  = trayectorias[i_split:]
		return data_train, data_test    
    
	def matriz_huellas (self, pos, datos):
		huellas = pd.DataFrame()
		cont = 0  
		for j in range(datos.shape[0]):        
			if (datos['zona'][j] == pos):
				cont = cont+1
				huellas = huellas.append(datos.iloc[j], ignore_index=True)
		return huellas

	def trayectorias_con_varias_huellas (self,huellas,trayectorias,cant_huellas):
		tra = trayectorias.values.astype('float32')
		Trayectorias_huellas = np.empty([tra.shape[0],tra.shape[1],cant_huellas,huellas.shape[1]])
		for fila in range(Trayectorias_huellas.shape[0]):
			for columna in range(tra.shape[1]):
				huella = self.matriz_huellas((tra[fila,columna]),huellas)
				Trayectorias_huellas[fila][columna] = huella
		return Trayectorias_huellas
    