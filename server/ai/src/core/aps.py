import numpy as np
import pandas as pd

class Aps():

	def __init__(self, listaAps):
		self.df = pd.read_csv(listaAps)

#Generar listado si los Aps estan en formato (0,2) todos juntos en el mismo str
	def listadoF1 (self,configs):
		lista_Aps=pd.DataFrame(columns=['codigo','Mac'])
		lista_Aps.at[0,'codigo'] = self.df.columns[1].split(',')[0].split('\\"')[5]
		lista_Aps.at[0,'Mac'] = self.df.columns[1].split(',')[0].split('\\"')[3]
		for i in range (len(self.df.columns[1].split(','))-2):
			aux = self.df.columns[1].split(',')
			if (aux[i+1].split('\\"')[1] == ''):
				aux1 = 0
			elif (aux[i+1].split('\\')[3] == '"'):  
				lista_Aps.at[i+1,'codigo'] = ','
				lista_Aps.at[i+1,'Mac'] = self.df.columns[1].split(',')[i+1].split('\\"')[1]
			else:    
				lista_Aps.at[i+1,'codigo'] = self.df.columns[1].split(',')[i+1].split('\\"')[3]
				lista_Aps.at[i+1,'Mac'] = self.df.columns[1].split(',')[i+1].split('\\"')[1]
		return lista_Aps

#Generar listado si los Aps estan en formato - cada MAC en una columna 
	def listadoF2 (self, configs):
		lista_Aps=pd.DataFrame(columns=['codigo','Mac'])
		lista_Aps.at[0,'codigo'] = self.df.columns[0].split('\\"')[5]
		lista_Aps.at[0,'Mac'] = self.df.columns[0].split('\\"')[3]
		for i in range (self.df.shape[1]-2):
			if (self.df.columns[i+1].split('\\"')[1] == ''):
				aux1 = 0
			elif (self.df.columns[i+1].split('\\')[3] == '"'):  
				lista_Aps.at[i+1,'codigo'] = ','
				lista_Aps.at[i+1,'Mac'] = self.df.columns[i+1].split('\\"')[1]
			else:    
				lista_Aps.at[i+1,'codigo'] = self.df.columns[i+1].split('\\"')[3]
				lista_Aps.at[i+1,'Mac'] = self.df.columns[i+1].split('\\"')[1]
		return lista_Aps
    
#Función auxiliar que nos dice si la MAC de antel esta en la lista de Aps
	def Aps_antel (self, lista_Aps, listaAntel):
		Antel_df = pd.read_csv(listaAntel)
		lista = []
		for i in range(lista_Aps.shape[0]):
			for j in range (Antel_df.shape[0]):
				if (Antel_df['AP;MAC Radio;Mac AP'][j].split(';')[1] == lista_Aps.iloc[i,1]):
					lista.append(lista_Aps.iloc[i,0])                  
		return lista
    
#Función auxiliar que nos dice si la mac de fing esta en la lista de Aps
	def Aps_fing (self, lista_Aps, listaFing):
		Fing = pd.read_fwf(listaFing)
		lista = []
		for i in range(lista_Aps.shape[0]):
			for j in range (Fing.shape[0]):
				if (Fing.iloc[j,0].split('\t')[0] == lista_Aps.iloc[i,1]):
					lista.append(lista_Aps.iloc[i,0])                  
		return lista
    
