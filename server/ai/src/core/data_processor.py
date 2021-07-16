import math
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class DataLoader():
 
    def __init__(self, filename):
        self.datos = pd.read_csv(filename)     

    def huellas (self, configs, antel, lista_aps):
        
        nan = configs['data']['dato_faltante']
        huellas_zona = configs['data']['huellas_zona']
        
        if configs['data']['tipo_data'] == 'brutos':
                huellas = self.column_separation(self.datos)
        if configs['data']['tipo_data'] == 'data_faltante':
                huellas = self.missing_data(nan,self.datos)
        if configs['data']['tipo_data'] == 'completos':
                huellas = self.missing_data(nan,self.datos)
                huellas = self.column_order (huellas, lista_aps)
        if configs['data']['tipo_data'] == 'truncado':
                huellas = self.missing_data(nan,self.datos)
                huellas = self.column_order (huellas, lista_aps)
                huellas = self.truncate (huellas,huellas_zona)
        if configs['data']['tipo_data'] == 'Aps_antel':
                huellas = self.missing_data(nan,self.datos)
                huellas = self.solo_Antel(huellas,antel)
                huellas = self.truncate (huellas,huellas_zona)
        
        return huellas
    
    def column_separation (self, data):  
        filas = data.shape[0]
        huellas = pd.DataFrame()
        aux = (data['wifi'][0].split(','))
        huellas[aux[0].split(':')[0].replace('"', '').strip()] = np.nan

        for i in range(filas):  
            aux = (data['wifi'][i].split(','))
            columnas = len(aux)
            j = 0
            for j in range(columnas):        
                if (aux[j].split(':')[0] == '"'):           
                    letra = aux[j].split(':')[0]
                else:
                    letra = aux[j].split(':')[0].replace('"', '').strip() 
                    nueva=0
                    for k in range (huellas.shape[1]): 
                        if letra == huellas.columns[k]:                
                            nueva = 1        
                    if nueva == 0:
                        nomColumna = aux[j].split(':')[0].replace('"', '').strip()
                        huellas[nomColumna] = np.nan
                    huellas.at[i, letra] = aux[j].split(':')[1].replace('"', '').strip()
        
        
        huellas['time'] = data['timestamp'].astype('float32') 
        huellas['zona'] = data['name']     
        return huellas
    
#Función que sustituye los Nan por valor    
    def missing_data (self,valor,data):
        
        huellas = self.column_separation(data)
        huellas = huellas.fillna(valor)
        return huellas
    
    def Mac (self, codigo, lista):
        mac='nan'
        for i in range(lista.shape[0]):
            if (codigo == lista.iloc[i,0]):
                mac = lista.iloc[i,1]
        return mac
    
    def column_order (self,data,lista_Aps):
        
        listado_Aps = pd.DataFrame(columns=['nºAp','codigo','MAC'])
        #Descartamos la columna time 
        huellassinT = data.drop(['time'], axis=1)
        Datos_Huellas = huellassinT
        #Huellas con la zona en entero
        Datos_Huellas['zona'] = huellassinT['zona'].str.split('loc').str[1].astype('float32') 
        Huellas_Ap = Datos_Huellas.astype('float32')
        for i in range(Datos_Huellas.shape[1]-1):
            columna = Datos_Huellas.columns[i]
            Ap = 'Ap' + str(i)
            Huellas_Ap = Huellas_Ap.rename(columns={columna: Ap})
            listado_Aps.at[i,'nºAp']=Ap
            listado_Aps.at[i,'codigo']=Datos_Huellas.columns[i]
            listado_Aps.at[i,'MAC']=self.Mac(Datos_Huellas.columns[i], lista_Aps)
            

        Huellas_Ap.to_csv('Huellas.csv', index=False)
        listado_Aps.to_csv('Listado_Aps.csv', index=False)
        return Huellas_Ap
    
    def truncate (self, data, valor):
        huellas = pd.DataFrame()
        for i in range(90):
            cont = 0  
            for j in range(data.shape[0]):        
                if (data['zona'][j] == i):
                    if (cont < valor):
                        cont = cont+1
                        huellas = huellas.append(data.iloc[j], ignore_index=True)
        return huellas
    
    
    #def una_huella_zona (self, data):
        #data1 = pd.DataFrame()
        #for i in range(90):
            #ran = randint(0, 5)
            #aux = (i*6) + ran
            #data1 = data1.append(data.iloc[aux], ignore_index=True)    
        #return data1
    
    def una_huella_zona (self, data):
        row = []
        for k in range(90):
            row.append(k)
        data1 = pd.DataFrame(columns=data.columns, index=row)
        loc = 0
        for i in range(0,data.shape[0],6):
            for j in range(data.shape[1]):        
                data1.iat[loc,j] = data.iloc[i:i+5,j].mean()
            loc = loc+1
        
        return data1
    
    def normaliza (self, data):
    
        zona = data.iloc[:, -1]
        norm = data.drop(['zona'], axis=1)
        x = norm.values
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = DataFrame(x_scaled, columns=data.columns[0:-1])
        df['zona']=zona
    
        return df

    def coord (self, loc):
        coordenada = ()
        if (loc < 18):
            coordenada = (1 + loc*2,1)
        elif (loc < 36):
            coordenada = (1 + (loc-18)*2, 3)
        elif (loc < 54):
            coordenada = ( 1 + (loc-(18*2))*2,5)
        elif (loc < 72):
            coordenada = (1 + (loc-(18*3))*2,7)
        elif (loc < 90):
            coordenada = (1 + (loc-(18*4))*2, 9)
        return coordenada

    def coordenadas (self, train_y):
        trainY_coord = np.empty([train_y.shape[0],train_y.shape[1],2])
        for i in range(train_y.shape[0]):
            for j in range(train_y.shape[1]):
                trainY_coord[i,j,0] = self.coord(train_y[i,j])[0]
                trainY_coord[i,j,1] = self.coord(train_y[i,j])[1]
        return trainY_coord
    
    def solo_Antel (self, datos, lista_codigos):
        data = pd.DataFrame()
        for i in range (len(lista_codigos)):
            for j in range (datos.shape[1]):
                if (datos.columns[j] == lista_codigos[i]):
                    data = data.append([datos[datos.columns[j]]])
        data = data.T
        data['zona'] = datos['zona'].str.split('loc').str[1].astype('float32')
        #data.to_csv('Huellas_Antel.csv', index=False)
        return data
    
    def filtro_Ap_pocaInfo (self, datos, cant_descarto,nan):

        datos_filtro=[]
        for i in range (datos.shape[1]):
            valores = (datos[datos.columns[i]].value_counts())  
            if nan in valores:
                if (valores[nan] > (datos.shape[0] - cant_descarto)):
                    datos_filtro.append(datos.columns[i])
        for j in datos_filtro:
            datos = datos.drop([j], axis=1)  
        return datos
   
    def filtro_Ap (self, datos,cant_descarto,valor):

        datos_filtro=[]
        for i in range (datos.shape[1]):
            valores = (datos[datos.columns[i]].value_counts())
            valores = valores.reset_index(name='cant')
            cont= 0
            for k in range(valores.shape[0]):
                if (valores['index'][k] < valor):
                    cont = cont + valores['cant'][k] 

            if (cont > (datos.shape[0] - cant_descarto)):
                datos_filtro.append(datos.columns[i])

        for j in datos_filtro:
            datos = datos.drop([j], axis=1)  
        
        return datos
