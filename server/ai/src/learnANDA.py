#!/usr/bin/python3

import io
import csv
import json
import warnings
import pickle
import operator
import time
import logging
import math
import functools
import numpy
from sklearn.preprocessing import MinMaxScaler
from threading import Thread
from random import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from s3_helper import put_file, get_file

#Librerias locindoor
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from core.data_processor import DataLoader
from core.model import Model
from core.trajectories import Trajectories
from core.aps import Aps

# create logger with 'spam_application'
logger = logging.getLogger('learn')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('learn.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - [%(name)s/%(funcName)s] - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (
                func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco
    
class AI(object):

    def __init__(self, family=None):
        self.logger = logging.getLogger('learn.AI')
        self.naming = {'from': {}, 'to': {}}
        self.family = 'posifi'
        

    def classify(self, sensor_data):
        self.logger.debug(sensor_data)
        header = self.header[1:]
        
        is_unknown = True
        lista_Ap = pd.read_csv("Listado_Aps.csv")
        step = len(sensor_data)
        #Filra los Ap detectados que no están en los Aps de entrenamiento
        csv_data = numpy.zeros((step,len(header)))
        for huella in range(len(sensor_data)):
            for sensorType in sensor_data[huella]["s"]:
                for sensor in (sensor_data[huella]["s"][sensorType]):
                    sensorName = 'Ap200'
                    for j in range (len(lista_Ap)):
                        if (lista_Ap['MAC'][j] == sensor):
                            sensorName = lista_Ap['nºAp'][j]
                        if sensorName in header:
                            is_unknown = False
                            csv_data[huella][header.index(sensorName)] = sensor_data[huella]["s"][sensorType][sensor]
                                                                                                                                                                                                                                                                                                                 
        self.headerClassify = header
        #self.csv_dataClassify = csv_data.reshape(1, -1)
        self.csv_dataClassify = csv_data
        payload = {'location_names': self.naming['to'], 'predictions': []}

        threads = [None]*len(self.algorithms)
        self.results = [None]*len(self.algorithms)

        for i, alg in enumerate(self.algorithms.keys()):
            threads[i] = Thread(target=self.do_classification, args=(i, alg,step))
            threads[i].start()

        for i, _ in enumerate(self.algorithms.keys()):
            threads[i].join()

        for result in self.results:
            if result != None:
                payload['predictions'].append(result)
        payload['is_unknown'] = is_unknown
        return payload

    def do_classification(self, index, name,step):
        t = time.time()
        pasos = np.empty([step,self.csv_dataClassify.shape[1]])
        try:
            if name == "LSTM":
                
                for h in range(self.csv_dataClassify.shape[0]):
                    self.csv_dataClassify[h][self.csv_dataClassify[h] == 0] = -100
                    huella = self.csv_dataClassify[h]
                    huella = huella.reshape(len(self.header)-1,1)
                    min_max_scaler = MinMaxScaler()
                    x_scaled = min_max_scaler.fit_transform(huella)
                    huella = x_scaled.reshape(1,len(self.header)-1)
                    pasos[h] = huella
                
                if (step == 15):
                    pasos = pasos.reshape(1,step,len(self.header)-1)
                    model_new= load_model('learnfull.h5')
                    prediction = model_new.predict(pasos)
                else:
                    pasos2 = np.empty([15,self.csv_dataClassify.shape[1]])
                    for i in range(15):
                        pasos2[i] = huella
                    pasos2 = pasos2.reshape(1,15,self.csv_dataClassify.shape[1])
                    model_new= load_model('learnfull.h5')
                    prediction = model_new.predict(pasos2)
           
            else:
                prediction = self.algorithms[name].predict_proba(self.csv_dataClassify)
        except Exception as e:
            logger.error(self.csv_dataClassify)
            logger.error(str(e))
            return
        
        predict = {}
        if name == "LSTM":
            a = np.int(prediction[0][step-1])
            prediction = np.zeros([1,90])
            for i in range(90):
                if (a == i):
                    prediction[0,i] = 100
        
        for i, pred in enumerate(prediction[0]):
            predict[i] = pred
        predict_payload = {'name': name,'locations': [], 'probabilities': []}
        badValue = False
        
        for tup in sorted(predict.items(), key=operator.itemgetter(1), reverse=True):
            predict_payload['locations'].append(str(tup[0]))
            predict_payload['probabilities'].append(round(float(tup[1]), 2))
            if math.isnan(tup[1]):
                badValue = True
                break
        if badValue:
            return
     
        self.results[index] = predict_payload

    @timeout(10)
    def train(self, clf, x, y):
        return clf.fit(x, y)
    
    def trayecto(self, fname): #Generar los trayectos
        
        configs = json.load(open('config.json', 'r'))
        
        #Clase Ap - lee el archivo de MAC de los Aps y se genera un listado
        Ap = Aps(os.path.join('data', configs['Aps']['listaAps']))
        lista_Aps = Ap.listadoF2(configs)
        #Litar Aps de Antel
        lista_antel = Ap.Aps_antel(lista_Aps,os.path.join('data',configs['Aps']['listaAntel']))
        #Listar Aps de Fing
        lista_Fing = Ap.Aps_fing(lista_Aps,os.path.join('data',configs['Aps']['listaFing']))
        
        # Se cargan los datos y se procesan 
        data = DataLoader(fname)
        # en datos se cargan la matriz de huellas recolectadas - según el tipo de matriz que queramos - se especifica en config
        datos = data.huellas(configs, lista_antel, lista_Aps)
        #Filtrado de columnas con poca información 
        datos = data.filtro_Ap(datos, configs['Aps']['descartamos'],-85)
        # cargamos los datos con una huella aleatoria por zona 
        datos_una_huella = data.una_huella_zona(datos)
        #Normalizamos los datos RSSI (Las zonas quedan con nùmero de zona)
        huellas_norm_df = data.normaliza(datos_una_huella)
        datos.to_csv('datos.csv', index=False)
        # Se genera Trayectorias_aleatorias que es un matriz que cada fila corresponde a una trayectoria de T pasos 
        # La cantidad de trayectorias que queremos generar y los pasos se pasan como parametro
        trayectorias = Trajectories(configs)
        mapa = trayectorias.crear_mapa(configs)
        Trayectorias_aleatorias = trayectorias.generacion_trayectorias(configs['trajectory']['T'],configs['trajectory']['cantidad'],mapa)

        #Se genera una matriz de 3D donde a cada paso de la trayectoria le corresponde una función de huellas
        Matriz_Trayectorias_una_huella = trayectorias.trayectorias_con_una_huella(huellas_norm_df,Trayectorias_aleatorias)
        
        data_train, data_test = trayectorias.train_and_test(Matriz_Trayectorias_una_huella,configs['data']['train_test_split'])
        train3D_X, train3D_y = data_train[:,:,:-1], data_train[:,:, -1]
        test3D_X, test3D_y = data_test[:,:,:-1], data_test[:,:, -1]
        #trainY_coord3D = data.coordenadas(train3D_y)
        #testY_coord3D = data.coordenadas(test3D_y)
        
        return  train3D_X, test3D_X, train3D_y, test3D_y


    def learn(self, fname):
        self.model = Model()
        t = time.time()
        configs = json.load(open('config.json', 'r'))
        #Cargo el archivo que contiene las huellas para clasificar
        fname = "../testing/HuellasPiso1.csv"
        #genero las trayectoiras y lo separo en train y test
        #train3D_X, test3D_X, train3D_y, test3Y_y = self.trayecto(fname)
        self.header = []
        rows = []
        naming_num = 0
        with open('../src/datos.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                #self.logger.debug(row)
                if i == 0:
                    self.header = row
                else:
                    for j, val in enumerate(row):
                        if j == len(row)-1:
                            # this is a name of the location
                            if val not in self.naming['from']:
                                self.naming['from'][val] = naming_num
                                valor = str(int(float(val)))
                                #self.naming['to'][naming_num] = "location" + "_" + valor
                                self.naming['to'][naming_num] = valor
                                naming_num += 1
                            row[j] = self.naming['from'][val]
                            continue
                        if val == '':
                            row[j] = 0
                            continue
                        try:
                            row[j] = float(val)
                        except:
                            self.logger.error(
                                "problem parsing value " + str(val))
                    rows.append(row)
        
        # first column in row is the classification, Y
        y = numpy.zeros(len(rows))
        x = numpy.zeros((len(rows), len(rows[0]) - 1))

        # shuffle it up for training
        record_range = list(range(len(rows)))
        shuffle(record_range)
        for i in record_range:
            y[i] = rows[i][0]
            x[i, :] = numpy.array(rows[i][1:])

        names = [
            "LSTM"]
            #"Linear SVM"]
        classifiers = [
            self.model.model_clas(configs)]
            #SVC(kernel="linear", C=0.025, probability=True)]
       
        self.algorithms = {}
        
        for name, clf in zip(names, classifiers):
            t2 = time.time()
            self.logger.debug("learning {}".format(name))
            try:
                if name == "LSTM":
                    var = 0
                    #self.algorithms[name] = self.model.train(train3D_X, train3D_y,epochs = 5,batch_size = 10,verbose=2,shuffle=True)
                    #self.model.save()
                    self.algorithms[name] = 'LSTM'
                    
                else:
                    self.algorithms[name] = self.train(clf, x, y)
              
               # self.logger.debug("learned {}, {:d} ms".format(name, int(1000 * (t2 - time.time()))))
            except Exception as e:
                self.logger.error("{} {}".format(name, str(e)))
  
   
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def save(self, save_file):
        t = time.time()
        save_data = {
            'header': self.header,
            'naming': self.naming,
            'algorithms': self.algorithms,
            'family': self.family
        }
        
        save_data = pickle.dumps(save_data)
        put_file(f'ai_metadata/{save_file}', save_data)
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def load(self, save_file):
        t = time.time()
        downloaded_data = get_file(f'ai_metadata/{save_file}')
        if not downloaded_data:
            raise Exception('There is no AI data on S3')
        saved_data = pickle.loads(downloaded_data)
        self.header = saved_data['header']
        self.naming = saved_data['naming']
        self.algorithms = saved_data['algorithms']
        self.family = saved_data['family']
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))
        
def do():
    ai = AI()
    ai.load()
    # ai.learn()
    params = {'quantile': .3,
              'eps': .3,
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': 3}
    bandwidth = cluster.estimate_bandwidth(ai.x, quantile=params['quantile'])
    connectivity = kneighbors_graph(
        ai.x, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')
    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            try:
                algorithm.fit(ai.x)
            except Exception as e:
                continue

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(numpy.int)
        else:
            y_pred = algorithm.predict(ai.x)
        if max(y_pred) > 3:
            continue
        known_groups = {}
        for i, group in enumerate(ai.y):
            group = int(group)
            if group not in known_groups:
                known_groups[group] = []
            known_groups[group].append(i)
        guessed_groups = {}
        for i, group in enumerate(y_pred):
            if group not in guessed_groups:
                guessed_groups[group] = []
            guessed_groups[group].append(i)
        for k in known_groups:
            for g in guessed_groups:
                print(
                    k, g, len(set(known_groups[k]).intersection(guessed_groups[g])))
                    
       
