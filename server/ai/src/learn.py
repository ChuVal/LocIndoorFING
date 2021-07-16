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

        datos = pd.read_csv('TrainM11.csv')
        header = list(datos.columns[0:67])

        is_unknown = True
        lista_Ap = pd.read_csv("Listado_Aps12000.csv")
        step = len(sensor_data)

        csv_data = numpy.zeros((step,len(header)))
        for huella in range(len(sensor_data)):
            for sensorType in sensor_data[huella]["s"]:
                for sensor in (sensor_data[huella]["s"][sensorType]):
                    sensorName = 'Ap300'
                    for j in range (len(lista_Ap)):
                        if (lista_Ap['MAC'][j] == sensor):
                            sensorName = lista_Ap['nºAp'][j]
                        if sensorName in header:
                            is_unknown = False
                            csv_data[huella][header.index(sensorName)] = sensor_data[huella]["s"][sensorType][sensor]
                                                                                                                                                                                                                                                                                                                 

        self.headerClassify = header
        csv_dataClassify = csv_data
        payload = {'location_names': self.naming['to'], 'predictions': []}

        threads = [None]*len(self.algorithms)
        self.results = [None]*len(self.algorithms)
        threads[0] = Thread(target=self.do_classification, args=(0, "LSTM",step,csv_dataClassify,header))
        threads[0].start()
        threads[0].join()

        for result in self.results:
            if result != None:
                payload['predictions'].append(result)
        payload['is_unknown'] = is_unknown
        return payload

    def minmax_norm(self, df, maximo, minimo):  
        df_minmax_norm = (df - minimo) / (maximo - minimo)    
        return df_minmax_norm

    def normaliza (self, step, Train, datos, header):
        
        df = pd.DataFrame(datos)
        pasos_norm = numpy.zeros((step,67))
        for j in range(len(header)):
            if header[j] in Train.columns:
                maximo = max(Train[header[j]])
                minimo = min(Train[header[j]])
            pasos_norm[:,step-j] = self.minmax_norm(df[j], maximo, minimo)
                                                    
        return pasos_norm

    def coord_zona(self, coord):
        if (coord[1] < 2):
            col = self.coordX(coord[0])
            zona = col    
        if  (coord[1] >= 2) and (coord[1] < 4): 
            col = self.coordX(coord[0])
            zona = col +18        
        if  (coord[1] >= 4) and (coord[1] < 6): 
            col = self.coordX(coord[0])
            zona = col +36
        if  (coord[1] >= 6) and (coord[1] < 8): 
            col = self.coordX(coord[0])
            zona = col +54        
        if  (coord[1] >= 8) and (coord[1] < 10): 
            col = self.coordX(coord[0])
            zona = col + 72
        return zona 
    
    def coordX(self, X):  
        aux = 0
        if X < 0:
            col1 = 0
        for i in range(18):
            if (X >= aux) and (X < aux+2):              
                col1 = i
            aux= aux +2
        if X > 36:
            col1 = 17
        
        return col1

    def do_classification(self, index, name,step,csv_dataClassify,header):
        t = time.time()
        pasos = np.empty([step,csv_dataClassify.shape[1]])
        try:
            if name == "LSTM":

                Huellas_Train = pd.read_csv('data/Huellas_sNorm.csv')
                csv_dataClassify[csv_dataClassify == 0] = -100
                pasos = self.normaliza(step,Huellas_Train,csv_dataClassify,header)

                if (step == 15):
                    pasos = pasos.reshape(1,step,67)
                    model_new = load_model('DLRNN_M11.h5', compile = False)
                    pred1 = model_new.predict(pasos)
                    prediccion = pred1[1]

                else:
                    pasos2 = np.empty([15,67])
                    for i in range(15):
                        pasos2[i] = pasos[0]
                    pasos2 = pasos2.reshape(1,15,67)
                    model_new= load_model.predict('DLRNN_M11.h5', compile = False)
                    pred1 = model_new(pasos2)
                    prediccion = pred1[1]
                
                self.logger.debug("Prediciión en coordenadas")
                self.logger.debug(prediccion) 
                pred_zona=np.zeros([prediccion.shape[0],prediccion.shape[1]])
                
                for i in range(prediccion.shape[0]):
                    for j in range(prediccion.shape[1]):
                        zona = self.coord_zona(prediccion[i,j,:])
                        pred_zona[i,j] = zona

                prediction =pred_zona.tolist()

            else:
                prediction = self.algorithms[name].predict_proba(csv_dataClassify)
        except Exception as e:
            self.logger.debug("Entro a Except")
            logger.error(csv_dataClassify)
            logger.error(str(e))
            return
        
        predict = {}
        if name == "LSTM":
            a = np.int(prediction[0][14]+1)
        
            self.logger.debug("Predicción en Zona")
            self.logger.debug(prediction[0][14]+1)
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


    def learn(self, fname):
        
        self.model = Model()
        t = time.time()
        configs = json.load(open('config.json', 'r'))
        
        self.header = []
        rows = []
        naming_num = 0
        with open('TrainM11.csv', 'r') as csvfile:
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
