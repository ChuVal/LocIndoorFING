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
from core.data_processor import DataLoader
from core.model import Model
from core.trajectories import Trajectories
from core.aps import Aps
from tensorflow.keras.models import load_model

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
        #self.logger.debug(len(header))
        datos = pd.read_csv('TrainM11.csv')
        header = list(datos.columns[0:67])
        is_unknown = True
        step = len(sensor_data)
        lista_Ap = pd.read_csv("Listado_ApsA.csv")
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
        self.logger.debug(self.algorithms.keys())
        for i, alg in enumerate(self.algorithms.keys()):
            self.logger.debug("for")
            threads[i] = Thread(target=self.do_classification, args=(i, alg, step,csv_dataClassify))
            threads[i].start()
        #self.do_classification("LSTM", step)
        for i, _ in enumerate(self.algorithms.keys()):
            threads[i].join()

        for result in self.results:
            if result != None:
                payload['predictions'].append(result)
        payload['is_unknown'] = is_unknown
        #self.logger.debug('payload ' + str(payload))
        return payload
    
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
        for i in range(18):
            if (X >= aux) and (X < aux+2):
                col1 = i
            aux= aux +2
        if X == 36:
            col1 = 17

        return col1

    def do_classification(self, index, name,step,csv_dataClassify):
        t = time.time()
        pasos = np.empty([step,csv_dataClassify.shape[1]])
        self.logger.debug("dentro")
        try:
            if name == "LSTM":
                for h in range(15):
                    csv_dataClassify[h][csv_dataClassify[h] == 0] = -100
                    huella = csv_dataClassify[h]
                    huella = huella.reshape(67,1)
                    min_max_scaler = MinMaxScaler()
                    x_scaled = min_max_scaler.fit_transform(huella)
                    huella = x_scaled.reshape(1,67)
                    pasos[15-1-h] = huella
                    
                if (step == 15):
                    pasos = pasos.reshape(1,step,67)
                    self.logger.debug(pasos.shape)
                    model_new= load_model('DLRNN_M11.h5', compile = False)
                    pred = model_new.predict(pasos)
                    prediccion = pred[1]
                else:
                    self.logger.debug("Dentro del else")
                    pasos2 = np.empty([15,csv_dataClassify.shape[1]])
                    for i in range(15):
                        pasos2[i] = huella
                    pasos2 = pasos2.reshape(1,15,csv_dataClassify.shape[1])
                    model_new= load_model('DLRNN_M11.h5', compile = False)
                    pred = model_new.predict(pasos2)
                    prediccion = pred[1]
                    
                pred_zona=np.zeros([prediccion.shape[0],prediccion.shape[1]])
                print(pred_zona.shape)
                for i in range(prediccion.shape[0]):
                    for j in range(prediccion.shape[1]):
                        zona = self.coord_zona(prediccion[i,j,:])
                        pred_zona[i,j] = zona


                prediction = pred_zona
                self.logger.debug(prediccion)

            else:
                prediction = self.algorithms[name].predict_proba(self.csv_dataClassify)
        except Exception as e:
            logger.error(self.csv_dataClassify)
            logger.error(str(e))
            return

        predict = {}
        if name == "LSTM":
            a = np.int(prediction[0][14])
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
                if i == 0:
                    self.header = row
                else:
                    for j, val in enumerate(row):
                        if j == len(row)-1:
                            # this is a name of the location
                            if val not in self.naming['from']:
                                self.naming['from'][val] = naming_num
                                valor = str(int(float(val)))
                                self.naming['to'][naming_num] = "location" + "_" + valor
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

        names = ["LSTM"]
        classifiers = [self.model.model_clas(configs)]

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
              
                self.logger.debug("learned {}, {:d} ms".format(name, int(1000 * (t2 - time.time()))))
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
