# reload(sys)
import pandas as pd
import numpy as np
import os
import random
from pdb import set_trace as bp
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import sys 
from sklearn.utils import shuffle
from dataGenerator import DataGenerator
from models import QuoraModel
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from utils import plotGraph,plotLengthHistogram,plotHist,labelPlot
from utils import createTiles, plotHistory
from modelTrainer import ModelTrainer
from keras.optimizers import Nadam




## Define parameters
timeSlice = 2
questionLen = 25
wordEmbeddDim = 100
batchSize  = 16
nEpochs = 10



dataGenerator = DataGenerator(batchSize = batchSize, 
                              questionLen = questionLen,
                              wordEmbeddDim = wordEmbeddDim)
nData = dataGenerator.nTrainData



net = QuoraModel(questionLen = questionLen,
                   wordEmbeddDim = wordEmbeddDim)
model = net.getModel1(load_weight = False)
nadam = Nadam(lr=1e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(optimizer=nadam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

trainer = ModelTrainer(modelName = 'model1', model = model)




trainHist = [[], []]
validHist = [[], []]
iterSize = 100
trainX, trainY = dataGenerator.getTrainBatch(2)
nIter = dataGenerator.nTrainData/(batchSize)
for i in range(nEpochs):
    for j in range(nIter):
#         trainX, trainY = dataGenerator.getTrainBatch()
        model.fit( trainX, trainY, verbose = False )
        if( j%50 == 0 ):
#             trainX, trainY = dataGenerator.getTrainBatch(batchSize = 10*batchSize)
            validX, validY = dataGenerator.getValidBatch(batchSize = 10*batchSize)
            trainLoss, trainAcc = model.evaluate( trainX, trainY, verbose = False )
            validLoss, validAcc = model.evaluate(validX, validY, verbose = False )
            trainer.addTrainDetails(trainLoss, trainAcc, validLoss, validAcc, compareWithBest = True)
            print( "Epoch:{0}\t Iter: {1}\t TrainLoss {2:.4f}\t ValidLos: {3:.4f} \
            TrainAcc: {4:.4f}\t ValidAcc: {5:.4f}"
                   .format(i, j+1, trainLoss, validLoss, trainAcc, validAcc))

trainer.saveDetails()        