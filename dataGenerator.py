#
import pandas as pd
import numpy as np
import os
import random
from pdb import set_trace as bp
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence


class DataGenerator:
    
    def __init__(self, batchSize = 10, dirpath = './data', trainFile = 'train.csv', testFile = 'test.csv',\
                 embeddFileName = 'embeddDict_glove_6b_100d.pickle',  questionLen = 40, wordEmbeddDim = 100):
        self.batchSize = batchSize
        self.dirpath = dirpath
        self.trainFile = os.path.join(dirpath, trainFile)
        self.testFile = os.path.join(dirpath, testFile)
        self.embeddFileName = embeddFileName
        self.wordEmbeddDim = wordEmbeddDim
        self.questionLen = questionLen
        self.__prepareData__()
        
    def __prepareData__(self):
        
        self.trainFilePath = os.path.join(self.dirpath, self.trainFile)
        self.testFilePath = os.path.join(self.dirpath, self.testFile)
        self.trainData, self.validData = train_test_split( pd.read_csv(self.trainFile) , test_size=0.2)
        self.trainData = shuffle(self.trainData)
        self.validData = shuffle(self.validData)
        self.trainData.dropna(inplace=True)
        self.validData.dropna(inplace=True)
        self.defaultValue = np.zeros(self.wordEmbeddDim)

        self.trainCounter = 0
        self.nTrainData = self.trainData.shape[0]
        self.validCounter = 0
        self.nValidData = self.validData.shape[0]
        
        with open(os.path.join(self.dirpath, self.embeddFileName), 'rb') as handle:
            self.embeddDict = pickle.load(handle)
        assert self.wordEmbeddDim == self.embeddDict.values()[0].shape[0]
        print( "Data Generator succesfully initialized :) :) :) " )

    def getTrainBatch(self, batchSize = None):
    	if not batchSize:
    		batchSize = self.batchSize
        trainX = np.zeros((batchSize, 2, self.questionLen, self.wordEmbeddDim))
        trainY = np.zeros(batchSize).astype(np.int32)
        trainX[:,:,0] = self.embeddDict['start']

        i = 0
        while( i < batchSize):
            if self.trainCounter >= self.nTrainData :
                self.trainCounter = 0
            currData = self.trainData.iloc[self.trainCounter]
            ques1 = text_to_word_sequence(currData['question1'])
            ques2 = text_to_word_sequence(currData['question2'])
            len1 = len(ques1)
            len2 = len(ques2)
            index = 0
            try:
                for index in range( min(len1, self.questionLen-1 )):
                    trainX[i,0,index+1] = self.embeddDict.get(ques1[index], self.defaultValue)
                trainX[i, 0,index+1] = self.embeddDict['end']
                for index in range( min(len2, self.questionLen-1 )):
                    trainX[i,1,index+1] = self.embeddDict.get(ques2[index],self.defaultValue)
                trainX[i,1,index+1] = self.embeddDict['end']
                trainY[i] = currData['is_duplicate']
                i += 1
                self.trainCounter += 1
                
            except Exception as e :
            	self.trainCounter += 1
                print ("TrainBatch Exception: " + str(e) )
                
                
        return trainX, trainY


    def getValidBatch(self, batchSize = None):
    	if not batchSize:
    		batchSize = self.batchSize

        validX = np.zeros((batchSize, 2, self.questionLen, self.wordEmbeddDim))
        validY = np.zeros(batchSize).astype(np.int32)
        validX[:,:,0] = self.embeddDict['start']

        i = 0
        defaultValue = np.zeros(self.wordEmbeddDim)
        while( i < batchSize):
            if self.validCounter >= self.nValidData:
                self.validCounter = 0
            currData = self.validData.iloc[self.validCounter]
            ques1 = text_to_word_sequence(currData['question1'])
            ques2 = text_to_word_sequence(currData['question2'])
            len1 = len(ques1)
            len2 = len(ques2)
            index = 0
            try:
                for index in range( min(len1, self.questionLen-1 )):
                    validX[i,0,index+1] = self.embeddDict.get(ques1[index], self.defaultValue )
                validX[i,0,index+1] = self.embeddDict['end']
                for index in range( min(len2, self.questionLen-1 )):
                    validX[i,1,index+1] = self.embeddDict.get(ques2[index], self.defaultValue)
                validX[i,1,index+1] = self.embeddDict['end']
                validY[i] = currData['is_duplicate']
                i+=1
                self.validCounter += 1
            except Exception as e:
            	self.validCounter += 1
                print("ValidBatch Exception:  " + str(e) )
                
        return validX, validY


    def trainGenerator(self):
    	while(1):
    		yield self.getTrainBatch()

    def validGenerator(self):
    	while(1):
    		yield self.getValidBatch()
