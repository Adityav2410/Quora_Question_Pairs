import os
from keras.models import Sequential
from keras.layers import Input,Dense, LSTM,TimeDistributed, Flatten, Reshape
from keras.models import Model
import keras.backend as K
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint

class QuoraModel:
    
    def __init__(self, timeSlice = 2, questionLen = 25, wordEmbeddDim = 300, dataPath = './modelWeights', weightFile = 'model.h5' ):
        self.timeSlice = timeSlice
        self.questionLen = questionLen
        self.wordEmbeddDim = wordEmbeddDim
        self.dataPath = dataPath
        self.model_best_loss = 100
        self.model_best_acc =  0
        self.weightFilePath = os.path.join(dataPath, weightFile )
        
    def getCallbackList(self, model_name = 'model1'):
        fileName= model_name + "_weights-{epoch:02d}-{val_loss:.3f}-{val_acc:.2f}.hdf5"
        filePath = os.path.join(self.dataPath, fileName)
        checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list
        
    def getModel1(self, load_weight = True):
        layer1Dim = 500
        layer2Dim = 250
        nDense1   = 256
        nDense2   = 128
        inputs = Input(shape=(self.timeSlice,self.questionLen, self.wordEmbeddDim))
        batchNorm1 = BatchNormalization(name = "batch_norm_1")(inputs)
        lstm1 = TimeDistributed(LSTM(layer1Dim, activation = 'relu', return_sequences = True, dropout = 0.3), name = 'lstm_1')(batchNorm1)
        batchNorm2 = BatchNormalization( name = "batch_norm_2")(lstm1)
        lstm2 = TimeDistributed(LSTM(layer2Dim, activation = 'relu', return_sequences = False), name = 'lstm_2')(batchNorm2)
        flatten = Flatten(name = 'flatten')(lstm2)
        batchNorm3 = BatchNormalization(name = "batch_norm_3")(flatten)
        dense1 = Dense(nDense1, activation = 'relu')(batchNorm3)
        batchNorm4 = BatchNormalization(name = "batch_norm_4")(dense1)
        dense2 = Dense(nDense2, activation = 'relu')(batchNorm4)
        output = Dense(1, activation = 'sigmoid')(dense2)
        self.model = Model(inputs = inputs, outputs = output)
        if load_weight:
            if os.path.isfile(self.weightFilePath) :
                try:
                    self.model.load_weights(self.weightFilePath)
                    print("Weights loaded")
                except:
                    print("Failed to load weights")
                    pass
        return self.model

    
    def getModel2(self):
        layer1Dim = 500
        layer2Dim = 250
        num_dense   = 100
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25
        act = 'relu'
        bp()
        lstm_layer = LSTM(layer1Dim)

        sequence_1_input = Input(shape=(self.questionLen, self.wordEmbeddDim), dtype='int32')
        x1 = lstm_layer(sequence_1_input)
        
        sequence_2_input = Input(shape=(self.questionLen, self.wordEmbeddDim), dtype='int32')
        x2 = lstm_layer(sequence_2_input)

        merged = concatenate([x1, x2])
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        self.model = Model(inputs=[sequence_1_input, sequence_2_input], \
                           outputs=preds)
        
        return self.model