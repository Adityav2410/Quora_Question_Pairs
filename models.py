
from keras.models import Sequential
from keras.layers import Input,Dense, LSTM,TimeDistributed, Flatten, Reshape
from keras.models import Model
import keras.backend as K
from keras.preprocessing import sequence

class QuoraModel:
    
    def __init__(self, timeSlice = 2, questionLen = 25, wordEmbeddDim = 300 ):
        self.timeSlice = timeSlice
        self.questionLen = questionLen
        self.wordEmbeddDim = wordEmbeddDim
        self.getModel()
        
        
        
        
    def getModel(self):
        layer1Dim = 500
        inputs = Input(shape=(self.timeSlice,self.questionLen, self.wordEmbeddDim))
        lstm1 = TimeDistributed(LSTM(layer1Dim, return_sequences = False), name = 'lstm_1')(inputs)
        flatten = Flatten()(lstm1)
        output = Dense(1, activation='sigmoid')(flatten)
        self.model = Model(inputs = inputs, outputs = output)
        print "Model created"
        return self.model