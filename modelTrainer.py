import pickle 
import os
from datetime import datetime
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, modelName, model , weightDirPath= './modelWeights'):
        self.weightDirPath = weightDirPath
        now = datetime.now()
        self.trainLoss = []
        self.trainAcc = []
        self.validLoss = []
        self.validAcc = []
        self.modelName = modelName
        self.model = model
        self.bestLoss = 10000
        self.bestAcc = -1
        self.weightDirPath = weightDirPath
        self.weightFilePath = './12345654321'
        self.historyFileName = self.modelName + '_' + now.strftime("%Y_%m_%d_%H:%M_") + 'trainHistory.pkl'
        if not os.path.isdir(self.weightDirPath):
            os.mkdir(self.weightDirPath, 0755)
        
    def getWeightFilePath(self, validLoss = 0.0, validAcc = 0.1 ):
        now = datetime.now()
        filename = self.modelName + '_' + now.strftime("%Y_%m_%d_%H:%M_") + 'loss_{0:.4f}_acc_{1:.3f}.h5py'.format(validLoss, validAcc)
        weightFilepath = os.path.join( self.weightDirPath, filename )
        return weightFilepath

        
    def updateBestParams(self, validLoss, validAcc):
        self.bestLoss = validLoss
        self.bestAcc =  validAcc
        if os.path.isfile(self.weightFilePath):
            os.remove(self.weightFilePath)
        self.weightFilePath = self.getWeightFilePath(validLoss, validAcc)
        self.model.save_weights(self.weightFilePath)
        print "Best loss: {0:.2f} \t Best Acc: {1:.2f}".format(validLoss, validAcc)
        print "Weights saved at \t " + self.weightFilePath
        
        
    def addTrainDetails(self, trainLoss, trainAcc, validLoss, validAcc, compareWithBest = False):
        self.trainLoss.append(trainLoss)
        self.trainAcc.append(trainAcc)
        self.validLoss.append(validLoss)
        self.validAcc.append(validAcc)
        if compareWithBest == True and validLoss < self.bestLoss:
            self.updateBestParams(validLoss, validAcc)
        
    def saveDetails(self):
        history = {}
        history['train_loss'] = self.trainLoss
        history['valid_loss'] = self.validLoss
        history['train_acc'] = self.trainAcc
        history['valid_acc'] = self.validAcc
        try:
            with open(self.historyFileName, 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print "Encountered exception: " + str(e)
            print "Failed to save training history"
    
    def createTiles(self, x=1,y=1,hwidth=8,vwidth=4): 
        fig,plots = plt.subplots(x,y,figsize=(hwidth,vwidth));
        plots = plots.flatten()
        return(fig, plots)

    def plotTrainHist(self):
        fig, plots = self.createTiles(1,2,15,5)
        plots[0].plot(self.trainLoss);
        plots[0].plot(self.validLoss)
        plots[0].set_title('Loss')
        plots[0].set_xlabel('Epochs')

        plots[1].plot(self.trainAcc);
        plots[1].plot(self.validAcc)
        plots[1].set_title('Accuracy')
        plots[1].set_xlabel('Epochs')
        plt.show()