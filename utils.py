import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plotGraph(x, y, xlabel='', ylabel ='', title = ''):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plotLengthHistogram(data):
	lenFreq = defaultdict(int)
	question1 = data['question1']
	try:
	    for question in question1:
	        lenFreq[len(question)]+=1
	    question2 = data['question2']
	    for question in question2:
	        lenFreq[len(question)]+=1
	except Exception as e:
	    print (e)
	plotGraph( lenFreq.keys(), lenFreq.values(), 'length of question', 'No of questions', 'Frequency vs length' )


def plotHist(values, xlabel='', ylabel='', title=''):
    plt.hist(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def labelPlot(xlabel='', ylabel='', title=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def createTiles(x=1,y=1,hwidth=8,vwidth=4): 
    fig,plots = plt.subplots(x,y,figsize=(hwidth,vwidth));
    plots = plots.flatten()
    return(fig, plots)


def createTiles(x=1,y=1,hwidth=8,vwidth=4): 
    fig,plots = plt.subplots(x,y,figsize=(hwidth,vwidth));
    plots = plots.flatten()
    return(fig, plots)


def plotHistory(trainHist, validHist):
    fig, plots = createTiles(1,2,15,5)
    plots[0].plot(trainHist[0]);
    plots[0].plot(validHist[0])
    plots[0].set_title('Loss')
    plots[0].set_xlabel('Epochs')
#     plots[1].imshow(img_gray, cmap = plt.get_cmap('gray'));     plots[1].set_title('Gray Scale Image')

    plots[1].plot(trainHist[1]);
    plots[1].plot(validHist[1])
    plots[1].set_title('Accuracy')
    plots[1].set_xlabel('Epochs')
    plt.show()
    
