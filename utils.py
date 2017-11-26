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
	    print e
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
