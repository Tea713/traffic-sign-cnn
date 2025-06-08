import os
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import ast
import pickle
import random as r
from sklearn.utils import shuffle

def buildValAccuracyPlot():
    path = "./results/"
    filenames = os.listdir(path)
    numEpoch = 20
    yList = []
    yLabelList = []
    for fn in filenames:
        if ".txt" in fn:
            hParams,trainResults,_ = readExperimentalResults("results/"+fn)
            if hParams["numEpochs"] == numEpoch:
                yList.append(trainResults['val_accuracy'])
                yLabelList.append(hParams['experimentName'])

    plotCurves(x=np.arange(0, numEpoch),yList=yList, xLabel="Epoch",yLabelList=yLabelList, title="Compare_Validation_Accuracy")

def correspondingShuffle(x, y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    return tf.gather(x, shuffled_indices), tf.gather(y, shuffled_indices)

def writeExperimentalResults(hParams, trainResults, testResults):
    with open("results/" + hParams['experimentName']+ '.txt', 'w') as f:
        f.write(json.dumps(hParams))
        f.write('\n')
        f.write(json.dumps(trainResults))
        f.write('\n')
        f.write(json.dumps(testResults))

def readExperimentalResults(fileName):
    with open(fileName) as f:
        data = f.read().split('\n')
    
    hParams, trainResults, testResults = ast.literal_eval(data[0]), ast.literal_eval(data[1]), ast.literal_eval(data[2])

    return hParams, trainResults, testResults

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)

def plotPoints(xList, yList, pointLabels=[], xLabel="",
    yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)


def readData():
    data="./data/"
    train_link=data+"train.p"
    test_link= data + "test.p"
    valid_link = data + "valid.p"

    with open(train_link,"rb") as f:
        train = pickle.load(f)

    with open(test_link,"rb") as f:
        test= pickle.load(f)

    with open(valid_link, "rb") as f:
        valid = pickle.load(f)

    trainX = train['features']
    trainY = train['labels']

    trainX,trainY= shuffle(trainX,trainY)
    abitrary=r.randint(0,trainX.shape[0])
    # print(classNames[trainY[abitrary]])

    validX=valid['features']
    validY=valid['labels']
    testX=test['features']
    testY=test['labels']

    trainX=trainX.astype("float")/255.0
    validX=validX.astype("float")/255.0
    testX=testX.astype("float")/255.0

    print("Training Set Shape: ", trainX.shape)
    print("Testing Set Shape: ", testX.shape)
    print("Validation Set Shape: ", validX.shape)
    return trainX, trainY, validX, validY, testX, testY