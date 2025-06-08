import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random as r
import timeit
import tensorflow as tf
from utils import *

# --Read Data--
classNames = {0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# print(validY)



# --Build Model--

def buildVGG(dataSubsets, hParams):
    x_train, y_train, x_val, y_val, x_test, y_test = dataSubsets
    # x_train = tf.reshape(x_train, (-1, 32, 32, 1))
    # x_val = tf.reshape(x_val, (-1, 28, 28, 1))
    # x_test = tf.reshape(x_test, (-1, 28, 28, 1))
    
    startTime = timeit.default_timer()
    model = tf.keras.Sequential([])

    for i in range(len(hParams['convLayers'])):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(hParams['convLayers'][i]['conv_numFilters'], hParams['convLayers'][i]['conv_f'], 
                                padding=hParams['convLayers'][i]['conv_p'],activation=hParams['convLayers'][i]['conv_act'],input_shape=(32,32,3)))
        else:
            model.add(tf.keras.layers.Conv2D(hParams['convLayers'][i]['conv_numFilters'], hParams['convLayers'][i]['conv_f'], 
                                padding=hParams['convLayers'][i]['conv_p'],activation=hParams['convLayers'][i]['conv_act']))
        
        model.add(tf.keras.layers.MaxPooling2D(hParams['convLayers'][i]['pool_f']))
        model.add(tf.keras.layers.Dropout(hParams['convLayers'][0]['drop_prop']))

    model.add(tf.keras.layers.Flatten())


    for i in range(len(hParams['denseLayers'])):
        if i == len(hParams['denseLayers'])-1:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][i]))
        else:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][i], activation="relu"))


    print(model.summary())
    lf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mt = "accuracy"
    model.compile(hParams['optimizer'], lf, mt)
    hist = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val),
                    epochs=hParams['numEpochs'], verbose=1)
    hParams["paramCount"]= model.count_params()
    # print(hist.history)
    elapsedTime = timeit.default_timer() - startTime
    print("Training Time: ", elapsedTime)
    print("Training accuracy:", hist.history['accuracy'])
    startTime = timeit.default_timer()
    evaluation = model.evaluate(x_test, y_test, 1)
    elapsedTime = timeit.default_timer() - startTime
    print("Testing Time: ", elapsedTime)
    print("Testing accuracy:", evaluation[1])
    return hist.history, evaluation

def main():
    hParams = {
        'experimentName': 'C32_64x2__d0.2__D128_10__rms',
        'datasetProportion': 1.0,
        'numEpochs': 20,
        'denseLayers':  [512, 256, 43],
        'optimizer': 'rmsprop',
        'convLayers': [
            {
                'conv_numFilters': 32,
                'conv_f': 3,
                'conv_p': 'same',
                'conv_act': 'relu',
                'pool_f': (2,2),
                'pool_s': 1,
                'drop_prop': 0.2
            },
            {
                'conv_numFilters': 64,
                'conv_f': 3,
                'conv_p': 'same',
                'conv_act': 'relu',
                'pool_f': (2,2),
                'pool_s': 1,
                'drop_prop': 0.2
            },
            {
                'conv_numFilters': 64,
                'conv_f': 3,
                'conv_p': 'same',
                'conv_act': 'relu',
                'pool_f': (2,2),
                'pool_s': 1,
                'drop_prop': 0.2
            },
            {
                'conv_numFilters': 128,
                'conv_f': 3,
                'conv_p': 'same',
                'conv_act': 'relu',
                'pool_f': (2,2),
                'pool_s': 1,
                'drop_prop': 0.2
            }

        ]
    }

    dataSubsets = readData()
    trainResults, testResults = buildVGG(dataSubsets, hParams)
    writeExperimentalResults(hParams,trainResults,testResults)


main()