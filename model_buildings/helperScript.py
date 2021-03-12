import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy.io
import  pandas as pd
from datetime import datetime, timedelta

def VggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


def getImagePixels(image_path):
    img = image.load_img(r"C:\Users\T-x23\Downloads\tensorflow-101-master\tensorflow-101-master\python\wiki_crop/%s" % image_path[0], grayscale=False, target_size = (224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    #x = preprocess_input(x)
    return x



def getImagePixels_race(file):
    #print(file)
    img = image.load_img(file, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x


def clean_data(df):
    #remove pictures does not include face
    df = df[df['face_score'] != -np.inf]
    #some pictures include more than one face, remove them
    df = df[df['second_face_score'].isna()]
    #check threshold
    df = df[df['face_score'] >= 3]
    #some records do not have a gender information
    df = df[~df['gender'].isna()]
    return df



def feature_normalization(df):
    features = []
    for i in range(0, df.shape[0]):
        features.append(df['pixels'].values[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)
    features /= 255 
    return features #normalize in [0, 1]


def freeze_layers(model):
    #freeze all layers of VGG-Face except last 7 one
    for layer in model.layers[:-7]:
        layer.trainable = False
    	
    base_model_output = Sequential()
    base_model_output = Convolution2D(2 , (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    model = Model(inputs=model.input, outputs=base_model_output)

    #check trainable layers
    if False:
        for layer in model.layers:
            print(layer, layer.trainable)
    
        print("------------------------")
        for layer in age_model.layers:
            print(layer, layer.trainable)

    return model


def freeze_layers_2(model):
    #freeze all layers of VGG-Face except last 7 one
    for layer in model.layers[:-7]:
        layer.trainable = False
    	
    base_model_output = Sequential()
    base_model_output = Convolution2D(101 , (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    model = Model(inputs=model.input, outputs=base_model_output)

    #check trainable layers
    if False:
        for layer in model.layers:
            print(layer, layer.trainable)
    
        print("------------------------")
        for layer in age_model.layers:
            print(layer, layer.trainable)

    return model

def freeze_layers_race(model):
    #freeze all layers of VGG-Face except last 7 one
    for layer in model.layers[:-7]:
        layer.trainable = False
    	
    base_model_output = Sequential()
    base_model_output = Convolution2D(6 , (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    model = Model(inputs=model.input, outputs=base_model_output)

    #check trainable layers
    if False:
        for layer in model.layers:
            print(layer, layer.trainable)
    
        print("------------------------")
        for layer in age_model.layers:
            print(layer, layer.trainable)

    return model

def acc_graph():
    val_loss_change = []
    loss_change = []
    for i in range(0, len(scores)):
        val_loss_change.append(scores[i].history['val_loss'])
        loss_change.append(scores[i].history['loss'])

    plt.plot(val_loss_change, label='val_loss')
    plt.plot(loss_change, label='train_loss')
    plt.legend(loc='upper right')
    plt.savefig("gender loss graph")
    plt.title("Gender Loss graph")
    plt.show()
    return val_loss_change,loss_change

    
def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    exact_date = datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)
    
    return exact_date.year