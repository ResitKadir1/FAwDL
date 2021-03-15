
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import time
import tensorflow.keras
import matplotlib.pyplot as plt
import cv2
import os
from helperScript import *

from matplotlib.pyplot import *
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

import warnings
warnings.filterwarnings('ignore')



#Creating files
if not os.path.exists(r"C:\...\plots"):
    os.makedirs(r"C:\...\plots")

if not os.path.exists(r"C:\...\models"):
    os.makedirs(r"C:\...\models")

if not os.path.exists(r"C:\...\Analyses_outputs"):
    os.makedirs(r"C:\...\Analyses_outputs")

Analyses_outputs = r"C:\...\Analyses_outputs"

models = r"C:\...\models"

plots =r"C:\...\plots"

path = r"C:/.../datasets/"


train_df = pd.read_csv(path+"/fairface_label_train.csv")
test_df = pd.read_csv(path+"/fairface_label_val.csv")

train_df = pd.read_csv(path+"/fairface_label_train.csv")
test_df = pd.read_csv(path+"/fairface_label_val.csv")



train_df = train_df[['file', 'race']]
test_df = test_df[['file', 'race']]
train_df['file'] = 'FairFace/'+train_df['file']
test_df['file'] = 'FairFace/'+test_df['file']

size_train=train_df.groupby(["race"]).size()
size_test = test_df.groupby(["race"]).size()

size_train.to_csv(Analyses_outputs + "/train_size_race.csv")
size_test.to_csv(Analyses_outputs + "/test_size_race.csv")

print(train_df.sample(3))


percentage_train_dist = 100*train_df.groupby(['race']).count()[['file']]/train_df.groupby(['race']).count()[['file']].sum()
percentage_train_dist.to_csv(Analyses_outputs+ "/percentage_train_dist.csv")
percentage_test_dist = 100*test_df.groupby(['race']).count()[['file']]/test_df.groupby(['race']).count()[['file']].sum()
percentage_test_dist.to_csv(Analyses_outputs + "/percentage_test_dist.csv")

print(percentage_test_dist)

#combine Southeast Asian and East Asian into a single ASian Race

idx = train_df[(train_df['race'] == 'East Asian') | (train_df['race'] == 'Southeast Asian')].index
train_df.loc[idx, 'race'] = 'Asian'

idx = test_df[(test_df['race'] == 'East Asian') | (test_df['race'] == 'Southeast Asian')].index
test_df.loc[idx, 'race'] = 'Asian'

combine_train_dist =100*train_df.groupby(['race']).count()[['file']]/train_df.groupby(['race']).count()[['file']].sum()
combine_train_dist.to_csv(Analyses_outputs +"/combine_train_dist.csv")
combine_race_numbers_train = train_df.groupby(["race"]).size()
combine_race_numbers_train.to_csv(Analyses_outputs +"/combine_race_numbers.csv")
combine_train_dist



def getImagePixels(file):
    img = image.load_img(file, grayscale=False, target_size = (224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x



train_df['pixels'] = train_df['file'].progress_apply(getImagePixels)
test_df['pixels'] = test_df['file'].progress_apply(getImagePixels)

train_features = []
test_features = []

for i in range(0, train_df.shape[0]):
    train_features.append(train_df['pixels'].values[i])

for i in range(0, test_df.shape[0]):
    test_features.append(test_df['pixels'].values[i])

#convert into numpy
train_features = np.array(train_features)
train_features = train_features.reshape(train_features.shape[0], 224, 224, 3)

test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], 224, 224, 3)

#normalization
train_features = train_features / 255
test_features = test_features / 255

train_label = train_df[['race']]
test_label = test_df[['race']]
races = train_df['race'].unique()


#label encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
train_label['race'] =labelencoder.fit_transform(train_label['race'])
test_label['race'] =labelencoder.fit_transform(test_label['race'])
test_label.sample(3)


#One-Hot Encoder
onehotencoder = OneHotEncoder(sparse = False, handle_unknown = "error")
train_target = pd.DataFrame(onehotencoder.fit_transform(train_label))
test_target = pd.DataFrame(onehotencoder.fit_transform(test_label))
train_target.sample(3)

train_target = pd.get_dummies(train_label['race'], prefix='race')
test_target = pd.get_dummies(test_label['race'], prefix='race')


## Validation set to avoid overfitting
train_x, val_x, train_y, val_y = train_test_split(train_features, train_target.values, test_size=0.15)


#modelling

#get vggfacemodel
model = VggFaceModel()
model.load_weights(r"C:\Users\T-x23\Downloads\tensorflow-101-master\tensorflow-101-master\python\vgg_face_weights.h5")

model = freeze_layers_race(model)
checkpointer = ModelCheckpoint(filepath=models+'/race_model.hdf5', monitor = "val_loss",
                               verbose=1 , save_best_only=True , mode = 'auto')


model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

scores = []
patience =25
epochs = 250
batch_size = 256

early_stop = EarlyStopping(monitor='val_loss', patience=patience)
score = model.fit(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y), callbacks=[checkpointer, early_stop])
scores.append(score)
model.save_weights(models+'/race_model_son.h5')
model_json = model.to_json()
with open(models+"/race_model_son.json", "w") as json_file:
    json_file.write(model_json)


plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Train', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(score.history['loss'], color='b', label='Training Loss')
plt.plot(score.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(score.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(score.history['val_accuracy'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig(plots+"/Race_model_Train_val graphs.png")
plt.show()


#evaluation

validation_perf = model.evaluate(val_x, val_y, verbose=1)
validation_perf = pd.DataFrame(validation_perf)
validation_perf.to_csv(Analyses_outputs+"/race_validation_performence.csv")
print(validation_perf)

#Prediction
predictions = model.predict(test_features)



import random
#Test on Test dataset
prediction_classes = []
actual_classes = []

for i in range(0, predictions.shape[0]):
    prediction = np.argmax(predictions[i])
    prediction_classes.append(races[prediction])
    actual = np.argmax(test_target.values[i])
    actual_classes.append(races[actual])

    if i in  [random.randrange(1, 10000, 1) for i in range(1000)]:

        print(i)
        print("Actual: ",races[actual])
        print("Predicted: ",races[prediction])

        img = (test_df.iloc[i]['pixels'].reshape([224, 224, 3])) / 255
        plt.imshow(img)

        plt.show()
        print("----------------------")

cm = confusion_matrix(actual_classes, prediction_classes)
df_cm = pd.DataFrame(cm, index=races, columns=races)
conf =sns.heatmap(df_cm, annot=True,center=0,fmt="d",linewidths=.8, cmap="YlGnBu",square=True)


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(actual_classes, prediction_classes)
# Normalise
cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=races, yticklabels=races,cmap="YlGnBu",)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

print(classification_report(actual_classes,prediction_classes))

race_conf_mat = classification_report(actual_classes, prediction_classes,output_dict =True)
race_conf_mat_ = pd.DataFrame(race_conf_mat).transpose()
race_conf_mat_.to_csv(Analyses_outputs+"/Race_conf_matrix_report.csv")
