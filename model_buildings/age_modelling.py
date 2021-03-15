from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#import libraries

import scipy.io
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

from helperScript import *

#IMPORT DATASETS AND PATH
##https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ (data link)
plots = r"C:....\plots"
models =r"C:....\models"
mat = scipy.io.loadmat(r"...\wiki_crop\wiki_crop\wiki.mat")
cols = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]
instances = mat['wiki'][0][0][0].shape[1]
df = pd.DataFrame(index = range(0,instances), columns = cols)
for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            df[cols[j]] = pd.DataFrame(current_array[j][0])



#convert data time
df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)
df['age'] = df['photo_taken'] - df['date_of_birth']

#CLEANING DATASET
df = clean_data(df)
df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])
df = df[(df.age < 100) & (df.age < 100)]
age_dist=df['age'].value_counts().sort_index()
age_dist.to_csv(plots+"/age_dist.csv")
print("Age distribution :",age_dist)


classes = 101 #(0,100)
df['pixels'] = df['full_path'].apply(getImagePixels)
target = df['age'].values
target_classes = to_categorical(target, classes)
features = feature_normalization(df)



train_x, test_x, train_y, test_y = train_test_split(features, target_classes , test_size=0.30)


#get vggfacemodel
model = VggFaceModel()
model.load_weights(r"C:...\vgg_face_weights.h5")

#vgg_weight_link :https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view

model = freeze_layers_2(model)
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=models+'/age_model_son.h5', verbose=1, save_best_only=True, mode = 'auto')

#MODEL cOMPILING

scores = []
epochs = 250
batch_size = 256
for i in range(epochs):
    print("epoch ",i)
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
    score = model.fit(train_x[ix_train], train_y[ix_train], epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer],verbose=1)
    scores.append(score)
    model.save_weights(models+'/age_model_weights.h5')
    model_json = model.to_json()
    with open(models+"/age_model_weights.json", "w") as json_file:
        json_file.write(model_json)


val_loss_change = []
loss_change = []
for i in range(0, len(scores)):
    val_loss_change.append(scores[i].history['val_loss'])
    loss_change.append(scores[i].history['loss'])

plt.plot(val_loss_change, label='val_loss')
plt.plot(loss_change, label='train_loss')
plt.legend(loc='upper right')
plt.title("Appereant Age Loss graph")
plt.savefig(plots+"\Appereant Age loss graph")

plt.show()



acc_loss_change = []
acc_change = []
for i in range(0, len(scores)):
    acc_loss_change.append(scores[i].history['accuracy'])
    acc_change.append(scores[i].history['val_accuracy'])


plt.plot(acc_loss_change, label='accuracy')
plt.plot(acc_change, label='val_accuracy')
plt.legend(loc='upper right')
plt.savefig(plots+"\Appereant accuracy graph")

plt.show()

#Model evaluation on test set
#loss and accuracy on validation set
print("Model Scores: ",model.evaluate(test_x, test_y, verbose=1))
predictions = model.predict(test_x)
output_indexes = np.array([i for i in range(0, 101)])
apparent_predictions = np.sum(predictions * output_indexes, axis = 1)


mae = 0

for i in range(0 ,apparent_predictions.shape[0]):
    prediction = int(apparent_predictions[i])
    actual = np.argmax(test_y[i])
    abs_error = abs(prediction - actual)
    #actual_mean = actual_mean + actual
    mae = mae + abs_error
mae = mae / apparent_predictions.shape[0]
print("mae: ",mae)
print("instances: ",apparent_predictions.shape[0])
