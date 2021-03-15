from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#IMPORT LIBRARIES
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing import image
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from helperScript import *




#IMPORT DATASETS AND PATH
Analyses_outputs = r"C:\....\Analyses_outputs"
plots = r"C:\....\plots"
models =r"C:\....\models"
mat = scipy.io.loadmat(r"C:...\wiki_crop\wiki_crop\wiki.mat")
cols = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]
instances = mat['wiki'][0][0][0].shape[1]
df = pd.DataFrame(index = range(0,instances), columns = cols)
for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            df[cols[j]] = pd.DataFrame(current_array[j][0])



#CLEANING DATASET
df = clean_data(df)


#drop unnecessary columns
df = df.drop(columns = ['dob','photo_taken','name','face_score','second_face_score','face_location'])
gender_dist = df['gender'].value_counts().sort_index()
gender_dist.to_csv(plots+"\gender_dist.csv")
print("Gender distribution",gender_dist)
classes = 2 #0: woman, 1: man


#DATA PREPROCESSING
df['pixels'] = df['full_path'].apply(getImagePixels)
target = df['gender'].values
target_classes = to_categorical(target, classes)
features = feature_normalization(df)


#split train test
train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)


#get vggfacemodel
model = VggFaceModel()
model.load_weights(r"C:\...\vgg_face_weights.h5")

#
model = freeze_layers(model)
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=models+'/gender_model_son.h5', verbose=1, save_best_only=True, mode = 'auto')

#MODEL COMPILING

scores = []
epochs = 250
batch_size = 256
for i in range(epochs):
    print("epoch ",i)
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
    score = model.fit(train_x[ix_train], train_y[ix_train], epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer])
    scores.append(score)
    model.save_weights(models+'\gender_model_weights.h5')
    model_json = model.to_json()
    with open(models+"\gender_model_weights.json", "w") as json_file:
        json_file.write(model_json)



validation_loss_change = []
loss_change = []
for i in range(0, len(scores)):
    validation_loss_change.append(scores[i].history['val_loss'])
    loss_change.append(scores[i].history['loss'])


#data visualization

plt.plot(validation_loss_change, label='val_loss')
plt.plot(loss_change, label='train_loss')
plt.legend(loc='upper right')
plt.savefig(plots+"\gender loss graph")
plt.title("Gender Loss graph")
plt.show()


accuracy_loss_change = []
acc_change = []
for i in range(0, len(scores)):
    accuracy_loss_change.append(scores[i].history['accuracy'])
    acc_change.append(scores[i].history['val_accuracy'])

plt.plot(accuracy_loss_change, label='accuracy')
plt.plot(acc_change, label='val_accuracy')
plt.legend(loc='upper right')
plt.savefig(plots + "\Gender accuracy graph")
plt.show()


model.evaluate(test_x, test_y, verbose=1)
predictions = model.predict(test_x)


pred_list = []
actual_list = []
for i in predictions:
    pred_list.append(np.argmax(i))
for i in test_y:
    actual_list.append(np.argmax(i))


cf_matrix =confusion_matrix(actual_list, pred_list)
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
plt.title("Confusion Gender Prediction ")
plt.savefig(plots + "\Gender Confusion matrix")


print(classification_report(actual_list,pred_list))

race_conf_mat = classification_report(actual_list, pred_list,output_dict =True)
race_conf_mat_ = pd.DataFrame(race_conf_mat).transpose()
race_conf_mat_.to_csv(Analyses_outputs+"/Gender_conf_matrix_report.csv")
