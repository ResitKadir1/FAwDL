

import numpy as np
import cv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.python.framework import tensor_util
from model_buildings.helperScript import VggFaceModel

def is_tensor(x):
	return tensor_util.is_tensor(x)
	#return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)
#-----------------------

items = True

male_icon = cv2.imread("items/male1.png")
male_icon = cv2.resize(male_icon, (40, 40))

female_icon = cv2.imread("items/female1.png")
female_icon = cv2.resize(female_icon, (40, 40))
#-----------------------

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def ageModel():
	model = VggFaceModel()

	base_model_output = Sequential()
	base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	age_model = Model(inputs=model.input, outputs=base_model_output)

	age_model.load_weights("Models/age_model_weights.h5")

	return age_model

def genderModel():
	model = VggFaceModel()

	base_model_output = Sequential()
	base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	gender_model = Model(inputs=model.input, outputs=base_model_output)


	gender_model.load_weights("Models/gender_model_weights.h5")

	return gender_model

age_model = ageModel()
gender_model = genderModel()

#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

#------------------------

cap = cv2.VideoCapture(0) #capture webcam

while(True):
	ret, img = cap.read()
	#img = cv2.imread("test_image/kemal_sunal.jpg")
	img = cv2.resize(img, (640, 440))


	faces = face_cascade.detectMultiScale(img, 1.3, 5)

	for (x,y,w,h) in faces:
		if w > 130:

			overlay = img.copy(); output = img.copy(); opacity = 0.8
			cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),cv2.FILLED)
			cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


			cv2.rectangle(img,(x,y),(x+w,y+h),(67,67,67),3)
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]

			try:
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print(" no margin founded")

			try:
				#vgg-face expects inputs (224, 224, 3)
				detected_face = cv2.resize(detected_face, (224, 224))
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				#find out age and gender
				age_distributions = age_model.predict(img_pixels)
				apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))

				gender_distribution = gender_model.predict(img_pixels)[0]
				gender_index = np.argmax(gender_distribution)

				if gender_index == 0:
					 gender = "F"
				else:
					gender = "M"

				#background for age gender declaration
				info_box_color = (255,255,255)
				triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)

				#labels for age and gender
				cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

				if items:
					if gender == 'M':
						 gender_icon = male_icon
					else:
						gender_icon = female_icon

					img[y-75:y-75+male_icon.shape[0], x+int(w/2)-45:x+int(w/2)-45+male_icon.shape[1]] = gender_icon
				else:
					cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

			except Exception as e:
				print("exception",str(e))

	cv2.imshow('img',img)
	#cv2.putText(img,"IT ",(0,700),cv2.FONT_HERSHEY_SIMPLEX,2,(67,67,67))
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

cap.release()
cv2.destroyAllWindows()
