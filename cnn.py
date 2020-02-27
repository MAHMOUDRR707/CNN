#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#data processing
data= pd.read_csv("Churn_Modelling.csv")
x=data.iloc[:,3:13].values
y=data.iloc[:,13].values


#categorical features
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label1=LabelEncoder()
x[:,1]=label1.fit_transform(x[:,1])
label2=LabelEncoder()
x[:,2]=label2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()


#trainign splitting testting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


#feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#cnn 
from keras.models import Sequential 
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten
classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from skelarn.metrics import  confusion_matrix
cm=confusion_matrix(y_test,y_pred)



























