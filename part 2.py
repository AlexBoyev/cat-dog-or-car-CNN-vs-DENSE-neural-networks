#Importing Libraries
import os
from tqdm import tqdm
import numpy as np
from sklearn.utils import class_weight
from sklearn import preprocessing
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import keras
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from sklearn.metrics import classification_report
from keras.utils import np_utils
import imutils
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2
import random

global globalDictValues
globalDictValues = {}

#Get File path
def filesall(path):
    global globalDictValues
    filename = []
    all_folder = os.listdir(path)
    for f in all_folder:
        mylist = os.listdir(path+'/'+f)
        filename.extend([path+'/'+f+'/'+s for s in mylist])
        globalDictValues[f] = mylist
    random.seed(42)
    random.shuffle(filename)
    return filename

#Read Images
def readimages(images_path_all):
    global globalDictValues
    data = []
    label = []
    random.seed(42)
    random.shuffle(images_path_all)
    for imagePath in tqdm(images_path_all):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        label.append(imagePath.split('/')[1])
    max_key = max(globalDictValues, key=lambda x: len(set(globalDictValues[x])))
    keys = set(globalDictValues.keys())
    for key in keys.difference(max_key):
        if len(globalDictValues[key]) < len(globalDictValues[max_key]):
            for i in range(len(globalDictValues[max_key]) - len(globalDictValues[key])):
                image = random.choice(globalDictValues[key])
                image = cv2.imread("data/" + key + "/" + image)
                image = cv2.resize(image, (64, 64))
                data.append(image)
                label.append(key)
    globalDictValues.clear()
    return data, label

#Reading images and labels

train_image_names=filesall('data')
data, labels = readimages(train_image_names)

labels = np.array(labels)

test_image_names = filesall('test')
data_test_final, labels_test_final=readimages(test_image_names)
labels_test_final = np.array(labels_test_final)

#Label Encode labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
labels_test_final = le.transform(labels_test_final)

#Split Data to train and test after shuffle and also normalize data

#Normalize
data = np.array(data, dtype="float") / 255.0
data_test_final=np.array(data_test_final, dtype="float") / 255.0

#Train and test split
x_train, x_test, y_train, y_test = train_test_split(data, labels,stratify=labels, 
                                                    test_size=0.25)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                               y_train)
d1=dict()
d1[0] = class_weights[0]
d1[1] = class_weights[1]
d1[2] = class_weights[2]

#Labels to category
num_classes = 3
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
labels_test_final = np_utils.to_categorical(labels_test_final, num_classes)

#Model Building
chanDim = -1
load = os.path.isfile("part2.h5")
if not load:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))

    model.summary()

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
    datagen.fit(x_train)

    #Training
    batch_size = 32
    epochs = 50
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), class_weight=d1)
    model.save("part2.h5")
    model.save_weights("part2_weights.h5")
else:
    model = load_model("part2.h5")
    model.load_weights("part2_weights.h5")

predictions = model.predict(x_test, batch_size=1)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=list(le.classes_)))


predictions = model.predict(data_test_final, batch_size=1)
print(classification_report(labels_test_final.argmax(axis=1), predictions.argmax(axis=1), target_names=list(le.classes_)))






