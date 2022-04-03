#Importing Libraries
import os
from tqdm import tqdm
import numpy as np
from sklearn.utils import class_weight
from sklearn import preprocessing
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
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
    for imagePath in tqdm(images_path_all):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        label.append(imagePath.split('/')[1])
    max_key = max(globalDictValues, key=lambda x: len(set(globalDictValues[x])))
    keys = set(globalDictValues.keys())
    for key in keys.difference(max_key):
        if len(globalDictValues[key]) < len(globalDictValues[max_key]):
            for i in range(len(globalDictValues[max_key]) - len(globalDictValues[key])):
                image = random.choice(globalDictValues[key])
                image = cv2.imread("data/" + key + "/" + image)
                image = cv2.resize(image, (32, 32)).flatten()
                data.append(image)
                label.append(key)
    globalDictValues.clear()
    return data, label

#Reading images and labels

train_image_names=filesall('data')
data, labels = readimages(train_image_names)
print(len(data))
labels = np.array(labels)

test_image_names = filesall('test')
data_test_final, labels_test_final = readimages(test_image_names)
labels_test_final = np.array(labels_test_final)

#Label Encode labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
labels_test_final = le.transform(labels_test_final)

#Split Data to train and test after shuffle and also normalize data

#Normalize
data = np.array(data, dtype="float") / 255.0
data_test_final = np.array(data_test_final, dtype="float") / 255.0

#Train and test split
x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    stratify=labels,
                                                    test_size=0.25, random_state=100)


class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


d1 = dict()
d1[0] = class_weights[0]
d1[1] = class_weights[1]
d1[2] = class_weights[2]

#Labels to category
num_classes = 3
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
labels_test_final = np_utils.to_categorical(labels_test_final, num_classes)

#Model Building
load = os.path.isfile("part1.h5")
if not load:
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,)))
    model.add(Dense(512))
    model.add(Dense(64))
    model.add(Dense(3, activation='softmax'))
    model.summary()


    #Training
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, class_weight=class_weights)
    model.save("part1.h5")
    model.save_weights("part1_weights.h5")
else:
    model = load_model("part1.h5")
    model.load_weights("part1_weights.h5")

predictions = model.predict(x_test, batch_size=1)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=list(le.classes_)))

predictions = model.predict(data_test_final, batch_size=1)
print(classification_report(labels_test_final.argmax(axis=1), predictions.argmax(axis=1), target_names=list(le.classes_)))



