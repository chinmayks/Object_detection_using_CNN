#Simple CNN for object detection

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from PIL import Image
import numpy as np
from PIL import Image
import PIL
import glob


yt = []
targetSize = (128, 128)
res = []

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

pathofTrainingDataset = 'C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/'
pathofTestingDataset =  'C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/'


#function for converting images to specific format for training dataset
def training_model(path, k):
    for filename in glob.glob(path):
        img = Image.open(filename)
        img = img.resize(targetSize)
        img = img.transpose(PIL.Image.TRANSPOSE)
        arr = np.array(img)
        arr = arr.T
        res.append(arr)
        ytemp = []
        ytemp.append(k)
        yt.append(ytemp)

ytest = []
res1 = []

#function for formatting images to a specific format for testing dataset
def testing_model(path, k):
    for filename in glob.glob(path):
        img = Image.open(filename)
        img = img.resize(targetSize)
        img = img.transpose(PIL.Image.TRANSPOSE)
        arr = np.array(img)
        arr = arr.T
        res1.append(arr)
        ytemp = []
        ytemp.append(k)
        ytest.append(ytemp)




#input for training dataset

training_model("pathofTrainingDataset/dataset/bookcase/*jpg", 12)
training_model("pathofTrainingDataset/dataset/desk/*jpg", 13)
training_model("pathofTrainingDataset/dataset/dining_table/*jpg", 5)
training_model("pathofTrainingDataset/dataset/four-poster/*jpg", 2)
training_model("pathofTrainingDataset/dataset/home_theater/*jpg", 4)
training_model("pathofTrainingDataset/dataset/microwave/*jpg", 9)
training_model("pathofTrainingDataset/dataset/patio/*jpg", 11)
training_model("pathofTrainingDataset/dataset/quilt/*jpg", 3)
training_model("pathofTrainingDataset/dataset/restaurant/*jpg", 10)
training_model("pathofTrainingDataset/dataset/sliding_door/*jpg", 6)
training_model("pathofTrainingDataset/dataset/studio_couch/*jpg", 1)
training_model("pathofTrainingDataset/dataset/wardrobe/*jpg", 7)
training_model("pathofTrainingDataset/dataset/window_shade/*jpg", 8)


#input for testing dataset

testing_model("pathofTestingDataset/testingdataset/one/*jpg", 1)
testing_model("pathofTestingDataset/testingdataset/two/*jpg", 2)
testing_model("pathofTestingDataset/testingdataset/three/*jpg", 3)
testing_model("pathofTestingDataset/testingdataset/four/*jpg", 4)
testing_model("pathofTestingDataset/testingdataset/five/*jpg", 5)
testing_model("pathofTestingDataset/testingdataset/six/*jpg", 6)
testing_model("pathofTestingDataset/testingdataset/seven/*jpg", 7)
testing_model("pathofTestingDataset/testingdataset/eight/*jpg", 8)
testing_model("pathofTestingDataset/testingdataset/nine/*jpg", 9)
testing_model("pathofTestingDataset/testingdataset/ten/*jpg", 10)
testing_model("pathofTestingDataset/testingdataset/eleven/*jpg", 11)
testing_model("pathofTestingDataset/testingdataset/twelve/*jpg", 12)
testing_model("pathofTestingDataset/testingdataset/thirteen/*jpg", 13)


X_train = np.array(res)
y_train = np.array(yt)
X_test = np.array(res1)
y_test = np.array(ytest)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
#adding conv layer with 32 filters and relu as an activation function
model.add(Conv2D(32, (3, 3), input_shape=(3, 128, 128), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#adding dropout to avoid overfitting
model.add(Dropout(0.3))
#adding conv layer with 32 filters and relu as an activation function
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#performing max pooling with size 2 * 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

list_of_epochs = [10000]
accuracy = []

# Compile model
for i in list_of_epochs:
    epochs = i
    lrate = 10e-7
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # fit the model according to the given training dataset
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

    # Final evaluation of the model after learning the weights
    scores = model.evaluate(X_test, y_test, verbose=0)
    #print the output
    print("Epoch:" , i , "Accuracy: %.2f%%" % (scores[1]*100))
    accuracy.append(scores[1] * 100)

print(accuracy)