#lets collect some resources

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

#

model= Sequential()

#CRP layer
model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu',
                        input_shape=(64, 64, 3) 
                       )
         )
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flatten layer
model.add(Flatten())


model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=20, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DataSet Generator

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/Pneumonia_and_COVID19/TRAIN/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/Pneumonia_and_COVID19/TEST/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
 

#Fit the dataset into the Model

history=model.fit(
        training_set,
        steps_per_epoch=5000,
        epochs=3,
        validation_data=test_set,
)
 
#get the accuracy//

print(history.history['accuracy'][0] * 100)
f = open("accuracy.txt", 'w')
f.write('%d' % int(history.history['accuracy'][0] * 100))
f.close()


#Save the Model For future testing and predictions.

model.save('CovidOrPneumonia')

#--------------------------------------------#
    
