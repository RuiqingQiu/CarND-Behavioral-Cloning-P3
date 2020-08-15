import csv
import cv2
import numpy as np 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
lines = []


# image_path = '/opt/training/IMG/'
# image_path = '../data/IMG/'
image_path = '../mydata/IMG/'
# with open ('/opt/training/driving_log.csv') as csvfile:
# with open('../data/driving_log.csv') as csvfile:
with open('../mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Skip 1st line
    next(reader)

    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    # center image only
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = image_path + filename
    image = cv2.imread(current_path)
    # normal size
    images.append(image)
#     images.append(cv2.resize(image, (64, 64)))
    measurement = float(line[3])
    measurements.append(measurement)

#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = image_path + filename
#     image = cv2.imread(current_path)
#     steering_center = float(line[3])
#     # create adjusted steering measurements for the side camera images
#     correction = 0.1 # this is a parameter to tune
#     steering_left = steering_center + correction
#     steering_right = steering_center - correction

#     # read in images from center, left and right cameras
#     img_center = cv2.imread(image_path + line[0].split('/')[-1])
#     img_left = cv2.imread(image_path + line[1].split('/')[-1])
#     img_right = cv2.imread(image_path + line[2].split('/')[-1])

#     # add images and angles to data set
#     images.extend([img_center, img_left, img_right])
#     measurements.extend([steering_center, steering_left, steering_right])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement * -1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# A very simple network regression network
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))

# LeNet
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0)))
# model.add(Conv2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# More Powerful network
model = Sequential()
# normal image size
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# resized   
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(1e-4))
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model-more-powerful-10.h5')
exit()
