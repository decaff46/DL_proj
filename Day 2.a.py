"""
Data Augmentation:

    Feature-wise standardization
    ZCA whitening
    Random rotation, shifts, shear, and flips

"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Feature-wise Standardization:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# convert from int to float to get the mean and std for each pixel
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# define data preparation --> this seems to have a bug. not working properly
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
##datagen.fit(x_train)
###compute the feature standardization manually.
datagen.mean = x_train.mean(axis=0)
datagen.std = x_train.std(axis=0)

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    print(X_batch.min(), X_batch.mean(), X_batch.max())
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j], cmap=plt.get_cmap("gray"))
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    break

# ZCA Whitening
"""
 Image whitening는 보통 Principal Component Analysis (PCA)을 써서 했었지만, 최근에는  ZCA를 더 많이 사용 
 ZCA의 특징은 원본의 모든 디맨션을 유지 한다는 점. 즉, PCA와 다르게 화이트닝을 시킨후에도 원본의 모습을 갖고 있음. 
 각각의 엘레멘트들은 zero mean and unit standard derivation을 갖고 각각이 독립적이다.
"""
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening = True)
x_mean = x_train.mean(axis=0)
datagen.fit(x_train - x_mean)

for X_batch, y_batch in datagen.flow(x_train - x_mean, y_train, batch_size=9, shuffle=False):
    print(X_batch.min(), X_batch.mean(), X_batch.max())
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j], cmap=plt.get_cmap("gray"))
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    break

# Random Rotation
datagen = ImageDataGenerator(rotation_range= 90)
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    print(X_batch.min(), X_batch.mean(), X_batch.max())
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j], cmap=plt.get_cmap("gray"))
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    break

# Random Shifting
datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
    # show the plot
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    break


# Random Flip
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
    # show the plot
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    break

