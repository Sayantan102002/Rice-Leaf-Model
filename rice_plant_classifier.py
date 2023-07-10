# -*- coding: utf-8 -*-
"""
Created on Fri May 12 01:44:01 2023

@author: Sayantan Chakraborty

"""

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel, laplace, gaussian, meijering
from sklearn.model_selection import train_test_split
from rembg import remove
from datetime import datetime

print(os.listdir("Diseases/"))
print(os.listdir("Nutrition_Defeciency/"))
SIZE = 256


images = []
labels = []
train_images = []
train_labels = []
test_images = []
test_labels = []


# Global Variables for storing data in excel
n_est = 100
accuracy = 0
models = []
precision_score = 0
recall_score = 0
f1_score = 0
# -------------------------------------------


for path in glob.glob("Nutrition_Defeciency/*"):
    label = path.split("\\")[-1]
    print(label)
    for i in range(0, 5):
        # for img_path in glob.glob(os.path.join(path, "*.jpg")):
        img_path = glob.glob(os.path.join(path, "*.JPG"))[i]
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        images.append(img)
        labels.append(label)

    # for img_path in glob.glob(os.path.join(path, "*.jpg")):
    #     print(img_path)
    #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     img = cv2.resize(img, (SIZE, SIZE))
    #     images.append(img)
    #     images.append(label)


for path in glob.glob("Diseases/*"):
    label = path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        images.append(img)
        labels.append(label)

# for path in glob.glob("Diseases/*"):
#     label = path.split("\\")[-1]
#     print(label)
#     # For images ending with JPG
#     for img_path in glob.glob(os.path.join(path, "*.JPG")):
#         print(img_path)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (SIZE, SIZE))
#         images.append(img)
#         labels.append(label)

images = np.array(images)
labels = np.array(labels)


train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.25, random_state=42)


le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
print(test_labels_encoded, train_labels_encoded)


x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape[0])


def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):
        input_img = x_train[image, :, :, :]
        img = input_img

        df = pd.DataFrame()

        # FEATURE 1 - Bunch of Gabor filter responses

        # Generate Gabor features
        num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):  # Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.5
                # Label Gabor columns as Gabor1, Gabor2, etc.
                gabor_label = 'Gabor' + str(num)
    #                print(gabor_label)
                ksize = 9
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                # Now filter the image and add values to a new column
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                # Labels columns as Gabor1, Gabor2, etc.
                df[gabor_label] = filtered_img
                num += 1  # Increment for gabor column label
        models.append("Gabor")
        # FEATURE 2 - Sobel filter

        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
        models.append("Sobel")

        # FEATURE 3 - Laplace filter
        edge_laplace = laplace(img)
        edge_laplace1 = edge_laplace.reshape(-1)
        df['Laplace'] = edge_laplace1
        models.append("Laplace")

        # FEATURE 4 - Gaussian filter
        # edge_gaussian = gaussian(
        #     img, sigma=1, mode='reflect', channel_axis=False)
        # edge_gaussian1 = edge_gaussian.reshape(-1)
        # df['Gaussian'] = edge_gaussian1

        # FEATURE 5 - Meijering Filter
        # edge_meijering = meijering(img)
        # edge_meijering1 = edge_meijering.reshape(-1)
        # df['Meijering'] = edge_meijering1

        # edge_otsu = threshold_otsu(img)
        # edge_otsu1 = edge_otsu.reshape(-1)
        # df['Otsu'] = edge_otsu1

        image_dataset = image_dataset._append(df)
    return image_dataset


image_features = feature_extractor(x_train)
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
# Reshape to #images, features
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))

# Define the classifier
RF_model = RandomForestClassifier(n_estimators=n_est, random_state=42)

# Fit the model on training data
RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding

# Predict on Test data
# Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Predict on test
test_prediction = RF_model.predict(test_for_RF)
# Inverse le transform to get original label back.
test_prediction = le.inverse_transform(test_prediction)

# Print overall accuracy
accuracy = metrics.accuracy_score(test_labels, test_prediction)
precision_score = metrics.precision_score(
    test_labels, test_prediction, average='micro')
recall_score = metrics.recall_score(
    test_labels, test_prediction, average='micro')
f1_score = metrics.f1_score(test_labels, test_prediction, average='micro')
print("Accuracy = ", accuracy)
print("Precision Score = ", precision_score)
print("Recall Score = ", recall_score)
print("F1 Score = ", f1_score)

# Print confusion matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6, 6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax)


"""
    Function to update details in excel file "Rice Leaf Model.xlsx"
"""


def writeData():
    # new dataframe with same columns
    df = pd.DataFrame({'Classifier': ['RF'],
                       'N_Estimators': [n_est],
                       'Sample Size': [x_train.shape[0]],
                       'Accuracy': [accuracy],
                       'Precision Score': [precision_score],
                       'Recall Score': [recall_score],
                       'F1 Score': [f1_score],
                       'Models': [set(models)],
                       'Date and Time': [datetime.now()]
                       })
    try:
        reader = pd.read_excel('Rice Leaf Model.xlsx')
        # read  file content
        # print(reader)
        # create writer object
        # used engine='openpyxl' because append operation is not supported by xlsxwriter
        writer = pd.ExcelWriter('Rice Leaf Model.xlsx', engine='openpyxl',
                                mode='a', if_sheet_exists="overlay")

        # append new dataframe to the excel sheet
        df.style.set_properties(**{'text-align': 'center'}).to_excel(
            writer, index=False, header=False, startrow=len(reader) + 1)
        # close file
        writer.close()
    except FileNotFoundError:
        writer = pd.ExcelWriter('Rice Leaf Model.xlsx', engine='xlsxwriter')
        df.style.set_properties(
            **{'text-align': 'center'}).to_excel(writer, index=False)

        # close file
        writer.close()


writeData()

# Extract features and reshape to right dimensions
# Expand dims so the input is (num images, x, y, c)
# for img in x_test:

#     input_img = np.expand_dims(img, axis=0)
#     input_img_features = feature_extractor(input_img)
#     input_img_features = np.expand_dims(input_img_features, axis=0)
#     input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
#     # Predict
#     img_prediction = RF_model.predict(input_img_for_RF)
#     # Reverse the label encoder to original name
#     img_prediction = le.inverse_transform(img_prediction)
#     print("The prediction for this image is: ", img_prediction)
#     # rows, cols = (x_test.shape[0]//4)+1,4
#     # plt.subplot(rows, cols, c)
#     # plt.imshow(img)
#     # plt.title(f'{img_prediction}')
#     c+=1
