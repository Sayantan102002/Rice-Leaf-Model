import random
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
from skimage.filters import sobel, laplace
from sklearn.model_selection import train_test_split

print(os.listdir("Diseases/"))
print(os.listdir("Nutrition_Defeciency/"))
SIZE = 256


# print("TRAIN IMAGES")
images = []
labels = []
train_images = []
train_labels = []
test_images = []
test_labels = []
# print(glob.glob("Diseases/*"))

for path in glob.glob("Nutrition_Defeciency/*"):
    label = path.split("\\")[-1]
    print(label)
    for i in range(0, 200):
        img_path = glob.glob(os.path.join(path, "*.jpg"))[i]
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        images.append(img)
        labels.append(label)

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
        # print(input_img)

        df = pd.DataFrame()
        # pixel_values = img.reshape(-1)
        # df['Pixel_Value'] = pixel_values  # Pixel value itself as a feature
        # df['Image_Name'] = image   #Capture image name as we read multiple images

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
                # print(gabor_label, ': theta=', theta, ': sigma=',
                #       sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  # Increment for gabor column label
                
                
        # FEATURE 2 - Sobel filter
        
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
        
        # FEATURE 3 - Laplace filter
        edge_laplace = laplace(img)
        edge_laplace1 = edge_laplace.reshape(-1)
        df['Laplace'] = edge_laplace1

        # edge_otsu = threshold_otsu(img)
        # edge_otsu1 = edge_otsu.reshape(-1)
        # df['Otsu'] = edge_otsu1

        image_dataset = image_dataset._append(df)
    return image_dataset


# print(feature_extractor(x_train))

image_features = feature_extractor(x_train)

# Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
# Reshape to #images, features
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))

# Define the classifier
# from sklearn.neighbors import KNeighborsClassifier
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

# Can also use SVM but RF is faster and may be more accurate.
# from sklearn import svm
# SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
# SVM_model.fit(X_for_RF, y_train)


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
print("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

# Print confusion matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6, 6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax)


# Extract features and reshape to right dimensions
# Expand dims so the input is (num images, x, y, c)
# c = 0
for img in x_test:
    input_img = np.expand_dims(img, axis=0)
    input_img_features = feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    # Predict
    img_prediction = RF_model.predict(input_img_for_RF)
    # Reverse the label encoder to original name
    img_prediction = le.inverse_transform(img_prediction)
    print("The prediction for this image is: ", img_prediction)

