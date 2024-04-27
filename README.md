# CottonDetection
- Base Version of the Detection Model

**Project Structure**

![image](https://github.com/kkwillijr/CottonDetection/assets/96938630/4cc0b4e8-528d-47f0-9bbb-bc8da313c708)

**Project Setup**

git clone https://github.com/yourusername/CottonDiseaseDetection.git

(or I recommend installing Github Desktop, its alot easier)

**Model Layout**

**Preprocessing**
-convertes the images to grayscale
-applies a Gaussian filter to smooth the images out
-resize images to a standard dimension (256x256 pixels)

**Feature Extraction**
**Edge Detection:** : Use the Canny method to detect edges
**Color Histograms** :Compute histograms for the grayscale values
**Image Segmentation** : Apply Otsu's thresholding method to segment the images

**Classification**
Uses the K-Nearest Neighbors (KNN) algorithm to classify the images based on the extracted features above.
Determine the optimal number of neighbors (K) through cross-validation.

**Evaluation and Metrics**
The model's performance is evaluated on a test set, not seen during the training phase.
Metrics such as accuracy, precision, recall, and F1 score are calculated to assess the effectiveness of the model

Confusion Matric

A confusion matrix is also generated. We use this to evaulate the prediction with the actual results from the dataset.
 
