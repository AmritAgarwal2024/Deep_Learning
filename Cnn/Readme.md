# MNIST Handwritten Digit Recognizer using CNN

## **Contributors**
- Amrit Agarwal (055004)
- Oishik Banerjee (055028)

**Group No:** 23

---

## **Problem Statement**
Handwritten digit recognition is a vital task in numerous real-world applications, such as postal sorting, bank check processing, and automated form digitization. Traditional methods often face challenges due to variations in handwriting styles, leading to inaccuracies. This project focuses on developing a Deep Convolutional Neural Network (CNN) model to enhance the accuracy of handwritten digit classification. By utilizing deep learning techniques, the model improves pattern detection and generalization, making it highly effective for deployment in automated handwriting recognition systems.

---

## **Abstract**
This project develops a handwritten digit classification system using a Deep Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The architecture employs convolutional layers to capture spatial patterns in digit images and fully connected layers for final classification. Trained on labeled datasets, the model harnesses deep learning's pattern recognition strengths to address handwriting style variations, enhancing prediction reliability.

### **Key Features**
- Convolutional operations for automated feature extraction from raw pixel data.
- Pooling layers to reduce spatial dimensions while preserving critical features.
- Activation functions (ReLU) introducing non-linearity for complex pattern detection.
- Dropout layers preventing overfitting during training.

Experimental results demonstrate the model achieves high classification accuracy, validating its effectiveness for real-world deployment. The system shows particular promise for automated document processing applications like bank check verification and postal sorting.

---

## **Project Structure**
1. Importing Libraries
2. Preparing the Dataset
3. Model Building
4. Model Fitting
5. Model Analysis
6. Predicting Using Test Data

---

## **Data Analysis**

### **Library Importation**
- TensorFlow v2: Google's open-source machine learning framework.
- Keras: A neural network library that operates on top of TensorFlow, simplifying deep learning model development.

### **Dataset Preparation**
The MNIST Handwritten Digit Recognition dataset was used. Key preprocessing steps:
1. **Normalization:** Pixel values ranging from 0 to 255 were scaled to [0,1].
2. **Reshaping:** Pixel arrays were reshaped into matrices of dimensions (28, 28, 1).
3. **Encoding:** Labels were converted into one-hot encoded vectors.
4. **Train-Test Split:** Data was divided into training and validation sets.

A countplot confirmed a balanced distribution of digit classes, and no missing values were found.

---

## **Model Building**

### **Architecture**
The project implemented a Deep Convolutional Neural Network (CNN) based on the LeNet-5 architecture:

### **Key Features**
- **Data Augmentation:** Techniques like rotation, cropping, flipping, and zooming were applied to improve generalization.
- **Optimization Strategy:** RMSProp optimizer was used for stable convergence, while ReduceLROnPlateau dynamically adjusted the learning rate.

---

## **Model Training**

### **Environment**
Training was conducted on Kaggle’s GPU-enabled environment for faster computation.

### **Performance Monitoring**
- Loss and accuracy metrics were tracked across epochs.
- Training and validation losses were monitored to ensure proper learning without overfitting.

---

## **Model Analysis**

### **Evaluation Metrics**
- The learning curve showed that training and validation losses decreased over time, indicating successful training.
- A confusion matrix revealed that the model performed well across most digit classes, with some misclassifications.

---

## **Predictions**
The trained model was used to classify the test dataset:
- Predictions were stored in a CSV file for submission.
- Results aligned well with validation accuracy.

---

## **Managerial Insights**

1. **Scalability & Adaptability:** The model can be extended to recognize characters in multiple languages.
2. **Cost-Effectiveness:** Reduces manual transcription costs by automating digit recognition.
3. **Automation Potential:** Ideal for banking, postal services, and form digitization systems.
4. **Error Handling:** Misclassifications can be addressed through active learning and periodic retraining.
5. **Infrastructure Considerations:** Requires GPU resources; cloud-based AI services can optimize deployment costs.

---



