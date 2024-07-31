# Gas-Sensor-Data-Classification-and-Anomaly-Detection

This repository contains the implementation of a deep learning model to classify different home activities based on gas sensor readings and to detect anomalies in the data. The project uses a variety of deep learning techniques and frameworks, including TensorFlow/Keras and PyTorch, to build, train, and evaluate the models.

Objective
The goal of this project is to:

Classify different home activities using gas sensor data.
Detect anomalies in the sensor data using an autoencoder model.
Dataset
The dataset used for this project is the Gas Sensors for Home Activity Monitoring Data Set, which can be found on Kaggle: Gas Sensors for Home Activity Monitoring Data Set.

Steps
Data Preprocessing

Download and load the dataset.
Handle missing values.
Merge datasets.
Normalize/standardize the data.
Split the dataset into training and testing sets.
Exploratory Data Analysis (EDA)

Perform detailed EDA.
Visualize sensor readings for different activities.
Classification Task

Implement a deep learning model to classify the different home activities.
Train the model and evaluate its performance using accuracy, precision, recall, and F1-score.
Visualize the model’s performance using confusion matrices and ROC curves.
Anomaly Detection Task

Use an autoencoder model to detect anomalies in the sensor data.
Train the autoencoder on the normal data.
Identify and visualize the anomalies in the test data.
Bonus Part - Hyperparameter Tuning and Framework Comparison

Perform hyperparameter tuning with Keras (TensorFlow backend).
Implement and train a model using PyTorch.
Compare the results of the Keras and PyTorch models.
Results
The Keras model with hyperparameter tuning outperformed the PyTorch model in terms of accuracy and loss.
The autoencoder model successfully detected anomalies in the sensor data.
Installation
To run the code in this repository, you will need to install the following dependencies:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner torch torchvision
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/gas-sensor-data-classification.git
cd gas-sensor-data-classification
Run the Jupyter notebook:
bash
Copy code
jupyter notebook Gas_Sensor_Data_Classification_and_Anamoly_Detection.ipynb
License
This project is licensed under the MIT License - see the LICENSE file for details.

References
Chollet, F. (2018). Deep Learning with Python. Manning Publications.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). TensorFlow: A System for Large-Scale Machine Learning. In 12th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 16) (pp. 265-283).
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems (pp. 8024-8035).
Brownlee, J. (2017). *Machine Learning Mastery with Python: Understand Your Data
