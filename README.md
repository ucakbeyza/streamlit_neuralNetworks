Rainfall Prediction Application 

This project is a rainfall prediction application developed using Python and Streamlit. Users can upload datasets to analyze data and make predictions using a Multilayer Perceptron (MLPClassifier) neural network.

Features
* Data Upload: Load CSV or Excel files and view the dataset
* Data Analysis: Summary statistics, filtering, and plotting
* Machine Learning:
  - Data preprocessing with MinMaxScaler
  - 5-Fold & 10-Fold Cross Validation and 66%-34% Train/Test model training
  - GridSearchCV to find the best parameters
  - Confusion Matrix to visualize model performance
  - Neural Network Graph visualization

Usage
1- Run Streamlit
    streamlit run app.py
2- Upload a CSV or Excel dataset
3- Analyze the data and choose a modeling method
4- View prediction results and model performance

Requirements
* Install the necessary libraries:
    pip install streamlit pandas scikit-learn matplotlib networkx
