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

    streamlit run main.py
    
2- Upload a CSV or Excel dataset

3- Analyze the data and choose a modeling method

4- View prediction results and model performance

Requirements
* Install the necessary libraries:
  
    pip install streamlit pandas scikit-learn matplotlib
  
<img width="1466" alt="Screenshot 2025-02-10 at 16 04 09" src="https://github.com/user-attachments/assets/4634a646-459e-4ebd-a915-56c4169e3803" />

<img width="1470" alt="Screenshot 2025-02-10 at 16 04 25" src="https://github.com/user-attachments/assets/7a885d68-7c9f-4f73-99b3-26eb9c7b0dbc" />

<img width="1470" alt="Screenshot 2025-02-10 at 16 04 54" src="https://github.com/user-attachments/assets/4097f939-c628-4794-8fc2-a9379ef14ced" />

<img width="1470" alt="Screenshot 2025-02-10 at 16 06 03" src="https://github.com/user-attachments/assets/906b0d9b-8b4f-4ca7-b76e-17851fd9bdc0" />
