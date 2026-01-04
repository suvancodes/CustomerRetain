# Bank Customer Retention Prediction using Deep Learning (ANN)

A Deep Learning web application that predicts whether a bank customer will churn (leave) or stay, using an Artificial Neural Network (ANN) trained on bank customer data.

The application is developed in Python and deployed using Streamlit.

---

## 🔗 Live Demo

https://customerretain-5c84hbynng3ipalktqmkzp.streamlit.app

---

## 📌 Problem Statement

Customer churn is a critical challenge in the banking sector. When customers leave, banks lose revenue and long-term value.

The objective of this project is to predict bank customer churn using a Deep Learning model (ANN) so that high-risk customers can be identified early.

---

## 🧠 Solution Overview

- Used a bank customer churn dataset  
- Performed data preprocessing and feature encoding  
- Designed and trained an Artificial Neural Network (ANN)  
- Implemented the trained model for real-time inference  
- Built an interactive web application using Streamlit  
- Deployed the application on Streamlit Cloud  

---

## 🏦 Dataset Description

The dataset contains bank-specific customer information such as:

- Credit Score  
- Age  
- Geography  
- Gender  
- Tenure with the bank  
- Account Balance  
- Number of bank products  
- Active membership status  

Target Variable:
- Exited  
  - 0 → Customer stays  
  - 1 → Customer leaves  

---

## 🧬 Model Architecture (ANN)

- Input layer with encoded customer features  
- One or more hidden dense layers  
- ReLU activation for hidden layers  
- Sigmoid activation in the output layer  
- Binary cross-entropy loss  
- Adam optimizer  

---

## 🛠️ Tech Stack

- Language: Python  
- Deep Learning: TensorFlow, Keras  
- Data Processing: Pandas, NumPy  
- Web Framework: Streamlit  
- Deployment: Streamlit Cloud  
- Tools: Git, GitHub  

---

## 📂 Project Structure

CustomerRetain/
├── app.py
├── requirements.txt
├── runtime.txt
├── imp.txt
├── notebook/
├── .gitignore
└── README.md

---

## ⚙️ How to Run Locally

1. Clone the repository

git clone https://github.com/suvancodes/CustomerRetain.git  
cd CustomerRetain  

2. Create and activate environment

conda create -n bank-ann python=3.10 -y  
conda activate bank-ann  

3. Install dependencies

pip install -r requirements.txt  

4. Run the application

streamlit run app.py  

---

## 📊 Features

- Deep Learning based churn prediction  
- ANN-powered inference  
- Clean Streamlit user interface  
- Real-time predictions  

---

## 🚀 Future Improvements

- Churn probability score  
- Model explainability (SHAP)  
- Hyperparameter tuning  
- Model versioning  
- Prediction history storage  

---

## 👤 Author

Suvankar Payra  
GitHub: https://github.com/suvancodes  

---

## 📜 License

This project is created for learning, academic, and portfolio purposes.
