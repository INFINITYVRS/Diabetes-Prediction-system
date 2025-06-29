# NeuroDiab — Diabetes Prediction System

## Overview

NeuroDiab is an Artificial Neural Network (ANN) powered web application designed to predict the likelihood of diabetes in patients based on clinical parameters. The system uses the PIMA Indians Diabetes Database for model training and includes a web interface for real-time predictions.

## Tech Stack

- Backend: Flask, Python, Keras, TensorFlow
- Frontend: HTML5, CSS3, Bootstrap
- Tools: Keras Tuner, GridSearchCV, Matplotlib

## Features

- Predict diabetes probability using a trained ANN model
- Real-time prediction interface with web form
- ROC and confusion matrix visualization
- Hyperparameter tuning with Keras Tuner and GridSearchCV
- Easily extendable and deployable

## Model Performance

- Accuracy: 80.52%
- AUC-ROC: 0.862

## Installation and Setup
- git clone https://github.com/INFINITYVRS/Diabetes-Prediction-system.git
- cd Diabetes-Prediction-system
- pip install -r requirements.txt
- python app.py
