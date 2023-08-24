# Fake-News-Detection
An end-to-end machine learning model used in detecting fake news

![Vehicle-Type-Prediction](image.avif)

Predicting news genuinity using machine learning models.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Source and Preprocessing](#data-source-and-preprocessing)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Visualizations](#results-and-visualizations)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation
Install the required packages from the `requirements.txt` file:

## Project Structure
The project is organized as follows:

- `fake.csv/`: Contains fake news data files.
- `true.csv/`: Contains real news data files.
- `FakeNewsDetection.ipynb/`: Holds Jupyter notebooks used for data analysis and exploration.
- `model.pkl/`: Stores trained machine learning models for type prediction.
- `fast.py`: A Python script to make predictions using trained models.
- `vectorizer.pkl`: Tfidfvectorizer model.

## Data Source and Preprocessing
- The dataset is obtained from [Kaggle](https://www.kaggle.com/dataset).
- Preprocessed data by handling missing values and encoding categorical features.

## Model Details
- Trained a GaussianNB,MultinomialNB, and BernoulliNB Classifier.

## Evaluation Metrics
- Evaluated models using Accurazy Score, Confusion Matrix and Precision Score.

## Acknowledgments
- Used the `scikit-learn` library for machine learning models.

## Contact
For questions or feedback, contact me at adefemiadeyanju101@hotmail.com.
