# Phishing Email Detection

This project aims to enhance email security by leveraging machine learning techniques to detect phishing emails. The notebook demonstrates various steps, including data preprocessing, model training, and evaluation, for identifying phishing emails from a given dataset.

## Project Overview

Phishing attacks are a significant concern in cybersecurity, where malicious actors use deceptive emails to obtain sensitive information from users. This notebook implements a machine learning approach to automate phishing email detection using a dataset of labeled emails.

## Features

- Data preprocessing and feature extraction from email contents
- Model training using machine learning algorithms
- Model evaluation using metrics such as accuracy, precision, recall, and F1-score
- Visualization of results and insights gained from the model
- Potential use of advanced models or techniques to improve detection accuracy

## Dependencies

To run the notebook, you need the following libraries installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- pytorch
- tensorflow
- transformers

## Usage

1.	Clone this repository:
```bash
git clone https://github.com/Abdelila/your-repository.git](https://github.com/AbdelilahNossair/Phishing-Email-Detection.git
```

2.	Navigate to the project directory:
```bash
cd Phishing-Email-Detection
```

3.	Open the Jupyter notebook:
```bash
jupyter notebook Phishing_Email_Detection_Deliverable_Complete.ipynb
```

4.	Follow the steps in the notebook to execute the code and reproduce the results.

## Dataset

The dataset used in this project contains labeled email data for phishing and legitimate emails (9 large curated datasets). This data is preprocessed to extract features such as email content, subject lines, and metadata for classification purposes. You can either use the included dataset or provide your own email dataset.

## Model Training

The notebook includes the following steps:

1.	Data Preprocessing: This involves cleaning the data and extracting relevant features from the emails (too good to be true sentences, special characters ratio, URL count).
2.	Model Selection: A variety of machine learning models are evaluated, including:
	•	Logistic Regression
	•	Gradient Boosting Classifier
	•	Random Forest
	•	DistilBERT (deep learning pre-trained model based on the transformers architecture
3.	Model Evaluation: The models are evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. Cross-validation is applied to prevent overfitting.

## Results

The notebook provides insights into the performance of the models and interprets the results.


## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

-	Special thanks to Deloitte Morocco Cyber Center for providing guidance during the project.
- The project was part of the internship titled “Leveraging AI to Enhance Email Security and Phishing Detection.”
