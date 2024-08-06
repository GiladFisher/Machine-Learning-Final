# Machine-Learning-Final
# Stellar Classification Project

This project aims to classify stellar objects using various machine learning models. The project includes data preprocessing, model training, evaluation, and visualization of results. Additionally, the Kolmogorov-Arnold Network (KAN) model is integrated for stellar classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Figures](#figures)
- [Report](#report)
- [Challenges and Techniques](#challenges-and-techniques)
- [References](#references)

## Installation

1. Clone the repository
   
2. Install the required packages

## Usage

1. Open the Jupyter Notebook


2. Run the Notebook


3. View the Report:

   The results and figures are compiled into a PDF report, which can be found at Stellar_Classification_Report.pdf.

## Project Structure

- stellar_classification_properly_commented.ipynb: The main Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and visualization.
- requirements.txt: A list of required Python packages.
- Stellar_Classification_Report.pdf: The final report containing the results and figures.

## Results

The project evaluates several machine learning models, including:

1. Support Vector Classifier (SVC):
   - Accuracy: 0.984
   - Detailed classification report included in the report.

2. Random Forest Classifier:
   - Accuracy: 0.980
   - Detailed classification report included in the report.

3. CatBoost Classifier:
   - Accuracy: 0.982
   - Detailed classification report included in the report.

4. Multi-Layer Perceptron (MLP) Classifier:
   - Accuracy: 0.978
   - Detailed classification report included in the report.

5. Kolmogorov-Arnold Network (KAN):
   - Details of the implementation and evaluation included in the notebook.

## Figures

The notebook includes visualizations of the training process, such as loss curves and other relevant figures. These figures are also included in the PDF report.

## Report

The PDF report Stellar_Classification_Report.pdf contains a comprehensive summary of the results and figures from the project.

## Challenges and Techniques

### Challenges

1. Data Preprocessing:
   - Challenge: Handling missing values and ensuring data consistency.
   - Solution: Used techniques like imputation for missing values and scaling features using StandardScaler.

2. Model Selection:
   - Challenge: Choosing the right model for classification.
   - Solution: Implemented and compared several models including SVC, Random Forest, CatBoost, MLP, and KAN.

3. Hyperparameter Tuning:
   - Challenge: Optimizing model parameters for better performance.
   - Solution: Used techniques like GridSearchCV and RandomizedSearchCV for hyperparameter tuning.

4. Model Evaluation:
   - Challenge: Accurately evaluating model performance.
   - Solution: Used metrics like accuracy, precision, recall, and F1-score to evaluate models.

5. Integration of KAN:
   - Challenge: Implementing and training the Kolmogorov-Arnold Network.
   - Solution: Installed the pykan library and adapted the code for training and evaluation of the KAN model.

### Techniques

1. Data Preprocessing:
   - Used pandas for data manipulation and cleaning.
   - Applied LabelEncoder for encoding target variables.
   - Scaled features using StandardScaler.
  

2. Model Training:
   - Implemented various models using scikit-learn, catboost, and tensorflow.
   - Trained models on the training dataset and evaluated on the test dataset.
  
3. Visualization:
   - Used matplotlib and seaborn for data visualization and plotting training loss curves.

4. Model Evaluation:
   - Calculated and compared performance metrics using scikit-learn's metrics module.

## References

- PyTorch Documentation
- GitHub - KindXiaoming/pykan
- Towards AI - KAN

