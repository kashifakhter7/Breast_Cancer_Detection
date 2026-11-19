import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
import warnings
warnings.filterwarnings('ignore')

"""Data Collection & Processing"""

#loading data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

print(breast_cancer_dataset)

#loading data  to pandas datafram
data_frame =pd.DataFrame(breast_cancer_dataset.data, columns= breast_cancer_dataset.feature_names)

#print 5 rows of data frame
data_frame.head()

data_frame.columns

#adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target

#print last 5 rows of data set
data_frame.tail()

#no. of  rows and column in the dataset
data_frame.shape

#getting some information about the dataset
data_frame.info()

# checking for missing values
data_frame.isnull().sum()

#Statical measures of dataset
data_frame.describe()

#checking the distribution of target variable
data_frame['label'].value_counts()

"""1---> Benign
0---> Malignant
"""

data_frame.groupby('label').mean()

"""Separating the features and Target"""

X= data_frame.drop(columns='label',axis=1) #droping column mention axis value is 1 and for row axis = 0
Y= data_frame['label']

print(X)

print(Y)

#Splitting the data into Training data & Testing data


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#Model Training

# Define models
models = {
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Voting Classifier': VotingClassifier(estimators=[('svm', SVC()), ('rf', RandomForestClassifier()),
     ('lr', LogisticRegression())]),
    'XGBoost': XGBClassifier()

}

# Train and evaluate models
accuracies = {}
confusion_matrixex ={}
precision_scores_value = {}
for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(Y_test, y_pred)
    confusion_matrixex[name]=confusion_matrix(Y_test,y_pred)
    precision_scores_value[name] =precision_score(Y_test,y_pred)


#Model Evaluation


# Print accuracy of each model
print("Model Accuracies:")
for model, acc in accuracies.items():
    print(f"{model}: {acc:.4f}")
print("Confusion matrix:")
for model, cm in confusion_matrixex.items():
    print(f"{model}:")
    print(cm)
print("Precision score:")
for model, ps in precision_scores_value.items():
    print(f"{model}: {ps:.4f}")

# Find the best model
best_model = max(accuracies, key=lambda model: (accuracies[model], precision_scores_value[model]))
print(f"The best model is {best_model} with an accuracy of {accuracies[best_model]:.4f}")

# Find the best model
print("max accuracy", max(accuracies, key=accuracies.get), "with accuracy", max(accuracies.values()))
print("max precision", max(precision_scores_value, key=precision_scores_value.get), "with precision", max(precision_scores_value.values()))

# Finding the best model

model_accuracies = {
    "Support Vector Machine": 0.9035,
    "Random Forest": 0.9386,
    "Logistic Regression": 0.9298,
    "K-Nearest Neighbors": 0.9123,
    "Decision Tree": 0.9123,
    "Naive Bayes": 0.9386,
    "AdaBoost": 0.9561,
    "Gradient Boosting": 0.9298,
    "Extra Trees": 0.9386,
    "Voting Classifier": 0.9386,
    "XGBoost": 0.9386
}

precision_scores = {
    "Support Vector Machine": 0.9028,
    "Random Forest": 0.9559,
    "Logistic Regression": 0.9420,
    "K-Nearest Neighbors": 0.9275,
    "Decision Tree": 0.9683,
    "Naive Bayes": 0.9559,
    "AdaBoost": 0.9571,
    "Gradient Boosting": 0.9420,
    "Extra Trees": 0.9559,
    "Voting Classifier": 0.9429,
    "XGBoost": 0.9697
}

confusion_matrices = {
    "Support Vector Machine": [[38, 7], [4, 65]],
    "Random Forest": [[42, 3], [4, 65]],
    "Logistic Regression": [[41, 4], [4, 65]],
    "K-Nearest Neighbors": [[40, 5], [5, 64]],
    "Decision Tree": [[43, 2], [8, 61]],
    "Naive Bayes": [[42, 3], [4, 65]],
    "AdaBoost": [[42, 3], [2, 67]],
    "Gradient Boosting": [[41, 4], [4, 65]],
    "Extra Trees": [[42, 3], [4, 65]],
    "Voting Classifier": [[41, 4], [3, 66]],
    "XGBoost": [[43, 2], [5, 64]]
}

# Function to analyze best model
def find_best_model():
    # Step 1: Find the model with the highest accuracy
    best_accuracy_model = max(model_accuracies, key=model_accuracies.get)

    # Step 2: Check precision for the best accuracy model
    best_accuracy = model_accuracies[best_accuracy_model]
    best_precision = precision_scores[best_accuracy_model]

    # Step 3: Analyze the confusion matrix (False Negatives and False Positives)
    cm = confusion_matrices[best_accuracy_model]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]

    print(f"Best Model Based on Accuracy: {best_accuracy_model}")
    print(f" - Accuracy: {best_accuracy:.4f}")
    print(f" - Precision: {best_precision:.4f}")
    print(f" - False Positives: {false_positives}")
    print(f" - False Negatives: {false_negatives}")

    return best_accuracy_model

# Find the best model
best_model = find_best_model()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_metrics(accuracies, confusion_matrices, precision_scores):
    models = list(accuracies.keys())

    # Plot Model Accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(models, accuracies.values(), color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracies")
    plt.ylim(0.85, 1)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Plot Confusion Matrices
    for model, cm in confusion_matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model}")
        plt.show()

    # Plot Precision Scores
    plt.figure(figsize=(12, 6))
    plt.bar(models, precision_scores.values(), color='lightcoral')
    plt.xlabel("Models")
    plt.ylabel("Precision Score")
    plt.title("Model Precision Scores")
    plt.ylim(0.85, 1)
    plt.xticks(rotation=45, ha='right')
    plt.show()

def find_best_model(accuracies, precision_scores, confusion_matrices):
    best_model = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model]
    best_precision = precision_scores[best_model]
    cm = confusion_matrices[best_model]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]

    print(f"Best Model Based on Accuracy: {best_model}")
    print(f" - Accuracy: {best_accuracy:.4f}")
    print(f" - Precision: {best_precision:.4f}")
    print(f" - False Positives: {false_positives}")
    print(f" - False Negatives: {false_negatives}")

    return best_model

# data
model_accuracies = {
    "Support Vector Machine": 0.9035,
    "Random Forest": 0.9386,
    "Logistic Regression": 0.9298,
    "K-Nearest Neighbors": 0.9123,
    "Decision Tree": 0.9123,
    "Naive Bayes": 0.9386,
    "AdaBoost": 0.9561,
    "Gradient Boosting": 0.9298,
    "Extra Trees": 0.9386,
    "Voting Classifier": 0.9386,
    "XGBoost": 0.9386
}

precision_scores = {
    "Support Vector Machine": 0.9028,
    "Random Forest": 0.9559,
    "Logistic Regression": 0.9420,
    "K-Nearest Neighbors": 0.9275,
    "Decision Tree": 0.9683,
    "Naive Bayes": 0.9559,
    "AdaBoost": 0.9571,
    "Gradient Boosting": 0.9420,
    "Extra Trees": 0.9559,
    "Voting Classifier": 0.9429,
    "XGBoost": 0.9697
}

confusion_matrices = {
    "Support Vector Machine": np.array([[38, 7], [4, 65]]),
    "Random Forest": np.array([[42, 3], [4, 65]]),
    "Logistic Regression": np.array([[41, 4], [4, 65]]),
    "K-Nearest Neighbors": np.array([[40, 5], [5, 64]]),
    "Decision Tree": np.array([[43, 2], [8, 61]]),
    "Naive Bayes": np.array([[42, 3], [4, 65]]),
    "AdaBoost": np.array([[42, 3], [2, 67]]),
    "Gradient Boosting": np.array([[41, 4], [4, 65]]),
    "Extra Trees": np.array([[42, 3], [4, 65]]),
    "Voting Classifier": np.array([[41, 4], [3, 66]]),
    "XGBoost": np.array([[43, 2], [5, 64]])
}

# Run functions
plot_model_metrics(model_accuracies, confusion_matrices, precision_scores)
best_model = find_best_model(model_accuracies, precision_scores, confusion_matrices)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_metrics(accuracies, confusion_matrices, precision_scores, recall_scores):
    models = list(accuracies.keys())

    # Plot Model Accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(models, accuracies.values(), color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracies")
    plt.ylim(0.85, 1)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Plot Precision Scores
    plt.figure(figsize=(12, 6))
    plt.bar(models, precision_scores.values(), color='lightcoral')
    plt.xlabel("Models")
    plt.ylabel("Precision Score")
    plt.title("Model Precision Scores")
    plt.ylim(0.85, 1)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Plot Recall Scores
    plt.figure(figsize=(12, 6))
    plt.bar(models, recall_scores.values(), color='lightgreen')
    plt.xlabel("Models")
    plt.ylabel("Recall Score")
    plt.title("Model Recall Scores")
    plt.ylim(0.85, 1)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Plot Confusion Matrices
    for model, cm in confusion_matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model}")
        plt.show()

def find_best_model(accuracies, precision_scores, recall_scores, confusion_matrices):
    best_model = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model]
    best_precision = precision_scores[best_model]
    best_recall = recall_scores[best_model]
    cm = confusion_matrices[best_model]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]

    print(f"Best Model Based on Accuracy: {best_model}")
    print(f" - Accuracy: {best_accuracy:.4f}")
    print(f" - Precision: {best_precision:.4f}")
    print(f" - Recall: {best_recall:.4f}")
    print(f" - False Positives: {false_positives}")
    print(f" - False Negatives: {false_negatives}")

    return best_model

#data
model_accuracies = {
    "Support Vector Machine": 0.9035,
    "Random Forest": 0.9386,
    "Logistic Regression": 0.9298,
    "K-Nearest Neighbors": 0.9123,
    "Decision Tree": 0.9123,
    "Naive Bayes": 0.9386,
    "AdaBoost": 0.9561,
    "Gradient Boosting": 0.9298,
    "Extra Trees": 0.9386,
    "Voting Classifier": 0.9386,
    "XGBoost": 0.9386
}

precision_scores = {
    "Support Vector Machine": 0.9028,
    "Random Forest": 0.9559,
    "Logistic Regression": 0.9420,
    "K-Nearest Neighbors": 0.9275,
    "Decision Tree": 0.9683,
    "Naive Bayes": 0.9559,
    "AdaBoost": 0.9571,
    "Gradient Boosting": 0.9420,
    "Extra Trees": 0.9559,
    "Voting Classifier": 0.9429,
    "XGBoost": 0.9697
}

recall_scores = {
    "Support Vector Machine": 0.9063,
    "Random Forest": 0.9420,
    "Logistic Regression": 0.9420,
    "K-Nearest Neighbors": 0.9275,
    "Decision Tree": 0.8841,
    "Naive Bayes": 0.9420,
    "AdaBoost": 0.9710,
    "Gradient Boosting": 0.9420,
    "Extra Trees": 0.9420,
    "Voting Classifier": 0.9571,
    "XGBoost": 0.9275
}

confusion_matrices = {
    "Support Vector Machine": np.array([[38, 7], [4, 65]]),
    "Random Forest": np.array([[42, 3], [4, 65]]),
    "Logistic Regression": np.array([[41, 4], [4, 65]]),
    "K-Nearest Neighbors": np.array([[40, 5], [5, 64]]),
    "Decision Tree": np.array([[43, 2], [8, 61]]),
    "Naive Bayes": np.array([[42, 3], [4, 65]]),
    "AdaBoost": np.array([[42, 3], [2, 67]]),
    "Gradient Boosting": np.array([[41, 4], [4, 65]]),
    "Extra Trees": np.array([[42, 3], [4, 65]]),
    "Voting Classifier": np.array([[41, 4], [3, 66]]),
    "XGBoost": np.array([[43, 2], [5, 64]])
}

# Run functions
plot_model_metrics(model_accuracies, confusion_matrices, precision_scores, recall_scores)
best_model = find_best_model(model_accuracies, precision_scores, recall_scores, confusion_matrices)

AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, Y_train)
y_pred = AdaBoost.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

input_data =(17.90,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = AdaBoost.predict(input_data_reshaped)
print(prediction)

if(prediction[0]== 0):
  print('The Breast cancer is Malignant')
else :
    print('The Breast cancer is Benign')

import pickle

pickle.dump(model, open('model.pkl', 'wb'))

