#!/usr/bin/env python
# coding: utf-8
"""
Credit Card Fraud Detection using Machine Learning

This project implements multiple machine learning algorithms to detect fraudulent
credit card transactions. It includes data preprocessing, SMOTE for handling
class imbalance, and comprehensive model evaluation.

Algorithms implemented:
- Logistic Regression
- K-Nearest Neighbors
- Random Forest (Gini criterion)
- Random Forest (Entropy criterion)

Requirements:
- Python 3.7+
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib
"""

import pandas as pd
import numpy as np

# importing the data set
df = pd.read_csv("creditcard.csv")

# creating target series
target = df['Class']

# dropping the target variable from the data set
df.drop('Class', axis=1, inplace=True)
print(f"Dataset shape: {df.shape}")

# converting them to numpy arrays
X = np.array(df)
y = np.array(target)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# distribution of the target variable
fraud_count = len(y[y==1])
normal_count = len(y[y==0])
print(f"Fraudulent transactions: {fraud_count}")
print(f"Normal transactions: {normal_count}")

# Check if this is a sample dataset
if fraud_count == 0:
    print("\n" + "="*60)
    print("WARNING: This appears to be a sample dataset!")
    print("="*60)
    print("The dataset contains only normal transactions (no fraud cases).")
    print("This is likely a sample of the full credit card fraud dataset.")
    print("For meaningful fraud detection results, you need the complete dataset.")
    print("="*60)
    print("\nMachine learning algorithms cannot train on single-class data.")
    print("This demonstration will show the data structure and preprocessing steps.")
    print("="*60 + "\n")
    
    # Show data preprocessing steps
    print("DATA PREPROCESSING DEMONSTRATION:")
    print("-" * 40)
    
    # splitting the data set into train and test (75:25)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    print(f"\nFeature statistics:")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(df.columns)}")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    print(df.isnull().sum().sum())
    
    print(f"\nFeature ranges:")
    print(f"Min values: {df.min().min():.4f}")
    print(f"Max values: {df.max().max():.4f}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("This was a demonstration with a sample dataset containing only normal transactions.")
    print("For real fraud detection, you need a dataset with both normal and fraudulent transactions.")
    print("The full credit card fraud dataset typically contains:")
    print("- 284,807 total transactions")
    print("- 492 fraudulent transactions (0.172%)")
    print("- 284,315 normal transactions (99.828%)")
    print("\nTo get the full dataset, you can download it from:")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("="*60)
    
else:
    # Complete analysis code for when we have both classes
    # splitting the data set into train and test (75:25)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # applying SMOTE to oversample the minority class
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=2)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE - X: {X_sm.shape}, y: {y_sm.shape}")
    print(f"After SMOTE - Fraud: {len(y_sm[y_sm==1])}, Normal: {len(y_sm[y_sm==0])}")

    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    from sklearn import metrics

    print("\n" + "="*50)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*50)

    # Logistic Regression
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_sm, y_sm)
    y_logreg = logreg.predict(X_test)
    y_logreg_prob = logreg.predict_proba(X_test)[:,1]

    # Performance metrics evaluation
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_logreg))
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_logreg):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_logreg):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_logreg):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_logreg_prob):.4f}")

    # plotting the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_logreg_prob)
    auc = metrics.roc_auc_score(y_test, y_logreg_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b', label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], 'r-.')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('Receiver Operating Characteristic - Logistic Regression')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

    print("\n" + "="*50)
    print("K-NEAREST NEIGHBORS RESULTS")
    print("="*50)

    # K Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_sm, y_sm)
    y_knn = knn.predict(X_test)
    y_knn_prob = knn.predict_proba(X_test)[:,1]

    # metrics evaluation
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_knn))
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_knn):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_knn):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_knn):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_knn_prob):.4f}")

    # plotting the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_knn_prob)
    auc = metrics.roc_auc_score(y_test, y_knn_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'g', label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], 'r-.')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('Receiver Operating Characteristic - K-Nearest Neighbors')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

    print("\n" + "="*50)
    print("RANDOM FOREST RESULTS")
    print("="*50)

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(random_state=3, n_estimators=100)
    rf.fit(X_sm, y_sm)
    y_rf = rf.predict(X_test)
    y_rf_prob = rf.predict_proba(X_test)[:,1]

    # Performance metrics evaluation
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_rf))
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_rf):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_rf):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_rf):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_rf_prob):.4f}")

    # plotting the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_rf_prob)
    auc = metrics.roc_auc_score(y_test, y_rf_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'm', label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], 'r-.')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('Receiver Operating Characteristic - Random Forest')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

    print("\n" + "="*50)
    print("RANDOM FOREST (ENTROPY) RESULTS")
    print("="*50)

    # Random Forest with entropy criterion
    rf_entropy = RandomForestClassifier(criterion='entropy', random_state=3, n_estimators=100)
    rf_entropy.fit(X_sm, y_sm)
    y_rf_entropy = rf_entropy.predict(X_test)
    y_rf_entropy_prob = rf_entropy.predict_proba(X_test)[:,1]

    # Performance metrics evaluation
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_rf_entropy))
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_rf_entropy):.4f}")
    print(f"Precision: {metrics.precision_score(y_test, y_rf_entropy):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_rf_entropy):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_rf_entropy_prob):.4f}")

    # plotting the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_rf_entropy_prob)
    auc = metrics.roc_auc_score(y_test, y_rf_entropy_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'c', label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], 'r-.')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.title('Receiver Operating Characteristic - Random Forest (Entropy)')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print("All models have been trained and evaluated successfully!")
    print("Check the plots above to compare the ROC curves of different models.")

