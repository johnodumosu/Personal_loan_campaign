# function to train a model and compute train accuracy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt       # matplotlib.pyplot plots data
import seaborn as sns

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score,
                            accuracy_score,
                            precision_recall_curve,
                            roc_auc_score,
                            roc_curve,
                            confusion_matrix,
                            ConfusionMatrixDisplay)

def train(model, X, y):
    """
    Function to train a model and compute accuracy
    
    model: Model not yet trained
    X: train features
    y: train target
    
    """
    
    trained_model = model.fit(X, y)
    preds = trained_model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Train Accuracy: {acc}")
    
    return trained_model

# function to evaluate the model and compute metrics on test data

def evaluate(model, X, y, threshold=0.5):
    """
    Function to evaluate our trained model with different metrics
    
    model: trained model(classifier)
    X: validation features
    y: target in validation data
    
    threshold: value to filter our prediction
    """
    
    pred_proba = model.predict_proba(X)[:, 1]
    pred_class = np.round(pred_proba > threshold) # convert to 0 or 1
    acc = accuracy_score(y, pred_class)
    recall = recall_score(y, pred_class)
    precision = precision_score(y, pred_class)
    f1 = f1_score(y, pred_class)
    
    df = pd.DataFrame({
        "Accuracy": acc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    }, index=[0])
    
    conf = confusion_matrix(y, pred_class, labels=[0,1])
    disp = ConfusionMatrixDisplay(conf, display_labels=[0,1])
    disp.plot()
    plt.show()
    
    return df