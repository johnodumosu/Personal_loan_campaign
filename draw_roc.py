import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt       # matplotlib.pyplot plots data
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score,
                            accuracy_score,
                            precision_recall_curve,
                            roc_auc_score,
                            roc_curve,
                            confusion_matrix,
                            ConfusionMatrixDisplay)



def draw_roc_curve(model, X, y):
    """
    model: Trained model
    X = test or validation features
    y = target in test or validation data
    
    """
    preds = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, preds)
    tpr, fpr, thres = roc_curve(y, preds)

    plt.figure(figsize = (7,5))
    plt.plot(tpr, fpr, label = f"Logistic Regression {auc:.2f}")
    plt.plot([0,1], [0,1], "r--" )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc = "lower right")
    plt.show()
    
    return tpr, fpr, thres