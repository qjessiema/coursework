#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">Modeling the Toronto Bicyle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed – Student ID: 501304501
"""


# In[2]:


# --- Import libraries ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve


# In[3]:


def train_and_evaluate_models(df):
    """
    Trains 3 classification models to predict if a bike will be RECOVERED.
    Returns: results dataframe, roc data dictionary, calibration data dictionary.
    """
    print("--- Starting Machine Learning Pipeline ---")

    # -------------------------------------------------    
    # 1. Filter and define target (STOLEN vs RECOVERED)
    # -------------------------------------------------   
    
    df_ml = df[df['STATUS'].isin(['STOLEN', 'RECOVERED'])].copy()
    # Target: 1 if RECOVERED, 0 if STOLEN
    df_ml['TARGET'] = (df_ml['STATUS'] == 'RECOVERED').astype(int)
    
    print(f"Target Distribution:\n{df_ml['TARGET'].value_counts(normalize=True)}")

    # -------------------------------------------------   
    # 2. Feature selection
    # -------------------------------------------------   
    
    features = ['OCC_HOUR', 'OCC_MONTH', 'OCC_DOW', 'BIKE_TYPE', 'BIKE_COST', 'PREMISES_TYPE']
    X = df_ml[features]
    y = df_ml['TARGET']

    # -------------------------------------------------    
    # 3. Preprocessing
    # -------------------------------------------------   
    
    numeric_features = ['BIKE_COST', 'OCC_HOUR']
    categorical_features = ['OCC_MONTH', 'OCC_DOW', 'BIKE_TYPE', 'PREMISES_TYPE']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # -------------------------------------------------   
    # 4. Define models
    # -------------------------------------------------   
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # -------------------------------------------------   
    # 5. Split
    # -------------------------------------------------   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # -------------------------------------------------       
    # 6. Train and store Results
    # -------------------------------------------------   
    
    results = []
    roc_data = {}
    calibration_data = {}

    for name, model in models.items():
        print(f"Training {name}...")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        res = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        }
        results.append(res)
        
        # ROC Data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, res['ROC AUC'])
        
        # Calibration Data
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        calibration_data[name] = (prob_true, prob_pred)
    
    return pd.DataFrame(results), roc_data, calibration_data

