"""
Modeling Module for Credit Card Fraud Detection
Functions for preprocessing data and building fraud detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            precision_recall_curve, auc, roc_curve, accuracy_score,
                            precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

def save_to_pickle(data, filename):
    """Save data to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nData saved to {filename}")

def load_from_pickle(filename):
    """Load data from a pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"\nData loaded from {filename}")
        return data
    return None

def preprocess_data(df, use_cached=True, cache_file='preprocessed_data.pkl'):
    """
    Preprocess the transaction data for fraud detection modeling.
    
    Parameters:
    - df: DataFrame containing transaction data
    - use_cached: Whether to try loading cached results
    - cache_file: File name for caching results
    
    Returns:
    - Preprocessed DataFrame ready for modeling
    """
    # Try to load cached results first
    if use_cached:
        preprocessed_data = load_from_pickle(cache_file)
        if preprocessed_data is not None:
            return preprocessed_data

    print("\n=== DATA PREPROCESSING ===")

    df_model = df.copy()
    
    # Convert date fields to datetime
    date_columns = ['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']
    for col in date_columns:
        if col in df_model.columns:
            df_model[col] = pd.to_datetime(df_model[col], errors='coerce')
    
    # Extract datetime features
    if 'transactionDateTime' in df_model.columns:
        df_model['txnHour'] = df_model['transactionDateTime'].dt.hour
        df_model['txnDayOfWeek'] = df_model['transactionDateTime'].dt.dayofweek
        df_model['isWeekend'] = df_model['txnDayOfWeek'].isin([5, 6]).astype(int)
        df_model['isNightTime'] = ((df_model['txnHour'] >= 22) | (df_model['txnHour'] < 6)).astype(int)
    
    # Account age at transaction time
    if 'accountOpenDate' in df_model.columns:
        df_model['accountAgeInDays'] = (df_model['transactionDateTime'] - 
                                      df_model['accountOpenDate']).dt.days
    
    # Time since last address change
    if 'dateOfLastAddressChange' in df_model.columns:
        df_model['daysSinceAddressChange'] = (df_model['transactionDateTime'] - 
                                            df_model['dateOfLastAddressChange']).dt.days
    
    # Transaction amount features
    if 'transactionAmount' in df_model.columns and 'creditLimit' in df_model.columns:
        df_model['transactionAmountToLimit'] = (df_model['transactionAmount'] / 
                                              df_model['creditLimit']).replace([np.inf, -np.inf], np.nan)
    
    # Is CVV correct
    if 'cardCVV' in df_model.columns and 'enteredCVV' in df_model.columns:
        df_model['isCVVCorrect'] = (df_model['cardCVV'] == df_model['enteredCVV']).astype(int)
    
    # Convert boolean to integer
    if 'cardPresent' in df_model.columns:
        df_model['cardPresent'] = df_model['cardPresent'].astype(int)
    
    # Encode categorical fields
    categorical_cols = ['merchantCategoryCode', 'posEntryMode', 'posConditionCode', 'transactionType']
    for col in categorical_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype('category')
    
    # Drop unnecessary columns
    drop_cols = ['cardCVV', 'enteredCVV', 'transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']
    df_model = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])

    # Fill missing values
    for col in df_model.select_dtypes(include=['number']).columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    print(f"Data shape after preprocessing: {df_model.shape}")
    
    # Save results before returning
    save_to_pickle(df_model, cache_file)
    return df_model

def build_fraud_model(df, use_cached=True, cache_file='model_results.pkl'):
    """
    Build a machine learning model to predict fraudulent transactions.
    
    Parameters:
    - df: Preprocessed DataFrame
    - use_cached: Whether to try loading cached results
    - cache_file: File name for caching results
    """
    # Try to load cached results first
    if use_cached:
        model_results = load_from_pickle(cache_file)
        if model_results is not None:
            return model_results

    print("\n=== FRAUD DETECTION MODEL TRAINING ===")

    df_model = df.copy()

    if 'isFraud' not in df_model.columns:
        print("Error: Target variable 'isFraud' not found in dataset.")
        return None

    X = df_model.drop('isFraud', axis=1)
    y = df_model['isFraud']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Fraud rate in training set: {y_train.mean() * 100:.2f}%")

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.select_dtypes(include=['number']).columns.tolist()),
            ('cat', categorical_transformer, X.select_dtypes(include=['category']).columns.tolist())
        ]
    )

    # Use SMOTE-Tomek for oversampling and noise reduction
    smote_tomek = SMOTETomek(random_state=42)

    # Random Forest Pipeline with SMOTE-Tomek
    rf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote_tomek),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_results = evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test, "Random Forest")

    # XGBoost Pipeline with scale_pos_weight
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, scale_pos_weight=10, eval_metric='logloss'))
    ])

    xgb_pipeline.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb_pipeline, X_train, X_test, y_train, y_test, "XGBoost")

    # Save results before returning
    results = {
        'random_forest': rf_results,
        'xgboost': xgb_results
    }
    save_to_pickle(results, cache_file)
    return results

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance with various metrics"""

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    return {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc
    }

    
    