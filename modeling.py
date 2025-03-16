"""
Enhanced Modeling Module for Credit Card Fraud Detection

This module provides a comprehensive approach to fraud detection modeling with:
1. Baseline model evaluation with minimal preprocessing
2. Feature engineering and selection
3. Multiple sampling techniques (SMOTE, undersampling, etc.)
4. Hyperparameter tuning with cross-validation
5. Model comparison and ensemble methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import time
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, auc, roc_curve, accuracy_score,
                           precision_score, recall_score, f1_score, average_precision_score)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
import xgboost as xgb

# For displaying in both notebook and script environments
def display(df):
    """Simple function to display a dataframe as text in non-notebook environments"""
    if isinstance(df, pd.DataFrame):
        if len(df) > 10:
            print(df.head(10))
            print(f"... {len(df) - 10} more rows ...")
        else:
            print(df)
    else:
        print(df)

def save_to_pickle(data, filename):
    """Save data to a pickle file in the appropriate directory"""
    # Determine the appropriate subdirectory based on filename
    if 'model' in filename.lower():
        subdir = 'models'
    elif 'feature' in filename.lower():
        subdir = 'data'
    else:
        subdir = 'data'
    
    # Create full path
    full_path = os.path.join('results', subdir, filename)
    
    # Save the file
    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else '.', exist_ok=True)
    with open(full_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {full_path}")

def load_from_pickle(filename):
    """Load data from a pickle file in the appropriate directory"""
    # Check multiple possible locations
    possible_paths = [
        filename,  # Original path
        os.path.join('results', 'models', filename),
        os.path.join('results', 'data', filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"Data loaded from {path}")
            return data
    return None

def evaluate_classifier(model, X_test, y_test, model_name='Model', threshold=0.5, plot=True):
    """
    Evaluate a classifier on test data.
    
    Parameters:
    - model: Trained classifier
    - X_test: Test features
    - y_test: Test target
    - model_name: Name of the model for plotting
    - threshold: Classification threshold
    - plot: Whether to generate plots
    
    Returns:
    - Dictionary of evaluation metrics
    """
    # Get predictions
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    except:
        # Some models don't have predict_proba
        y_pred = model.predict(X_test)
        y_proba = None
    
    # Calculate metrics
    results = {}
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    
    # Calculate AUC if probabilities are available
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
        results['pr_auc'] = average_precision_score(y_test, y_proba)
    else:
        results['roc_auc'] = None
        results['pr_auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    
    # Print results
    print(f"\nEvaluation of {model_name}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    if results['roc_auc'] is not None:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"PR AUC: {results['pr_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot results if requested
    if plot and y_proba is not None:
        # ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {results["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('results', 'plots', f'{model_name.replace(" ", "_").lower()}_roc.png'))
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, marker='.', label=f'{model_name} (AUC = {results["pr_auc"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('results', 'plots', f'{model_name.replace(" ", "_").lower()}_pr.png'))
        plt.close()
        
        # Confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'plots', f'{model_name.replace(" ", "_").lower()}_cm.png'))
        plt.close()
    
    return results

def create_baseline_model(df, model_type='logistic', cache_dir='models'):
    """
    Create a baseline model with minimal preprocessing.
    
    Parameters:
    - df: Input DataFrame
    - model_type: Type of model ('logistic', 'rf', 'xgb')
    - cache_dir: Directory to cache results
    
    Returns:
    - Dictionary of results
    """
    print("\n===== BASELINE MODEL EVALUATION =====")
    print("Creating baseline model with minimal preprocessing")
    
    # Check for cached results
    cache_file = os.path.join(cache_dir, f'baseline_{model_type}_results.pkl')
    if os.path.exists(cache_file):
        return load_from_pickle(cache_file)
    
    # Create a copy of the data
    df_copy = df.copy()
    
    # Basic preprocessing - handle NA values
    numeric_cols = df_copy.select_dtypes(include=['number']).columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    
    # Fill missing values
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    for col in categorical_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode().iloc[0])
    
    # Prepare features and target
    X = df_copy.drop('isFraud', axis=1)
    y = df_copy['isFraud']
    
    # Convert categorical columns to string to avoid errors
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].astype(str)
    
    # Drop columns that might cause issues
    cols_to_drop = ['transactionDateTime', 'merchantName', 'accountNumber', 'customerId']
    X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with basic preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.select_dtypes(include=['number']).columns),
            ('cat', categorical_transformer, X.select_dtypes(include=['object', 'category']).columns)
        ]
    )
    
    # Choose classifier based on model_type
    if model_type == 'logistic':
        classifier = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    elif model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    elif model_type == 'xgb':
        classifier = xgb.XGBClassifier(random_state=42, scale_pos_weight=10, eval_metric='logloss')
    else:
        raise ValueError("model_type must be one of 'logistic', 'rf', or 'xgb'")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Train model
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Model training time: {training_time:.2f} seconds")
    
    # Evaluate model
    eval_results = evaluate_classifier(pipeline, X_test, y_test, f"Baseline {model_type}")
    
    # Save results
    results = {
        'model': pipeline,
        'evaluation': eval_results,
        'training_time': training_time,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'fraud_rate': y.mean() * 100
    }
    
    save_to_pickle(results, cache_file)
    return results

def feature_engineering(df, cache_dir='data'):
    """
    Perform feature engineering on the transaction data.
    
    Parameters:
    - df: Input DataFrame
    - cache_dir: Directory to cache results
    
    Returns:
    - DataFrame with engineered features
    """
    print("\n===== FEATURE ENGINEERING =====")
    
    # Check for cached results
    cache_file = os.path.join(cache_dir, 'engineered_features.pkl')
    if os.path.exists(cache_file):
        return load_from_pickle(cache_file)
    
    # Create a copy
    df_eng = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']
    for col in date_columns:
        if col in df_eng.columns:
            df_eng[col] = pd.to_datetime(df_eng[col], errors='coerce')
    
    print("Extracting temporal features...")
    # 1. Time-based features
    if 'transactionDateTime' in df_eng.columns:
        # Hour of day
        df_eng['txn_hour'] = df_eng['transactionDateTime'].dt.hour
        # Day of week
        df_eng['txn_day_of_week'] = df_eng['transactionDateTime'].dt.dayofweek
        # Weekend flag
        df_eng['is_weekend'] = df_eng['txn_day_of_week'].isin([5, 6]).astype(int)
        # Night time flag (10 PM to 6 AM)
        df_eng['is_night'] = ((df_eng['txn_hour'] >= 22) | (df_eng['txn_hour'] < 6)).astype(int)
        # Month
        df_eng['txn_month'] = df_eng['transactionDateTime'].dt.month
        # Day of month
        df_eng['txn_day'] = df_eng['transactionDateTime'].dt.day
        # Is end of month (last 5 days)
        df_eng['is_end_of_month'] = (df_eng['txn_day'] >= 25).astype(int)
    
    print("Creating account-related features...")
    # 2. Account age and time since address change
    if 'accountOpenDate' in df_eng.columns:
        df_eng['account_age_days'] = (df_eng['transactionDateTime'] - df_eng['accountOpenDate']).dt.days
        # Bin account age
        df_eng['account_age_bin'] = pd.cut(
            df_eng['account_age_days'], 
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['<1mo', '1-3mo', '3-6mo', '6-12mo', '>12mo']
        )
    
    if 'dateOfLastAddressChange' in df_eng.columns:
        df_eng['days_since_address_change'] = (df_eng['transactionDateTime'] - df_eng['dateOfLastAddressChange']).dt.days
        # Recent address change flag (last 30 days)
        df_eng['recent_address_change'] = (df_eng['days_since_address_change'] <= 30).astype(int)
    
    print("Creating transaction-related features...")
    # 3. Transaction amount features
    if 'transactionAmount' in df_eng.columns:
        # Transaction amount bins
        df_eng['amount_bin'] = pd.cut(
            df_eng['transactionAmount'], 
            bins=[0, 10, 25, 50, 100, 250, 500, 1000, float('inf')],
            labels=['$0-10', '$10-25', '$25-50', '$50-100', '$100-250', '$250-500', '$500-1K', '$1K+']
        )
        
        # Transaction amount to credit limit ratio
        if 'creditLimit' in df_eng.columns:
            df_eng['amount_to_limit_ratio'] = (df_eng['transactionAmount'] / df_eng['creditLimit']).replace([np.inf, -np.inf], 1)
            # High ratio flag
            df_eng['high_amount_to_limit'] = (df_eng['amount_to_limit_ratio'] > 0.5).astype(int)
        
        # Available money ratio
        if 'availableMoney' in df_eng.columns:
            df_eng['amount_to_available_ratio'] = (df_eng['transactionAmount'] / df_eng['availableMoney']).replace([np.inf, -np.inf], 1)
    
    print("Creating behavioral features...")
    # 4. Behavioral flags
    # Card verification
    if 'cardCVV' in df_eng.columns and 'enteredCVV' in df_eng.columns:
        df_eng['cvv_match'] = (df_eng['cardCVV'] == df_eng['enteredCVV']).astype(int)
    
    # Card present vs online
    if 'cardPresent' in df_eng.columns:
        df_eng['card_present'] = df_eng['cardPresent'].astype(int)
    
    # Foreign transaction
    if 'merchantCountryCode' in df_eng.columns:
        df_eng['is_foreign'] = (~df_eng['merchantCountryCode'].isin(['US', 'USA'])).astype(int)
    
    # 5. Merchant category features
    if 'merchantCategoryCode' in df_eng.columns:
        # Convert to category
        df_eng['merchantCategoryCode'] = df_eng['merchantCategoryCode'].astype('category')
        
        # Group similar categories (optional)
        high_risk_categories = ['online_retail', 'online_gifts', 'online_subscriptions', 'mobileapps']
        df_eng['is_high_risk_category'] = df_eng['merchantCategoryCode'].isin(high_risk_categories).astype(int)
    
    print("Creating velocity features...")
    # 6. Velocity features (transaction frequency)
    # Group by account
    if 'accountNumber' in df_eng.columns:
        # Sort by account and time
        df_eng = df_eng.sort_values(['accountNumber', 'transactionDateTime'])
        
        # Initialize velocity columns for different time windows
        for window in [1, 7, 30]:  # 1 day, 7 days, 30 days
            df_eng[f'txn_count_{window}d'] = 0
            df_eng[f'txn_amount_{window}d'] = 0.0
        
        # Calculate transactions per account within time windows
        for account, group in df_eng.groupby('accountNumber'):
            # Sort by date
            group = group.sort_values('transactionDateTime')
            
            # For each transaction
            for i, row in group.iterrows():
                for window in [1, 7, 30]:
                    # Calculate window start
                    window_start = row['transactionDateTime'] - timedelta(days=window)
                    
                    # Find transactions in window
                    in_window = group[
                        (group['transactionDateTime'] >= window_start) & 
                        (group['transactionDateTime'] < row['transactionDateTime'])
                    ]
                    
                    # Update counts
                    df_eng.at[i, f'txn_count_{window}d'] = len(in_window)
                    df_eng.at[i, f'txn_amount_{window}d'] = in_window['transactionAmount'].sum()
        
        # Normalize velocity features
        for window in [1, 7, 30]:
            # Avoid division by zero
            count_mean = df_eng[f'txn_count_{window}d'].mean()
            amount_mean = df_eng[f'txn_amount_{window}d'].mean()
            
            if count_mean > 0:
                df_eng[f'txn_count_{window}d_norm'] = df_eng[f'txn_count_{window}d'] / count_mean
            
            if amount_mean > 0:
                df_eng[f'txn_amount_{window}d_norm'] = df_eng[f'txn_amount_{window}d'] / amount_mean
        
        # High velocity flags
        for window in [1, 7, 30]:
            df_eng[f'high_txn_count_{window}d'] = (df_eng[f'txn_count_{window}d'] > 
                                                 df_eng[f'txn_count_{window}d'].quantile(0.95)).astype(int)
            df_eng[f'high_txn_amount_{window}d'] = (df_eng[f'txn_amount_{window}d'] > 
                                                   df_eng[f'txn_amount_{window}d'].quantile(0.95)).astype(int)
    
    print("Creating recurrence features...")
    # 7. Recurrence patterns
    # Group by account, merchant, amount
    recurrence_groups = df_eng.groupby(['accountNumber', 'merchantName', 'transactionAmount']).size()
    recurrence_df = recurrence_groups.reset_index(name='recurrence_count')
    
    # Merge back
    df_eng = df_eng.merge(
        recurrence_df, 
        on=['accountNumber', 'merchantName', 'transactionAmount'],
        how='left'
    )
    
    # Is recurring flag
    df_eng['is_recurring'] = (df_eng['recurrence_count'] > 1).astype(int)
    
    # Drop temporary columns
    cols_to_drop = ['accountOpenDate', 'dateOfLastAddressChange']
    df_eng = df_eng.drop([col for col in cols_to_drop if col in df_eng.columns], axis=1)
    
    # Fill missing values
    numeric_cols = df_eng.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_eng[col] = df_eng[col].fillna(df_eng[col].median())
    
    categorical_cols = df_eng.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df_eng[col] = df_eng[col].fillna(df_eng[col].mode().iloc[0] if len(df_eng[col].mode()) > 0 else "unknown")
    
    print(f"Feature engineering complete. New shape: {df_eng.shape}")
    
    # Save engineered features
    save_to_pickle(df_eng, cache_file)
    return df_eng

def feature_selection(X, y, method='forest', k=20, cache_dir='data'):
    """
    Perform feature selection using various methods.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - method: Selection method ('forest', 'mutual_info', 'rfe')
    - k: Number of features to select
    - cache_dir: Directory to cache results
    
    Returns:
    - List of selected feature names
    """
    print(f"\n===== FEATURE SELECTION: {method.upper()} =====")
    
    # Check for cached results
    cache_file = os.path.join(cache_dir, f'feature_selection_{method}_{k}.pkl')
    if os.path.exists(cache_file):
        try:
            cached_features = load_from_pickle(cache_file)
            # Verify that the cached features exist in the current dataframe
            valid_features = [f for f in cached_features if f in X.columns]
            if len(valid_features) > 0:
                print(f"Using {len(valid_features)} valid cached features out of {len(cached_features)}")
                return valid_features
            else:
                print("Cached features don't match current dataframe columns. Recomputing...")
        except Exception as e:
            print(f"Error loading cached features: {str(e)}. Recomputing...")
    
    try:
        # Convert categorical columns to string
        X_proc = X.copy()
        for col in X_proc.select_dtypes(include=['category']).columns:
            X_proc[col] = X_proc[col].astype(str)
        
        # Create preprocessor for categorical and numeric features
        numeric_features = X_proc.select_dtypes(include=['number']).columns
        categorical_features = X_proc.select_dtypes(include=['object']).columns
        
        # Ensure we have features to work with
        if len(numeric_features) == 0 and len(categorical_features) == 0:
            print("Warning: No suitable features found for selection")
            return []
        
        transformers = []
        if len(numeric_features) > 0:
            transformers.append(('num', StandardScaler(), numeric_features))
        if len(categorical_features) > 0:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Apply preprocessing
        X_transformed = preprocessor.fit_transform(X_proc)
        
        # Get feature names after preprocessing
        feature_names = []
        if len(numeric_features) > 0:
            feature_names.extend(list(numeric_features))
        
        if len(categorical_features) > 0:
            try:
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(list(cat_feature_names))
            except Exception as e:
                print(f"Warning: Could not get categorical feature names: {str(e)}")
                # Create generic feature names for categorical features
                cat_count = X_transformed.shape[1] - len(numeric_features)
                feature_names.extend([f'cat_feature_{i}' for i in range(cat_count)])
        
        # Ensure k is not larger than the number of features
        k = min(k, X_transformed.shape[1])
        
        # Feature selection method
        if method == 'forest':
            # Random Forest feature importance
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                max_features=k
            )
            selector.fit(X_transformed, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'mutual_info':
            # Mutual information
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_transformed, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'rfe':
            # Recursive feature elimination
            selector = RFE(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                n_features_to_select=k
            )
            selector.fit(X_transformed, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            
        else:
            raise ValueError("method must be one of 'forest', 'mutual_info', or 'rfe'")
        
        # Get selected feature names
        if len(selected_indices) > 0 and len(feature_names) >= max(selected_indices) + 1:
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            print(f"Warning: Feature selection indices out of range. Using top {k} features.")
            selected_features = feature_names[:k] if len(feature_names) > 0 else []
        
        print(f"Selected {len(selected_features)} features:")
        for feature in selected_features[:10]:
            print(f"- {feature}")
        if len(selected_features) > 10:
            print(f"- ... and {len(selected_features) - 10} more")
        
        # Save results
        save_to_pickle(selected_features, cache_file)
        return selected_features
        
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        print("Returning empty feature list")
        return []

def train_models_with_sampling(X, y, sampling_methods=None, models=None, cache_dir='models'):
    """
    Train models with various sampling techniques.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - sampling_methods: List of sampling methods to try
    - models: List of models to try
    - cache_dir: Directory to cache results
    
    Returns:
    - Dictionary of results
    """
    print("\n===== MODEL TRAINING WITH SAMPLING TECHNIQUES =====")
    
    # Default sampling methods if none provided
    if sampling_methods is None:
        sampling_methods = [
            ('none', None),
            ('smote', SMOTE(random_state=42)),
            ('adasyn', ADASYN(random_state=42)),
            ('random_under', RandomUnderSampler(random_state=42)),
            ('smote_tomek', SMOTETomek(random_state=42))
        ]
    
    # Default models if none provided
    if models is None:
        models = [
            ('logistic', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(random_state=42, scale_pos_weight=10, eval_metric='logloss'))
        ]
    
    # Convert categorical columns to string
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['category']).columns:
        X_proc[col] = X_proc[col].astype(str)
    
    # Drop identifier columns that might cause issues
    id_columns = ['accountNumber', 'customerId', 'merchantName', 'transactionDateTime']
    X_proc = X_proc.drop([col for col in id_columns if col in X_proc.columns], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print original class balance
    fraud_count = y_train.sum()
    total_count = len(y_train)
    fraud_rate = fraud_count / total_count * 100
    print(f"\nOriginal training data class balance:")
    print(f"- Non-fraud: {total_count - fraud_count} ({100 - fraud_rate:.2f}%)")
    print(f"- Fraud: {fraud_count} ({fraud_rate:.2f}%)")
    print(f"- Fraud to non-fraud ratio: 1:{(total_count - fraud_count) / fraud_count:.2f}")
    
    # Create preprocessor
    numeric_features = X_proc.select_dtypes(include=['number']).columns
    categorical_features = X_proc.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Results dictionary
    results = {}
    
    # Train models with different sampling techniques
    for sampling_name, sampler in sampling_methods:
        print(f"\n----- Sampling Method: {sampling_name} -----")
        
        # Apply preprocessing and sampling to check class balance
        if sampler is not None:
            # Apply preprocessing
            X_train_transformed = preprocessor.fit_transform(X_train)
            
            # Apply sampling
            X_resampled, y_resampled = sampler.fit_resample(X_train_transformed, y_train)
            
            # Print class balance after sampling
            fraud_count_after = y_resampled.sum()
            total_count_after = len(y_resampled)
            fraud_rate_after = fraud_count_after / total_count_after * 100
            
            print(f"\nClass balance after {sampling_name} sampling:")
            print(f"- Non-fraud: {total_count_after - fraud_count_after} ({100 - fraud_rate_after:.2f}%)")
            print(f"- Fraud: {fraud_count_after} ({fraud_rate_after:.2f}%)")
            print(f"- Fraud to non-fraud ratio: 1:{(total_count_after - fraud_count_after) / fraud_count_after if fraud_count_after > 0 else 'N/A':.2f}")
            print(f"- Total samples: {total_count_after} (vs. original {total_count})")
        else:
            print("\nNo sampling applied, using original class distribution")
        
        for model_name, model in models:
            # Create cache file name
            cache_file = os.path.join(cache_dir, f'{model_name}_{sampling_name}_results.pkl')
            
            # Check for cached results
            if os.path.exists(cache_file):
                results[f'{model_name}_{sampling_name}'] = load_from_pickle(cache_file)
                continue
            
            print(f"\nTraining {model_name} with {sampling_name} sampling...")
            
            # Create pipeline
            if sampler is None:
                # No sampling
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
            else:
                # With sampling
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('sampler', sampler),
                    ('classifier', model)
                ])
            
            # Train model
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            print(f"Training time: {training_time:.2f} seconds")
            
            # Evaluate model
            eval_results = evaluate_classifier(
                pipeline, X_test, y_test, f"{model_name.capitalize()} with {sampling_name}"
            )
            
            # Save results
            model_results = {
                'model': pipeline,
                'evaluation': eval_results,
                'training_time': training_time,
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape,
                'sampling_method': sampling_name,
                'model_type': model_name
            }
            
            # Add class balance information to results
            if sampler is not None:
                model_results['class_balance'] = {
                    'original_fraud_rate': fraud_rate,
                    'sampled_fraud_rate': fraud_rate_after,
                    'original_samples': total_count,
                    'sampled_samples': total_count_after
                }
            else:
                model_results['class_balance'] = {
                    'original_fraud_rate': fraud_rate,
                    'sampled_fraud_rate': fraud_rate,
                    'original_samples': total_count,
                    'sampled_samples': total_count
                }
            
            results[f'{model_name}_{sampling_name}'] = model_results
            save_to_pickle(model_results, cache_file)
    
    # Compare results
    comparison_df = pd.DataFrame({
        'Model': [f"{res['model_type'].capitalize()} with {res['sampling_method']}" for res in results.values()],
        'Precision': [res['evaluation']['precision'] for res in results.values()],
        'Recall': [res['evaluation']['recall'] for res in results.values()],
        'F1 Score': [res['evaluation']['f1'] for res in results.values()],
        'ROC AUC': [res['evaluation'].get('roc_auc', 0) for res in results.values()],
        'Training Time': [res['training_time'] for res in results.values()],
        'Fraud %': [res['class_balance']['sampled_fraud_rate'] for res in results.values()]
    })
    
    # Sort by F1 score
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='F1 Score', data=comparison_df)
    plt.title('Model F1 Score Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'model_f1_comparison.png'))
    plt.close()
    
    # Visualize class balance impact
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Fraud %', data=comparison_df)
    plt.title('Fraud Percentage After Sampling')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 1, 2)
    sns.scatterplot(x='Fraud %', y='F1 Score', hue='Model', data=comparison_df)
    plt.title('F1 Score vs Fraud Percentage')
    plt.xlabel('Fraud Percentage After Sampling')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'class_balance_impact.png'))
    plt.close()
    
    # Return results dictionary
    return results

def optimize_hyperparameters(X, y, model_type='rf', sampling='smote', cache_dir='models'):
    """
    Optimize hyperparameters for a given model and sampling method.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - model_type: Type of model ('logistic', 'rf', 'xgb')
    - sampling: Sampling method ('none', 'smote', 'random_under', etc.)
    - cache_dir: Directory to cache results
    
    Returns:
    - Dictionary of results
    """
    print(f"\n===== HYPERPARAMETER OPTIMIZATION: {model_type.upper()} with {sampling} =====")
    
    # Check for cached results
    cache_file = os.path.join(cache_dir, f'{model_type}_{sampling}_optimized.pkl')
    if os.path.exists(cache_file):
        return load_from_pickle(cache_file)
    
    # Convert categorical columns to string
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['category']).columns:
        X_proc[col] = X_proc[col].astype(str)
    
    # Drop identifier columns that might cause issues
    id_columns = ['accountNumber', 'customerId', 'merchantName', 'transactionDateTime']
    X_proc = X_proc.drop([col for col in id_columns if col in X_proc.columns], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print original class balance
    fraud_count = y_train.sum()
    total_count = len(y_train)
    fraud_rate = fraud_count / total_count * 100
    print(f"\nOriginal training data class balance:")
    print(f"- Non-fraud: {total_count - fraud_count} ({100 - fraud_rate:.2f}%)")
    print(f"- Fraud: {fraud_count} ({fraud_rate:.2f}%)")
    print(f"- Fraud to non-fraud ratio: 1:{(total_count - fraud_count) / fraud_count:.2f}")
    
    # Create preprocessor
    numeric_features = X_proc.select_dtypes(include=['number']).columns
    categorical_features = X_proc.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Set up sampling method
    if sampling == 'none':
        sampler = None
        print("\nNo sampling applied, using original class distribution")
    elif sampling == 'smote':
        sampler = SMOTE(random_state=42)
    elif sampling == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif sampling == 'random_under':
        sampler = RandomUnderSampler(random_state=42)
    elif sampling == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")
    
    # Check class balance after sampling if a sampler is used
    if sampler is not None:
        # Apply preprocessing
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_train_transformed, y_train)
        
        # Print class balance after sampling
        fraud_count_after = y_resampled.sum()
        total_count_after = len(y_resampled)
        fraud_rate_after = fraud_count_after / total_count_after * 100
        
        print(f"\nClass balance after {sampling} sampling:")
        print(f"- Non-fraud: {total_count_after - fraud_count_after} ({100 - fraud_rate_after:.2f}%)")
        print(f"- Fraud: {fraud_count_after} ({fraud_rate_after:.2f}%)")
        print(f"- Fraud to non-fraud ratio: 1:{(total_count_after - fraud_count_after) / fraud_count_after if fraud_count_after > 0 else 'N/A':.2f}")
        print(f"- Total samples: {total_count_after} (vs. original {total_count})")
    
    # Set up model and parameter grid
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga'],  # Only saga supports all penalties
            'classifier__l1_ratio': [0.0, 0.5, 1.0],  # Only used with elasticnet
            'classifier__class_weight': ['balanced', None]
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__gamma': [0, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0],
            'classifier__scale_pos_weight': [1, 5, 10, 20]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    if sampler is None:
        pipeline = Pipeline([
        ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    else:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', sampler),
            ('classifier', model)
        ])
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Find smaller subset for grid search to save time
    if len(X_train) > 10000:
        print(f"Using subset of {10000} samples for hyperparameter search")
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train, train_size=10000, random_state=42, stratify=y_train
        )
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    # Perform grid search
    print("Starting grid search...")
    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid, cv=cv, scoring='f1',
        n_jobs=-1, verbose=1, return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train_sample, y_train_sample)
    grid_search_time = time.time() - start_time
    
    print(f"Grid search time: {grid_search_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters on full training set
    print("Training final model with best parameters...")
    best_model = grid_search.best_estimator_
    
    start_time = time.time()
    best_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Final model training time: {training_time:.2f} seconds")
    
    # Evaluate model
    eval_results = evaluate_classifier(
        best_model, X_test, y_test, f"Optimized {model_type.upper()} with {sampling}"
    )
    
    # Save results
    results = {
        'model': best_model,
        'evaluation': eval_results,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'grid_search_time': grid_search_time,
        'training_time': training_time
    }
    
    # Add class balance information to results
    if sampler is not None:
        results['class_balance'] = {
            'original_fraud_rate': fraud_rate,
            'sampled_fraud_rate': fraud_rate_after,
            'original_samples': total_count,
            'sampled_samples': total_count_after
        }
    else:
        results['class_balance'] = {
            'original_fraud_rate': fraud_rate,
            'sampled_fraud_rate': fraud_rate,
            'original_samples': total_count,
            'sampled_samples': total_count
        }
    
    save_to_pickle(results, cache_file)
    return results

def compare_all_models(results_dict):
    """
    Compare all model results and visualize the comparison.
    
    Parameters:
    - results_dict: Dictionary of model results
    
    Returns:
    - DataFrame with comparison results
    """
    print("\n===== COMPREHENSIVE MODEL COMPARISON =====")
    
    # Extract results
    models = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    training_times = []
    
    for name, results in results_dict.items():
        # Extract model info
        model_name = name if isinstance(name, str) else "Unknown"
        
        # Extract evaluation metrics
        eval_metrics = results.get('evaluation', {})
        if eval_metrics:
            precision = eval_metrics.get('precision', 0)
            recall = eval_metrics.get('recall', 0)
            f1 = eval_metrics.get('f1', 0)
            roc_auc = eval_metrics.get('roc_auc', 0)
            
            # Extract training time
            training_time = results.get('training_time', 0)
            
            # Add to lists
            models.append(model_name)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            roc_aucs.append(roc_auc)
            training_times.append(training_time)
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        'Model': models,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs,
        'Training Time (s)': training_times
    })
    
    # Sort by F1 score
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Visualize comparison
    plt.figure(figsize=(14, 10))
    
    # Plot F1 scores
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='F1 Score', data=comparison_df)
    plt.title('Model F1 Score Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Plot precision and recall
    plt.subplot(2, 1, 2)
    metrics_df = comparison_df.melt(
        id_vars='Model', 
        value_vars=['Precision', 'Recall', 'ROC AUC'],
        var_name='Metric', value_name='Score'
    )
    sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df)
    plt.title('Precision, Recall, and ROC AUC Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'model_comparison.png'))
    plt.close()
    
    return comparison_df

def comprehensive_evaluation(df, output_dir='fraud_detection_results', use_cache=True, force_recalculate=False):
    """
    Run comprehensive evaluation of fraud detection models.
    
    Parameters:
    - df: Input DataFrame
    - output_dir: Directory for saving results
    - use_cache: Whether to use cached results
    - force_recalculate: Force recalculation even if cache exists
    
    Returns:
    - Dictionary of results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Include data size in the cache file name
    data_size = len(df)
    cache_file = os.path.join(output_dir, f'comprehensive_evaluation_results_size_{data_size}.pkl')
    
    # Check for cached comprehensive results
    if use_cache and not force_recalculate and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                all_results = pickle.load(f)
            print(f"\nLoaded cached comprehensive evaluation results from {cache_file}")
            print(f"Using cached results for data size {data_size}")
            
            # Print a summary of the cached results
            print("\nCached Results Summary:")
            for model_name, results in all_results.items():
                if 'evaluation' in results:
                    print(f"- {model_name}: F1 Score = {results['evaluation'].get('f1', 'N/A'):.4f}")
            
            return all_results
        except Exception as e:
            print(f"Error loading cached results: {str(e)}. Recomputing...")
    
    # Results dictionary
    all_results = {}
    
    # 1. Baseline models
    print("\nStep 1: Evaluating baseline models")
    for model_type in ['logistic', 'rf', 'xgb']:
        results = create_baseline_model(df, model_type, cache_dir=output_dir)
        all_results[f'baseline_{model_type}'] = results
    
    # 2. Feature engineering
    print("\nStep 2: Performing feature engineering")
    df_engineered = feature_engineering(df, cache_dir=output_dir)
    
    # 3. Feature selection
    print("\nStep 3: Performing feature selection")
    if 'isFraud' in df_engineered.columns:
        X = df_engineered.drop('isFraud', axis=1)
        y = df_engineered['isFraud']
        
        selected_features = {}
        for method in ['forest', 'mutual_info']:
            try:
                features = feature_selection(X, y, method=method, k=20, cache_dir=output_dir)
                selected_features[method] = features
            except Exception as e:
                print(f"Warning: Feature selection with {method} failed: {str(e)}")
                selected_features[method] = []
        
        # Initialize X_selected to use all features by default
        X_selected = X
        
        # Keep only selected features that actually exist in the dataframe
        union_features = []
        if len(selected_features.get('forest', [])) > 0 or len(selected_features.get('mutual_info', [])) > 0:
            all_features = list(set(selected_features.get('forest', []) + selected_features.get('mutual_info', [])))
            # Filter to only include features that exist in X
            union_features = [f for f in all_features if f in X.columns]
            print(f"Selected {len(union_features)} valid features out of {len(all_features)} total selected features")
            
            if len(union_features) > 0:
                # Only use selected features if we have valid ones
                X_selected = X[union_features]
                print(f"Using {len(union_features)} selected features for modeling")
            else:
                print("Warning: No valid features were selected. Using all features.")
    else:
        print("Error: 'isFraud' column not found in engineered data")
        # Create dummy variables to avoid errors
        X = df_engineered
        y = pd.Series(0, index=df_engineered.index)
        X_selected = X
    
    # 4. Train models with sampling techniques
    print("\nStep 4: Training models with sampling techniques")
    sampling_results = train_models_with_sampling(
        X_selected, y,
        sampling_methods=[
            ('none', None),
            ('smote', SMOTE(random_state=42)),
            ('random_under', RandomUnderSampler(random_state=42)),
            ('smote_tomek', SMOTETomek(random_state=42))
        ],
        models=[
            ('logistic', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(random_state=42, scale_pos_weight=10, eval_metric='logloss'))
        ],
        cache_dir=output_dir
    )
    all_results.update(sampling_results)
    
    # 5. Hyperparameter optimization for the best model (RF with SMOTE)
    print("\nStep 5: Optimizing hyperparameters")
    best_model_type = 'rf'
    best_sampling = 'smote'
    
    optimized_results = optimize_hyperparameters(
        X_selected, y, model_type=best_model_type, sampling=best_sampling, cache_dir=output_dir
    )
    all_results['optimized_model'] = optimized_results
    
    # 6. Compare all models
    print("\nStep 6: Comprehensive model comparison")
    comparison_df = compare_all_models(all_results)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Save comprehensive results
    print(f"\nSaving comprehensive evaluation results to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results

def run_fraud_detection_pipeline(data_path, sample_size=None, output_dir='fraud_detection_results', use_cache=True, force_recalculate=False):
    """
    Run the complete fraud detection pipeline on the given data.
    
    Parameters:
    - data_path: Path to transaction data file
    - sample_size: Number of records to sample (optional)
    - output_dir: Directory for output files
    - use_cache: Whether to use cached results
    - force_recalculate: Force recalculation even if cache exists
    
    Returns:
    - Dictionary of results
    """
    print("===== FRAUD DETECTION PIPELINE =====")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Include sample size in the cache file name
    sample_size_str = f"_sample_{sample_size}" if sample_size else ""
    pipeline_cache_file = os.path.join(output_dir, f'pipeline_results{sample_size_str}.pkl')
    
    # Check for cached pipeline results
    if use_cache and not force_recalculate and os.path.exists(pipeline_cache_file):
        try:
            with open(pipeline_cache_file, 'rb') as f:
                results = pickle.load(f)
            print(f"\nLoaded cached pipeline results from {pipeline_cache_file}")
            print(f"Using cached results for sample size {sample_size}")
            return results
        except Exception as e:
            print(f"Error loading cached pipeline results: {str(e)}. Recomputing...")
    
    # 1. Load data
    print("\nStep 1: Loading data")
    import pandas as pd
    import json
    
    try:
        # Load data
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        # Parse JSON
        data = []
        for line in tqdm(lines, desc="Parsing JSON"):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line[:100]}...")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
            print(f"Sampled {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # 2. Basic data exploration
    print("\nStep 2: Basic data exploration")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for fraud label
    if 'isFraud' in df.columns:
        fraud_count = df['isFraud'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count * 100
        print(f"Fraud rate: {fraud_count} out of {total_count} ({fraud_rate:.2f}%)")
    else:
        print("Warning: 'isFraud' column not found in the data")
    
    # 3. Run comprehensive evaluation
    print("\nStep 3: Running comprehensive evaluation")
    results = comprehensive_evaluation(df, output_dir=output_dir, use_cache=use_cache, force_recalculate=force_recalculate)
    
    # 4. Save final model
    print("\nStep 4: Saving final model")
    if 'optimized_model' in results:
        best_model = results['optimized_model']['model']
        model_path = os.path.join(output_dir, 'final_model.pkl')
        save_to_pickle(best_model, model_path)
        print(f"Final model saved to {model_path}")
    
    # Save pipeline results
    with open(pipeline_cache_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Pipeline results saved to {pipeline_cache_file}")
    
    print("\nFraud detection pipeline completed successfully")
    return results

def train_ensemble_model(X, y, base_models=None, sampling='smote', cache_dir='models'):
    """
    Train an ensemble model using multiple base models.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - base_models: List of base models to use
    - sampling: Sampling method to use
    - cache_dir: Directory to cache results
    
    Returns:
    - Dictionary of results
    """
    print("\n===== ENSEMBLE MODEL TRAINING =====")
    
    # Check for cached results
    cache_file = os.path.join(cache_dir, f'ensemble_{sampling}_results.pkl')
    if os.path.exists(cache_file):
        return load_from_pickle(cache_file)
    
    # Default base models if none provided
    if base_models is None:
        base_models = [
            ('logistic', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(random_state=42, scale_pos_weight=10, eval_metric='logloss'))
        ]
    
    # Convert categorical columns to string
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['category']).columns:
        X_proc[col] = X_proc[col].astype(str)
    
    # Drop identifier columns that might cause issues
    id_columns = ['accountNumber', 'customerId', 'merchantName', 'transactionDateTime']
    X_proc = X_proc.drop([col for col in id_columns if col in X_proc.columns], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print original class balance
    fraud_count = y_train.sum()
    total_count = len(y_train)
    fraud_rate = fraud_count / total_count * 100
    print(f"\nOriginal training data class balance:")
    print(f"- Non-fraud: {total_count - fraud_count} ({100 - fraud_rate:.2f}%)")
    print(f"- Fraud: {fraud_count} ({fraud_rate:.2f}%)")
    print(f"- Fraud to non-fraud ratio: 1:{(total_count - fraud_count) / fraud_count:.2f}")
    
    # Create preprocessor
    numeric_features = X_proc.select_dtypes(include=['number']).columns
    categorical_features = X_proc.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Set up sampling method
    if sampling == 'none':
        sampler = None
        print("\nNo sampling applied, using original class distribution")
    elif sampling == 'smote':
        sampler = SMOTE(random_state=42)
    elif sampling == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif sampling == 'random_under':
        sampler = RandomUnderSampler(random_state=42)
    elif sampling == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")
    
    # Check class balance after sampling if a sampler is used
    if sampler is not None:
        # Apply preprocessing
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_train_transformed, y_train)
        
        # Print class balance after sampling
        fraud_count_after = y_resampled.sum()
        total_count_after = len(y_resampled)
        fraud_rate_after = fraud_count_after / total_count_after * 100
        
        print(f"\nClass balance after {sampling} sampling:")
        print(f"- Non-fraud: {total_count_after - fraud_count_after} ({100 - fraud_rate_after:.2f}%)")
        print(f"- Fraud: {fraud_count_after} ({fraud_rate_after:.2f}%)")
        print(f"- Fraud to non-fraud ratio: 1:{(total_count_after - fraud_count_after) / fraud_count_after if fraud_count_after > 0 else 'N/A':.2f}")
        print(f"- Total samples: {total_count_after} (vs. original {total_count})")
    
    # Create base model pipelines
    base_pipelines = []
    for name, model in base_models:
        if sampler is None:
            pipeline = Pipeline([
                ('preprocessor', clone(preprocessor)),
                ('classifier', clone(model))
            ])
        else:
            pipeline = ImbPipeline([
                ('preprocessor', clone(preprocessor)),
                ('sampler', clone(sampler)),
                ('classifier', clone(model))
            ])
        base_pipelines.append((name, pipeline))
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=base_pipelines,
        voting='soft'
    )
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate model
    eval_results = evaluate_classifier(
        voting_clf, X_test, y_test, f"Ensemble with {sampling}"
    )
    
    # Save results
    results = {
        'model': voting_clf,
        'evaluation': eval_results,
        'training_time': training_time,
        'base_models': base_models,
        'sampling_method': sampling
    }
    
    # Add class balance information to results
    if sampler is not None:
        results['class_balance'] = {
            'original_fraud_rate': fraud_rate,
            'sampled_fraud_rate': fraud_rate_after,
            'original_samples': total_count,
            'sampled_samples': total_count_after
        }
    else:
        results['class_balance'] = {
            'original_fraud_rate': fraud_rate,
            'sampled_fraud_rate': fraud_rate,
            'original_samples': total_count,
            'sampled_samples': total_count
        }
    
    save_to_pickle(results, cache_file)
    return results

def visualize_class_balance(results_dict):
    """
    Visualize the class balance across different sampling methods.
    
    Parameters:
    - results_dict: Dictionary of model results
    
    Returns:
    - None
    """
    print("\n===== CLASS BALANCE VISUALIZATION =====")
    
    # Extract class balance information
    sampling_methods = []
    original_fraud_rates = []
    sampled_fraud_rates = []
    original_samples = []
    sampled_samples = []
    f1_scores = []
    
    for model_name, results in results_dict.items():
        if 'class_balance' in results:
            # Extract sampling method from model name
            if '_' in model_name:
                model_type, sampling = model_name.split('_', 1)
            else:
                sampling = results.get('sampling_method', 'unknown')
            
            # Skip duplicates
            if sampling in sampling_methods:
                continue
            
            # Add data
            sampling_methods.append(sampling)
            original_fraud_rates.append(results['class_balance']['original_fraud_rate'])
            sampled_fraud_rates.append(results['class_balance']['sampled_fraud_rate'])
            original_samples.append(results['class_balance']['original_samples'])
            sampled_samples.append(results['class_balance']['sampled_samples'])
            f1_scores.append(results['evaluation']['f1'])
    
    if not sampling_methods:
        print("No class balance information found in results.")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sampling Method': sampling_methods,
        'Original Fraud Rate (%)': original_fraud_rates,
        'Sampled Fraud Rate (%)': sampled_fraud_rates,
        'Original Samples': original_samples,
        'Sampled Samples': sampled_samples,
        'F1 Score': f1_scores
    })
    
    # Sort by F1 score
    df = df.sort_values('F1 Score', ascending=False)
    
    # Print table
    print("\nClass Balance Comparison:")
    print(df.to_string(index=False))
    
    # Plot fraud rates
    plt.figure(figsize=(12, 6))
    
    # Bar chart of fraud rates
    x = np.arange(len(sampling_methods))
    width = 0.35
    
    plt.bar(x - width/2, df['Original Fraud Rate (%)'], width, label='Original')
    plt.bar(x + width/2, df['Sampled Fraud Rate (%)'], width, label='After Sampling')
    
    plt.xlabel('Sampling Method')
    plt.ylabel('Fraud Rate (%)')
    plt.title('Fraud Rate Before and After Sampling')
    plt.xticks(x, df['Sampling Method'])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'fraud_rate_comparison.png'))
    plt.close()
    
    # Plot sample counts
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, df['Original Samples'], width, label='Original')
    plt.bar(x + width/2, df['Sampled Samples'], width, label='After Sampling')
    
    plt.xlabel('Sampling Method')
    plt.ylabel('Number of Samples')
    plt.title('Sample Count Before and After Sampling')
    plt.xticks(x, df['Sampling Method'])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'sample_count_comparison.png'))
    plt.close()
    
    # Plot F1 score vs fraud rate
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df['Sampled Fraud Rate (%)'], df['F1 Score'], s=100, alpha=0.7)
    
    # Add labels to each point
    for i, method in enumerate(df['Sampling Method']):
        plt.annotate(method, 
                    (df['Sampled Fraud Rate (%)'].iloc[i], df['F1 Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Fraud Rate After Sampling (%)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Fraud Rate After Sampling')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'f1_vs_fraud_rate.png'))
    plt.close()
    
    # Return the DataFrame for further analysis
    return df

# If run as a script
if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Create directories for results
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
    
    # Train models with different sampling techniques
    sampling_results = train_models_with_sampling(X, y)
    
    # Visualize class balance across sampling methods
    balance_df = visualize_class_balance(sampling_results)
    
    # Optimize hyperparameters for best model
    # Based on sampling results, choose the best sampling method
    best_sampling = 'smote'  # This should be determined from sampling_results
    
    # Optimize for different model types
    rf_results = optimize_hyperparameters(X, y, model_type='rf', sampling=best_sampling)
    xgb_results = optimize_hyperparameters(X, y, model_type='xgb', sampling=best_sampling)
    logistic_results = optimize_hyperparameters(X, y, model_type='logistic', sampling=best_sampling)
    
    # Train ensemble model
    ensemble_results = train_ensemble_model(X, y, sampling=best_sampling)
    
    # Compare all optimized models
    optimized_results = {
        'rf_optimized': rf_results,
        'xgb_optimized': xgb_results,
        'logistic_optimized': logistic_results,
        'ensemble': ensemble_results
    }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': [
            'Random Forest (Optimized)',
            'XGBoost (Optimized)',
            'Logistic Regression (Optimized)',
            'Ensemble'
        ],
        'Precision': [
            rf_results['evaluation']['precision'],
            xgb_results['evaluation']['precision'],
            logistic_results['evaluation']['precision'],
            ensemble_results['evaluation']['precision']
        ],
        'Recall': [
            rf_results['evaluation']['recall'],
            xgb_results['evaluation']['recall'],
            logistic_results['evaluation']['recall'],
            ensemble_results['evaluation']['recall']
        ],
        'F1 Score': [
            rf_results['evaluation']['f1'],
            xgb_results['evaluation']['f1'],
            logistic_results['evaluation']['f1'],
            ensemble_results['evaluation']['f1']
        ],
        'ROC AUC': [
            rf_results['evaluation'].get('roc_auc', 0),
            xgb_results['evaluation'].get('roc_auc', 0),
            logistic_results['evaluation'].get('roc_auc', 0),
            ensemble_results['evaluation'].get('roc_auc', 0)
        ]
    })
    
    # Sort by F1 score
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
    
    print("\nOptimized Model Comparison:")
    print(comparison_df)
    
    # Visualize optimized model comparison
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    metrics = ['Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, model in enumerate(comparison_df['Model']):
        values = comparison_df.loc[comparison_df['Model'] == model, metrics].values.flatten()
        plt.bar(x + i*width, values, width, label=model)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Optimized Model Comparison')
    plt.xticks(x + width*1.5, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'plots', 'optimized_model_comparison.png'))
    plt.close()
    
    print("\nAnalysis complete. Results saved to 'results' directory.")