#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">Preprocessing the Toronto Bicyle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed – Student ID: 501304501
"""


# In[2]:


# --- Import libraries ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# --- Load data ---

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("File not found.")
        return None


# In[4]:


# --- Inspect data types ---

def inspect_data_types(df):
    """
    Prints a summary of columns and their data types.
    """
    df.info()
    print("\n[INSPECT] Count of column types:")
    print(df.dtypes.value_counts())


# In[5]:


# --- Clean data ---

def clean_data(df):
    """
    Performs data cleaning and preprocessing.
    """
    # ---------------------------------------------    
    # 1. Data transformations (and some initial cleaning)
    # ---------------------------------------------
    
    # 1.1 Date and time cleaning, conversion, and normalization
    df['OCC_DATE'] = pd.to_datetime(df['OCC_DATE'], errors='coerce').dt.normalize()
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce').dt.normalize()
    print("[STEP COMPLETE] OCC_DATE and REPORT_DATE converted to datetime objects.")
    
    # Identify rows with missing dates
    missing_dates_count = df[['OCC_DATE', 'REPORT_DATE']].isnull().any(axis=1).sum()
    print(f"[IDENTIFY] Found {missing_dates_count} rows with missing critical dates (OCC_DATE or REPORT_DATE).")
    
    # Drop rows where critical dates are missing
    if missing_dates_count > 0:
        df.dropna(subset=['OCC_DATE', 'REPORT_DATE'], inplace=True)
        print(f"[HANDLE] Dropped {missing_dates_count} rows with missing dates.")
    else:
        print("[HANDLE] No rows with missing critical dates found.")

    # 1.2 Remove historical outliers to clean years
    old_data = df[df['OCC_YEAR'] < 2014]
    if not old_data.empty:
        df = df[df['OCC_YEAR'] >= 2014].copy()
        print(f"[HANDLE] Dropped {len(old_data)} historical records (pre-2014) to focus on current trends.")

    # 1.3 Column cleanup
    # Drop the old neighbourhood model columns
    cols_to_drop = ['NEIGHBOURHOOD_140', 'HOOD_140']
    cols_dropping = [c for c in cols_to_drop if c in df.columns]
    if cols_dropping:
        df.drop(columns=cols_dropping, inplace=True)
        print(f"[HANDLE] Dropped redundant columns: {cols_dropping}")

    # Rename '158' columns to standard names
    if 'NEIGHBOURHOOD_158' in df.columns:
        df.rename(columns={'NEIGHBOURHOOD_158': 'NEIGHBOURHOOD', 'HOOD_158': 'HOOD'}, inplace=True)
        print("[HANDLE] Renamed 'NEIGHBOURHOOD_158' to 'NEIGHBOURHOOD' and 'HOOD_158' to 'HOOD'.")
              
    # ---------------------------------------------
    # 2. Data cleaning and feature engineering
    # ---------------------------------------------
    
    # 2.1 Remove duplicate records
    initial_count = len(df)
    df = df.drop_duplicates().copy()
    print(f"Removed {initial_count - len(df)} duplicate records.")

    # 2.2 Feature engineering
    # 2.2.1 Add reporting delay feature
    df['REPORT_DELAY_DAYS'] = (df['REPORT_DATE'] - df['OCC_DATE']).dt.days
    print("[STEP COMPLETE] 'REPORT_DELAY_DAYS' feature created.")
    
    # Remove invalid negative delays (errors)
    neg_delays = df[df['REPORT_DELAY_DAYS'] < 0]
    if not neg_delays.empty:
        df = df[df['REPORT_DELAY_DAYS'] >= 0].copy()
        print(f"[HANDLE] Dropped {len(neg_delays)} rows with negative reporting delays.")
    else:
        print("   -> No negative reporting delays found.")
        
    # 2.2.2 Add year-month for trend analysis
    df['YEAR_MONTH'] = df['OCC_DATE'].dt.to_period('M')
    print("[STEP COMPLETE] 'YEAR_MONTH' trend feature created.")

    # ---------------------------------------------
    # 3. Handling missing and 0 data
    # ---------------------------------------------
    
    # 3.1.1 Identify missing values in categorical columns
    categorical_cols = ['BIKE_MAKE', 'BIKE_MODEL', 'BIKE_COLOUR', 'BIKE_TYPE', 'LOCATION_TYPE', 'PREMISES_TYPE']
    print("[IDENTIFY] Missing values in categorical columns:")
    missing_cats = df[categorical_cols].isnull().sum()
    print(missing_cats[missing_cats > 0])

    # 3.1.2 Handle missing values in categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
    print("[HANDLE] Filled missing categorical values with 'Unknown'.")

    # 3.2.1 Identify missing and 0 values in numerical columns
    # 3.2.1.1 BIKE_SPEED identification
    df['BIKE_SPEED'] = pd.to_numeric(df['BIKE_SPEED'], errors='coerce')
    speed_zeros = (df['BIKE_SPEED'] == 0).sum()
    speed_nans = df['BIKE_SPEED'].isnull().sum()
    print(f"[IDENTIFY] BIKE_SPEED issues: {speed_zeros} zeros, {speed_nans} NaNs.")

    # 3.2.1.2 BIKE_COST identification
    df['BIKE_COST'] = pd.to_numeric(df['BIKE_COST'], errors='coerce')
    cost_zeros = (df['BIKE_COST'] == 0).sum()
    cost_nans = df['BIKE_COST'].isnull().sum()
    print(f"[IDENTIFY] BIKE_COST issues: {cost_zeros} zeros, {cost_nans} NaNs.")

    # 3.2.2 Handle missing and 0 values in numerical columns
    # 3.2.2.1 BIKE_SPEED handling
    # Replace 0's in BIKE_SPEED with NaN to prepare for imputation
    df.loc[df['BIKE_SPEED'] == 0, 'BIKE_SPEED'] = np.nan

    # Impute BIKE_SPEED with median
    median_speed = df['BIKE_SPEED'].median()
    df['BIKE_SPEED'] = df['BIKE_SPEED'].fillna(median_speed)
    print(f"[HANDLE] Imputed missing/zero BIKE_SPEED with Median ({median_speed}).")

    # 3.2.2.2 BIKE_COST handling
    # Replace 0's in BIKE_COST with NaN. 
    if cost_zeros > 0:
        df.loc[df['BIKE_COST'] == 0, 'BIKE_COST'] = np.nan
        print(f"[HANDLE] BIKE_COST: Converted {cost_zeros} zero values to NaN (treated as missing).")

    # 3.2.2.3 Spatial data cleaning
    # Drop spatial records with (0,0) coordinates
    zeros_loc = df[(df['LAT_WGS84'] == 0) | (df['LONG_WGS84'] == 0)]
    if not zeros_loc.empty:
        df = df[(df['LAT_WGS84'] != 0) & (df['LONG_WGS84'] != 0)].copy()
        print(f"[HANDLE] Dropped {len(zeros_loc)} records with invalid spatial coordinates (0,0).")

    print("--- PREPROCESSING COMPLETE ---\n")
    return df


# In[6]:


# --- Filter cost outliers ---

def treat_cost_outliers_iqr(df):
    """
    Detects and handles cost outliers using IQR, including the lower bound
    to remove negative or unrealistically low values.
    """
    import numpy as np # Needed for the final filter comparison
    
    df_clean = df.copy()
    costs = df_clean['BIKE_COST'].dropna()
    if costs.empty: return df_clean

    Q1 = costs.quantile(0.25)
    Q3 = costs.quantile(0.75)
    IQR = Q3 - Q1
    
    # 4.1 Identify outliers using BOTH bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Negative costs are usually impossible, so the lower bound should ideally be max(0, lower_bound)
    # However, to strictly follow IQR method, we use the calculated lower_bound.
    
    # Identify all outliers: those below the lower bound OR above the upper bound
    outliers_to_remove = df_clean[
        (df_clean['BIKE_COST'] < lower_bound) | 
        (df_clean['BIKE_COST'] > upper_bound)
    ]
    
    print(f"[IDENTIFY] Cost Outliers (Range: ${lower_bound:.2f} to ${upper_bound:.2f}): {len(outliers_to_remove)} records.")

    # 4.2 Handle/remove outliers
    df_clean = df_clean[~df_clean.index.isin(outliers_to_remove.index)]
    print(f"[HANDLE] Removed {len(outliers_to_remove)} cost outliers.")

    # Ensure no remaining values are actually negative.
    negative_costs = df_clean[df_clean['BIKE_COST'] < 0]
    if not negative_costs.empty:
        df_clean = df_clean[df_clean['BIKE_COST'] >= 0]
        print(f"[CLEANUP] Removed {len(negative_costs)} final negative cost records (Data Quality).")
    
    print("--- OUTLIERS TREATED ---\n")
    return df_clean

