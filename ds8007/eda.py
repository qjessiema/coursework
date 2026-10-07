#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">Exploratory Data Analysis on Toronto Bicyle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed - Student ID: 501304501
"""


# In[2]:


# --- Import libraries ---

import pandas as pd
import preprocessing


# In[3]:


# --- Summary statistics and data profiling ---

def get_dataset_overview(df):
    """
    Returns a dictionary containing the shape and missing value counts.
    """
    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum()[df.isnull().sum() > 0],
        "duplicates": df.duplicated().sum()
    }
    return overview

def get_numerical_summary(df, columns):
    """
    Calculates detailed summary statistics (Mean, Median, Std, Min, Max) 
    for specific numerical columns.
    """
    stats = df[columns].describe().T
  
    stats['median'] = df[columns].median()
    return stats[['count', 'mean', 'median', 'std', 'min', 'max']]

def get_recovery_rate(df):
    """
    Calculates the percentage of bikes with status 'RECOVERED'.
    """
    if 'STATUS' not in df.columns: return 0.0
    recovered = df[df['STATUS'] == 'RECOVERED'].shape[0]
    total = df.shape[0]
    return (recovered / total) * 100


# In[4]:


# --- Data distributions and trends ---

def get_categorical_distribution(df, column, n=10):
    """
    Returns the top N most frequent categories and their percentage.
    """
    counts = df[column].value_counts().head(n)
    percentages = (df[column].value_counts(normalize=True).head(n) * 100).round(2)
    return pd.DataFrame({'Count': counts, 'Percentage': percentages})

def get_temporal_patterns(df):
    """
    Identifies peak times for thefts (Year, Month, Day, Hour).
    """
    patterns = {
        "Peak Year": df['OCC_YEAR'].mode()[0],
        "Peak Month": df['OCC_MONTH'].mode()[0],
        "Peak Day": df['OCC_DOW'].mode()[0],
        "Peak Hour": df['OCC_HOUR'].mode()[0]
    }
    return patterns


# In[5]:


# --- Helper functions for visualization (to be used by GUI) ---

def get_hourly_dow_pivot(df):
    """Returns pivot table for Heatmap (Day vs Hour)."""
    pivot = df.pivot_table(index='OCC_DOW', columns='OCC_HOUR', values='OBJECTID', aggfunc='count').fillna(0)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Reindex for correct day order
    pivot = pivot.reindex([d for d in days if d in pivot.index])
    return pivot

def get_temporal_trend(df):
    """Returns monthly theft counts sorted by time for line charts."""
    return df.groupby('YEAR_MONTH').size()


# In[6]:


# --- Reusable EDA report function ---
def print_eda_report(df):
    """
    This function contains ONLY the print logic. 
    It can be called by main.py OR by the block below.
    """
            
    # Apply outlier treatment for the report stats
    df_clean = preprocessing.treat_cost_outliers_iqr(df)

    print("\n" + "="*40)
    print(" EXPLORATORY DATA ANALYSIS REPORT")
    print("="*40)

    # -------------------------------------------------   
    # 1. Overview
    # -------------------------------------------------   
    
    overview = get_dataset_overview(df_clean)
    print(f"\n[DATASET OVERVIEW]")
    print(f"Total Records: {overview['rows']}")
    print(f"Total Features: {overview['columns']}")
    print(f"Missing Values:\n{overview['missing_values'].to_string()}")

    # -------------------------------------------------   
    # 2. Inspect data types
    # -------------------------------------------------   
    
    preprocessing.inspect_data_types(df)

    # -------------------------------------------------   
    # 3. Numerical distributions
    # -------------------------------------------------   
    
    print(f"\n[NUMERICAL SUMMARY (Cost & Speed)]")
    # Check columns exist before summarizing
    num_cols = [c for c in ['BIKE_COST', 'BIKE_SPEED'] if c in df_clean.columns]
    print(get_numerical_summary(df_clean, num_cols))

    # -------------------------------------------------       
    # 4. Categorical distributions
    # -------------------------------------------------   
    
    print(f"\n[TOP 5 NEIGHBORHOODS]")
    print(get_categorical_distribution(df_clean, 'NEIGHBOURHOOD', n=5))

    print(f"\n[TOP 5 BIKE MAKES]")
    print(get_categorical_distribution(df_clean, 'BIKE_MAKE', n=5))

    print(f"\n[TOP 5 BIKE MODELS]")
    print(get_categorical_distribution(df_clean, 'BIKE_MODEL', n=5))

    print(f"\n[TOP 5 BIKE TYPES]")
    print(get_categorical_distribution(df_clean, 'BIKE_TYPE', n=5))

    print(f"\n[TOP 5 BIKE COLOUR]")
    print(get_categorical_distribution(df_clean, 'BIKE_COLOUR', n=5))

    print(f"\n[TOP 5 LOCATION TYPES]")
    print(get_categorical_distribution(df_clean, 'LOCATION_TYPE', n=5))

    print(f"\n[TOP 5 PREMISE TYPES]")
    print(get_categorical_distribution(df_clean, 'PREMISES_TYPE', n=5))

    print(f"\n[TOP 5 PRIMARY OFFENCES]")
    print(get_categorical_distribution(df_clean, 'PRIMARY_OFFENCE', n=5))

    # -------------------------------------------------    
    # 5. Findings and trends
    # -------------------------------------------------    
    
    patterns = get_temporal_patterns(df_clean)
    recovery = get_recovery_rate(df_clean)
        
    print(f"\n[KEY FINDINGS]")
    print(f"1. Recovery Rate: {recovery:.2f}% (The vast majority of bikes are never found)")
    print(f"2. Peak Theft Time: {patterns['Peak Day']}s at Hour {patterns['Peak Hour']}:00")
    print(f"3. Seasonal Peak: {patterns['Peak Month']}")
    print(f"4. Peak Year: {patterns['Peak Year']}")
    print(f"5. Most Dangerous Neighborhood: {df['NEIGHBOURHOOD'].mode()[0]}")
        
    print("\n" + "="*40)
    print(" END OF REPORT")
    print("="*40)


# In[7]:


if __name__ == "__main__":
    print("Loading and Preprocessing Data...")
    
    # Load data using pipeline
    raw_df = preprocessing.load_data('Bicycle_Thefts_Open_Data_3945305316907060423.csv')
    preprocessing.inspect_data_types(raw_df)
    if raw_df is not None:
        df = preprocessing.clean_data(raw_df)

        # Call report function
        print_eda_report(df)

