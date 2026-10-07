#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">Visualizations on the Toronto Bicyle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed – Student ID: 501304501
"""


# In[2]:


# --- Import libraries ---
#import sys
#!{sys.executable} -m pip install plotly
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates


# In[3]:


# --- GLOBAL CONFIGURATION ---
# False = GUI Mode (save and show popup)
# True = Main Batch Mode (save & close, no popup)
BATCH_MODE = False  

def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14

def save_or_show(fig, filename=None, interactive=False):
    """
    Handles output: Saves if filename exists, Shows if BATCH_MODE is False.
    """
    # 1. Save to disk
    is_saved = False
    if filename:
        if interactive:
            fig.write_html(filename)
            print(f"[Saved] {filename}")
        else:
            plt.savefig(filename, bbox_inches='tight')
            print(f"[Saved] {filename}")
        is_saved = True

    # 2. Show on screen or close
    if BATCH_MODE:
        if not interactive:
            plt.close()
    else:
        # GUI Mode: show the window (Matplotlib shows directly and Plotly is handled by the saved file)
        if interactive:
            if not is_saved: 
                 fig.show() 
        else:
            plt.show()

# -------------------------------------
# GRAPHS
# -------------------------------------

# -------------------------------------------------   
# 1. Exploratory and data quality
# -------------------------------------------------  

def plot_missing_heatmap(df, filename=None):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm', yticklabels=False)
    plt.title('Heatmap of Missing Values')
    save_or_show(plt, filename)

def plot_correlation_heatmap(df, cols, filename):
    """
    Plots a correlation heatmap. 
    Automatically converts categorical columns (like 'STATUS') to numeric codes 
    so they can be included in the calculation.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # 1. Create a specific dataframe for correlation
    corr_df = df[cols].copy()
    
    # 2. Handle categorical conversions
    # Loop through columns to check for text/category types
    for col in cols:
        if corr_df[col].dtype == 'object' or corr_df[col].dtype.name == 'category':
            # Convert to category type
            cat_series = corr_df[col].astype('category')
            # Save the mapping
            mapping = dict(enumerate(cat_series.cat.categories))
            print(f"Column '{col}': {mapping}")
            
            # Replace text with code in the dataframe
            corr_df[col] = cat_series.cat.codes

    # 3. Calculate correlation
    corr = corr_df.corr()

    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    
    save_or_show(plt, filename)

def plot_pair_plot(df, cols, hue_col=None, filename=None):
    sns.pairplot(df[cols + ([hue_col] if hue_col else [])], hue=hue_col, palette='coolwarm')
    save_or_show(plt, filename)

def plot_distributions_grid(df, cols, filename=None):
    """
    Subplots of histograms for numerical distributions, colored by Status.
    """
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 5 * len(cols)), squeeze=False)
    axes = axes.flatten() 
    
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, hue='STATUS', kde=True, ax=axes[i], 
                     palette='viridis', multiple='stack')
        axes[i].set_title(f'Distribution of {col} by Status')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    save_or_show(plt, filename)

import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def plot_categorical_association_heatmap(df, cols, filename=None):
    """
    Calculates and plots the Cramér's V correlation matrix for categorical variables.
    """
    # 1. Initialize matrix
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    # 2. Calculate Cramér's V for every pair
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                val = cramers_v(df[cols[i]], df[cols[j]])
                corr_matrix.iloc[i, j] = val

    # 3. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='magma', vmin=0, vmax=1, fmt='.2f')
    plt.title("Categorical Association (Cramér's V)")
    save_or_show(plt, filename)

def plot_parallel_categories(df, cols, color_col, filename, max_cats=10):
    """
    Visualizes flow between categorical columns. 
    Automatically groups rare categories into 'Other' to keep the chart readable.
    """
    # 1. Create a copy
    plot_df = df.copy()
    
    # 2. Keep only top N
    for col in cols:
        if col in plot_df.columns:
            top_n_values = plot_df[col].value_counts().nlargest(max_cats).index
            plot_df = plot_df[plot_df[col].isin(top_n_values)]
            
    # 3. Handle color encoding and capture mappings
    color_target = color_col
    tick_vals = []
    tick_text = []
    
    # Check if the color column is categorical/string
    if plot_df[color_col].dtype == 'object' or plot_df[color_col].dtype.name == 'category':
        # Convert to category type explicitly
        cat_series = plot_df[color_col].astype('category')
        
        # Create the numeric column for Plotly
        color_target = color_col + '_code'
        plot_df[color_target] = cat_series.cat.codes
        
        # Save the mapping for the legend
        # Get unique codes and their corresponding text
        categories = cat_series.cat.categories
        tick_vals = list(range(len(categories)))
        tick_text = list(categories)

    # 4. Plot
    fig = px.parallel_categories(
        plot_df,
        dimensions=cols,
        color=color_target, 
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Categorical Flow (Top {max_cats} Strict Filter)"
    )
    
    # 5. Override numbers with text
    if tick_vals:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=color_col,
                tickvals=tick_vals,
                ticktext=tick_text
            )
        )
    
    save_or_show(fig, filename, interactive=True)

# -------------------------------------------------   
# 2. Temporal analysis
# -------------------------------------------------  

def plot_temporal_overview(df, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(data=df, x='OCC_YEAR', hue='OCC_YEAR', palette='viridis', legend=False, ax=axes[0])
    axes[0].set_title('Theft Count by Year')
    sns.countplot(data=df, x='OCC_MONTH', hue='OCC_MONTH', palette='coolwarm', legend=False, ax=axes[1])
    axes[1].set_title('Seasonality (Month)')
    plt.tight_layout()
    save_or_show(plt, filename)

def plot_dow_heatmap(df, filename=None):
    pivot = df.pivot_table(index='OCC_DOW', columns='OCC_HOUR', values='OBJECTID', aggfunc='count').fillna(0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Heatmap: Day vs Hour')
    save_or_show(plt, filename)

def plot_time_series_decomposition(df, date_col='OCC_DATE', freq='W', filename=None):
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # 1. Prepare data
    ts = df.set_index(date_col).resample(freq).size()
    
    # 2. Decompose
    decomposition = seasonal_decompose(ts, model='additive', period=52)
    
    # 3. Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14))
     
    ax1.plot(decomposition.observed.index, decomposition.observed, color='black')
    ax1.set_title('Observed')
    
    ax2.plot(decomposition.trend.index, decomposition.trend, color='blue')
    ax2.set_title('Trend')
    
    ax3.plot(decomposition.seasonal.index, decomposition.seasonal, color='green')
    ax3.set_title('Seasonality')
    
    ax4.plot(decomposition.resid.index, decomposition.resid, color='red', marker='.', linestyle='None')
    ax4.set_title('Residuals')
    
    # 4. Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylabel('Thefts')
        
        # Set ticks every 3 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        # Format as "Mmm YYY"
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_or_show(plt, filename)

def plot_anomaly_detection(df, date_col='OCC_DATE', filename=None):
    ts = df.set_index(date_col).resample('D').size()
    rolling_mean = ts.rolling(window=30).mean()
    rolling_std = ts.rolling(window=30).std()
    anomalies = ts[ts > (rolling_mean + 3 * rolling_std)]
    
    plt.figure(figsize=(14, 7))
    plt.plot(ts.index, ts, label='Daily Thefts', alpha=0.6)
    plt.plot(rolling_mean.index, rolling_mean, color='red', label='30-Day Trend')
    plt.scatter(anomalies.index, anomalies, color='red', s=50, label='Anomaly', zorder=5)
    plt.title('Anomaly Detection (Spikes)')
    plt.legend()
    save_or_show(plt, filename)

def plot_interactive_line(df, date_col='OCC_DATE', filename=None):
    daily_counts = df.groupby(date_col).size().reset_index(name='Thefts')
    fig = px.line(daily_counts, x=date_col, y='Thefts', title='Interactive Timeline')
    save_or_show(fig, filename, interactive=True)

# -------------------------------------------------   
# 3. Spatial analysis
# -------------------------------------------------  

def plot_folium_map(df, lat_col='LAT_WGS84', lon_col='LONG_WGS84', sample_n=1000, filename='theft_map.html'):
    sample = df.sample(min(sample_n, len(df)))
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=11)
    from folium.plugins import MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in sample.iterrows():
        folium.Marker([row[lat_col], row[lon_col]]).add_to(marker_cluster)
    
    if filename: m.save(filename)
    print(f"Folium map saved to {filename}")

def plot_interactive_scatter_map(df, filename=None):
    fig = px.scatter_mapbox(df.sample(min(1000, len(df))), lat="LAT_WGS84", lon="LONG_WGS84", 
                            color="NEIGHBOURHOOD", zoom=10, mapbox_style="carto-positron")
    save_or_show(fig, filename, interactive=True)
    
def plot_network_graph(df, source_col, target_col, min_weight=5, filename=None):
    """
    Plots a network graph where node size is proportional to total traffic 
    and edge width is proportional to the flow volume.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 1. Calculate weighted edges (flow volume)
    edges = df.groupby([source_col, target_col]).size().reset_index(name='weight')
    edges = edges[edges['weight'] >= min_weight]
    
    # Check if any edges remain after filtering
    if edges.empty:
        print(f"No edges found with weight >= {min_weight}. Skipping plot.")
        return

    G = nx.from_pandas_edgelist(edges, source_col, target_col, 'weight')

    # 2. Calculate node sizes (proportional to total connections/traffic volume)
    node_weights = dict(G.degree(weight='weight'))
    max_node_weight = max(node_weights.values())
    node_sizes = [v / max_node_weight * 2000 for v in node_weights.values()]

    # 3. Calculate edge widths (proportional to flow volume)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_edge_weight = max(edge_weights)
    edge_widths = [w / max_edge_weight * 10 for w in edge_weights]
    
    # 4. Draw graph using Spring layout (better separation)
    pos = nx.spring_layout(G, k=0.15, iterations=50) 
    plt.figure(figsize=(14, 14))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(f'Bike Movement Network: {source_col} to {target_col} (Min Flow: {min_weight})', fontsize=16)
    plt.axis('off')
    save_or_show(plt, filename)

# -------------------------------------------------   
# 4. Characteristics
# ------------------------------------------------- 

def plot_word_cloud(df, text_col, filename=None):
    text = " ".join(str(i) for i in df[text_col].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
    plt.title(f'Word Cloud of {text_col}')
    save_or_show(plt, filename)

def plot_static_bubble_chart(df, x_col, y_col, color_col, filename=None, min_count=5, max_cats=15):
    """
    Creates a static bubble chart using Seaborn's scatterplot.

    - Automatically filters x_col and y_col to the top N categories for legibility.
    - Manually positions the legends outside the plot area to prevent overlap.
    """
   
    # 1. Top N filtering 
    plot_df = df.copy()
    cols_to_simplify = [x_col, y_col]
    
    for col in cols_to_simplify:
        # Only simplify if the number of unique values exceeds the limit
        if plot_df[col].nunique() > max_cats:
            # Find top N categories
            top_n = plot_df[col].value_counts().nlargest(max_cats).index
            
            # Create a simplified column by grouping rare values into 'Other'
            plot_df[col] = plot_df[col].apply(lambda x: x if x in top_n else 'Other')

    # 2. Aggregate data to determine bubble size (theft count)
    df_agg = plot_df.groupby([x_col, y_col, color_col]).size().reset_index(name='Theft_Count')
    
    # 3. Filter out rare combinations for a cleaner chart
    df_agg = df_agg[df_agg['Theft_Count'] >= min_count]
    
    # 4. Create the static scatter/bubble chart
    plt.figure(figsize=(18, 10)) 
    
    # Generate the scatter plot, storing the output axes object
    ax = sns.scatterplot(
        data=df_agg, 
        x=x_col, 
        y=y_col, 
        hue=color_col,
        size='Theft_Count',
        sizes=(50, 1000), 
        palette='viridis',
    )
    
    # 5. Legend formatting
    plt.title(f"Static Bubble Chart: {x_col} vs {y_col} by {color_col} (Top {max_cats} Categories)", fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
 
    # Get all handles and labels generated by Seaborn
    handles, labels = ax.get_legend_handles_labels()
    
    # Determine the index where the hue (color) handles end and the size handles begin
    num_hue_categories = df_agg[color_col].nunique()
    
    # 5a. Color legend (status)
    hue_handles = handles[0 : num_hue_categories]
    hue_labels = labels[0 : num_hue_categories]
    
    # 5b. Size legend
    size_legend_title = labels[num_hue_categories] 
    size_handles = handles[num_hue_categories + 1:] 
    size_labels = labels[num_hue_categories + 1:]

    # Hue (color/status) legend placement
    ax.legend(
        hue_handles, hue_labels,
        loc='upper left', bbox_to_anchor=(1.01, 1), title=color_col, frameon=False
    )
    
    # Place the size legend on the bottom right (outside)
    ax.legend(
        size_handles, size_labels,
        loc='lower left', bbox_to_anchor=(1.01, 0.5), title=size_legend_title, frameon=False
    )
    
    # 6. Save/show
    plt.tight_layout(rect=[0, 0, 1.0, 1]) 
    save_or_show(plt, filename)

def plot_3d_scatter(df, x_col, y_col, z_col, color_col, filename=None):
    """
    Creates an interactive 3D scatter plot (X vs Y vs Z) colored by a categorical column.
    """
    import plotly.express as px

    fig = px.scatter_3d(
        df, 
        x=x_col, 
        y=y_col, 
        z=z_col, 
        color=color_col,
        title=f"3D Scatter: {x_col} vs {y_col} vs {z_col} by {color_col}",
        height=700
    )
    
    # Customize the marker to be smaller and clearer
    fig.update_traces(marker=dict(size=3))
    
    # 2. Save/show
    save_or_show(fig, filename, interactive=True)

# -------------------------------------------------   
# 5. Status and reporting
# ------------------------------------------------- 

def plot_status_pie(df, filename=None):
    counts = df['STATUS'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title('Recovery Status')
    save_or_show(plt, filename)

def plot_recovery_rate_by_delay(df, filename=None):
    """
    Plots the percentage of recovered bikes categorized by reporting delay.
    Insight: Does reporting faster actually help?
    """
    # Create a working copy
    data = df.copy()
    
    # 1. Create bins for "Time to Report"
    bins = [-1, 0, 1, 7, 30, 9999]
    labels = ['Same Day', '1 Day', '2-7 Days', '8-30 Days', '> 30 Days']
    data['Delay_Group'] = pd.cut(data['REPORT_DELAY_DAYS'], bins=bins, labels=labels)
    
    # 2. Calculate recovery rate per bin
    # Convert STATUS to a boolean (1 for Recovered, 0 for Stolen)
    # The 'mean' of this column represents the % of bikes recovered
    data['is_recovered'] = (data['STATUS'] == 'RECOVERED').astype(int)
    
    # Group by the bins and calculate the rate
    recovery_stats = data.groupby('Delay_Group')['is_recovered'].mean() * 100
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=recovery_stats.index, y=recovery_stats.values, palette="Greens_r")
    
    plt.title('Does Speed Matter? Recovery Rate by Reporting Delay', fontsize=14)
    plt.xlabel('Time Taken to Report Theft')
    plt.ylabel('Recovery Rate (%)')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(recovery_stats.values):
        ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold')
    
    save_or_show(plt, filename)

# -------------------------------------------------   
# 6. ML evaluation
# -------------------------------------------------

def plot_ml_metrics_heatmap(results_df, filename=None):
    plt.figure(figsize=(8, 5))
    sns.heatmap(results_df.set_index('Model'), annot=True, cmap='RdYlGn', fmt='.3f')
    plt.title('ML Model Performance')
    save_or_show(plt, filename)

def plot_roc_curves(roc_data, filename=None):
    plt.figure(figsize=(8, 8))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend()
    save_or_show(plt, filename)

def plot_calibration_curves(cal_data, filename=None):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    for name, (true, pred) in cal_data.items():
        plt.plot(pred, true, marker='o', label=name)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Calibration'); plt.legend()
    save_or_show(plt, filename)

