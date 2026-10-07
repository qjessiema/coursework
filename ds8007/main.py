#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">Integration Module for Toronto Bicycle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed – Student ID: 501304501
"""


# In[2]:


# --- Import libraries and files ---

import sys
import os
import preprocessing
import visualization
import modeling
import eda
import gui


# In[3]:


def run_batch_report():
    print("\n--- STARTING BATCH REPORT ---")
    OUT_DIR = "batch_visuals_output"
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    # -------------------------------------------------       
    # 1. Turn ON Batch Mode (saves images and closes them immediately)
    # -------------------------------------------------   
    
    visualization.BATCH_MODE = True

    # -------------------------------------------------   
    # 2. Load
    # -------------------------------------------------   
    
    print("Loading data...")
    raw_df = preprocessing.load_data('Bicycle_Thefts_Open_Data_3945305316907060423.csv')
    if raw_df is None: return

    df = preprocessing.clean_data(raw_df)
    df_clean = preprocessing.treat_cost_outliers_iqr(df)

    # -------------------------------------------------   
    # 3. EDA
    # ------------------------------------------------- 
    
    print("Running EDA...")
    eda.print_eda_report(df_clean)

    # -------------------------------------------------       
    # 4. Visuals
    # -------------------------------------------------   
    
    print("Generating Static Visuals...")
    visualization.set_style()
    
    # Exploratory
    visualization.plot_missing_heatmap(df_clean, f"{OUT_DIR}/exploratory_missing_heatmap.png")
    visualization.plot_correlation_heatmap(df_clean, ['STATUS', 'BIKE_COST','BIKE_SPEED','OCC_HOUR','OCC_DAY','OCC_DOY'], f"{OUT_DIR}/exploratory_correlation_heatmap.png")
    visualization.plot_pair_plot(df_clean, ['BIKE_COST','BIKE_SPEED','OCC_HOUR','OCC_DAY','OCC_DOY'], 'STATUS', f"{OUT_DIR}/exploratory_pair_plot.png")
    visualization.plot_distributions_grid(df_clean, ['BIKE_COST','BIKE_SPEED','OCC_HOUR','OCC_DAY','OCC_DOY','BIKE_TYPE','PREMISES_TYPE','STATUS','OCC_DOW','OCC_MONTH'], f"{OUT_DIR}/exploratory_distributions_grid.png")
    visualization.plot_categorical_association_heatmap(df_clean, ['BIKE_MODEL','BIKE_MAKE','BIKE_TYPE','BIKE_COLOUR','PREMISES_TYPE','LOCATION_TYPE','STATUS','NEIGHBOURHOOD','OCC_DOW','OCC_MONTH'], f"{OUT_DIR}/exploratory_categorical_correlation_heatmap.png")
    visualization.plot_parallel_categories(df_clean, ['BIKE_MODEL','BIKE_MAKE','BIKE_TYPE','BIKE_COLOUR','PREMISES_TYPE','LOCATION_TYPE','STATUS','NEIGHBOURHOOD','OCC_DOW','OCC_MONTH'],'STATUS', f"{OUT_DIR}/exploratory_parallel_categories.html", max_cats=5)

    # Temporal
    visualization.plot_temporal_overview(df, f"{OUT_DIR}/temporal_overview.png")
    visualization.plot_dow_heatmap(df, f"{OUT_DIR}/temporal_day_vs_hr_heatmap.png")
    visualization.plot_time_series_decomposition(df_clean, 'OCC_DATE', 'W', f"{OUT_DIR}/temporal_time_series_decomposition.png")
    start_date = '2017-01-01'
    end_date = '2020-12-31'
    df_subset = df_clean[(df_clean['OCC_DATE'] >= start_date) & (df_clean['OCC_DATE'] <= end_date)].copy()
    visualization.plot_time_series_decomposition(df_subset, 'OCC_DATE', 'W', f"{OUT_DIR}/temporal_time_series_decomposition_2017_2020.png")
    visualization.plot_anomaly_detection(df_subset,'OCC_DATE',f"{OUT_DIR}/temporal_anomaly_detection_spikes_2017_2020.png")
    visualization.plot_interactive_line(df_clean,'OCC_DATE',f"{OUT_DIR}/temporal_interactive_theft_timeline.html")
                                       
    # Spatial
    visualization.plot_folium_map(df_clean,'LAT_WGS84','LONG_WGS84',5000,f"{OUT_DIR}/spatial_interactive_folium_map.html")
    visualization.plot_interactive_scatter_map(df_clean,f"{OUT_DIR}/spatial_interactive_neighbourhood_map.html")
    visualization.plot_network_graph(df_clean,'PREMISES_TYPE','NEIGHBOURHOOD',20,f"{OUT_DIR}/spatial_network_repeat_premises_neighbourhood.png")

    # Characteristics
    visualization.plot_word_cloud(df_clean,'LOCATION_TYPE',f"{OUT_DIR}/char_location_type_word_cloud.png")
    visualization.plot_word_cloud(df_clean,'BIKE_MAKE',f"{OUT_DIR}/char_bike_make_word_cloud.png")
    visualization.plot_word_cloud(df_clean,'BIKE_MODEL',f"{OUT_DIR}/char_bike_model_word_cloud.png")
    visualization.plot_static_bubble_chart(df_clean,'BIKE_MAKE','LOCATION_TYPE','STATUS',f"{OUT_DIR}/char_bubble_chart.png")
    visualization.plot_3d_scatter(df_clean,'BIKE_COST','BIKE_SPEED','OCC_HOUR','STATUS',f"{OUT_DIR}/char_interactive_3d_scatter.html")

    # Status and reporting
    visualization.plot_status_pie(df_clean,f"{OUT_DIR}/status_pie.png")
    visualization.plot_recovery_rate_by_delay(df_clean, f"{OUT_DIR}/status_recovery_by_delay.png")
    
    # Machine learning
    print("Training Models...")
    ml, roc, cal = modeling.train_and_evaluate_models(df)
    visualization.plot_ml_metrics_heatmap(ml, f"{OUT_DIR}/machinel_ml_metrics.png")
    visualization.plot_roc_curves(roc, f"{OUT_DIR}/machinel_roc.png")
    visualization.plot_calibration_curves(cal, f"{OUT_DIR}/machinel_calibration_curves.png")

    print(f"\n[DONE] Batch Report saved to: {os.path.abspath(OUT_DIR)}")
    
    # Reset Batch Mode so GUI works if chosen next
    visualization.BATCH_MODE = False

def main():
    while True:
        print("\n=========================================")
        print("   BICYCLE THEFT ANALYSIS TOOLKIT        ")
        print("=========================================")
        print("1. Launch Interactive Dashboard (GUI)")
        print("2. Generate Static Report (Batch Output)")
        print("3. Exit")
        print("=========================================")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            print("\n>> Launching GUI...")
            # Ensure Batch Mode is OFF for GUI
            visualization.BATCH_MODE = False
            gui.launch_dashboard()
            
        elif choice == '2':
            print("\n>> Starting Batch Process...")
            run_batch_report()
            
        elif choice == '3':
            print("\nExiting program. Goodbye!")
            sys.exit()
            
        else:
            print("\n[!] Invalid choice. Please try again.")

if __name__ == "__main__":
    main()


# In[ ]:


# --- Main execution --- 

run_batch_report()

