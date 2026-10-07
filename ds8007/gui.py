#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;">GUI Customizations for the Toronto Bicycle Thefts Dataset</h2>

# In[1]:


"""
Name: Jessie Ma – Student ID: 501274167
Maliha Saeed – Student ID: 501304501
"""


# In[2]:


# --- Import libraries and files ---
#import sys
#!{sys.executable} -m pip install PyQt5
import sys
import os
import traceback
import functools
import datetime
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import io 
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
import preprocessing
import visualization
import modeling
import eda


# In[3]:


# --- Healper function for time series ---
def _subset_decomposition_helper(df, date_col, freq, filename):
    pass


# In[4]:


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bicycle Theft Analysis Dashboard")
        self.setGeometry(100, 100, 600, 500)
        self.df = None
        self.df_cost_filtered = None
        
        # Output folder for GUI saves
        self.output_dir = "batch_visuals_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.initUI()
  
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Header
        header = QHBoxLayout()
        self.btn_load = QPushButton("Load Dataset")
        self.btn_load.clicked.connect(self.load_data)
        header.addWidget(self.btn_load)
        self.lbl_status = QLabel("Status: Waiting for Data...")
        header.addWidget(self.lbl_status)
        main_layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setEnabled(False)
        self.create_tabs()
        main_layout.addWidget(self.tabs)

    def create_tabs(self):
        
        # --- Common column lists from main.py ---
        # Numerical/status columns for correlation/pair plots
        NUM_COLS = ['BIKE_COST','BIKE_SPEED','OCC_HOUR','OCC_DAY','OCC_DOY']
        CORR_COLS = ['STATUS'] + NUM_COLS
        
        # Categorical columns for association/parallel plots
        CAT_COLS = ['BIKE_MODEL','BIKE_MAKE','BIKE_TYPE','BIKE_COLOUR','PREMISES_TYPE','LOCATION_TYPE','STATUS','NEIGHBOURHOOD','OCC_DOW','OCC_MONTH']

        # ------------------------------------------------- 
        # 1. Exploratory
        # ------------------------------------------------- 
        
        tab_exp = QWidget(); l_exp = QVBoxLayout()
        self.add_btn(l_exp, "Missing Values Heatmap", self.run_viz, visualization.plot_missing_heatmap)
        self.add_btn(l_exp, "Correlation Heatmap", self.run_viz, functools.partial(visualization.plot_correlation_heatmap, cols=CORR_COLS), use_cost_df=True)
        self.add_btn(l_exp, "Pair Plot of Key Variables", self.run_viz, functools.partial(visualization.plot_pair_plot, cols=NUM_COLS, hue_col='STATUS'), use_cost_df=True)
        self.add_btn(l_exp, "Distributions Grid", self.run_viz, functools.partial(visualization.plot_distributions_grid, cols=NUM_COLS + ['BIKE_TYPE','PREMISES_TYPE','STATUS','OCC_DOW','OCC_MONTH']), use_cost_df=True)
        self.add_btn(l_exp, "Categorical Association Heatmap", self.run_viz, functools.partial(visualization.plot_categorical_association_heatmap, cols=CAT_COLS))
        self.add_btn(l_exp, "Parallel Categories Plot (Top 5)", self.run_viz, functools.partial(visualization.plot_parallel_categories, cols=CAT_COLS, color_col='STATUS', max_cats=5), is_html=True)
        tab_exp.setLayout(l_exp); self.tabs.addTab(tab_exp, "Exploratory")

        # ------------------------------------------------- 
        # 2. Temporal
        # ------------------------------------------------- 
        
        tab_temp = QWidget(); l_temp = QVBoxLayout()
        self.add_btn(l_temp, "Temporal Overview", self.run_viz, visualization.plot_temporal_overview)
        self.add_btn(l_temp, "Day vs Hour Heatmap", self.run_viz, visualization.plot_dow_heatmap)
        self.add_btn(l_temp, "Trend Decomposition (Full Span)", self.run_viz, functools.partial(visualization.plot_time_series_decomposition, freq='W'))
        self.add_btn(l_temp, "Trend Decomposition (2017-2020)", self.run_viz, functools.partial(_subset_decomposition_helper, date_col='OCC_DATE', freq='W'))
        self.add_btn(l_temp, "Anomaly Detection (2017-2020)", self.run_viz, visualization.plot_anomaly_detection)
        self.add_btn(l_temp, "Interactive Timeline (HTML)", self.run_viz, visualization.plot_interactive_line, is_html=True)
        tab_temp.setLayout(l_temp); self.tabs.addTab(tab_temp, "Temporal")

        # ------------------------------------------------- 
        # 3. Spatial
        # ------------------------------------------------- 
        
        tab_spat = QWidget(); l_spat = QVBoxLayout()
        self.add_btn(l_spat, "Folium Map (Interactive)", self.run_viz, functools.partial(visualization.plot_folium_map, lat_col='LAT_WGS84', lon_col='LONG_WGS84', sample_n=5000), is_html=True)
        self.add_btn(l_spat, "Interactive Clusters (Plotly)", self.run_viz, visualization.plot_interactive_scatter_map, is_html=True)
        self.add_btn(l_spat, "Network Graph (Premises-Neighborhood)", self.run_viz, functools.partial(visualization.plot_network_graph, source_col='PREMISES_TYPE', target_col='NEIGHBOURHOOD', min_weight=20))
        tab_spat.setLayout(l_spat); self.tabs.addTab(tab_spat, "Spatial")

        # ------------------------------------------------- 
        # 4. Characteristics and status
        # ------------------------------------------------- 
        
        tab_char = QWidget(); l_char = QVBoxLayout()
        self.add_btn(l_char, "Word Cloud (Location Type)", self.run_viz, functools.partial(visualization.plot_word_cloud, text_col='LOCATION_TYPE'))
        self.add_btn(l_char, "Word Cloud (Bike Make)", self.run_viz, functools.partial(visualization.plot_word_cloud, text_col='BIKE_MAKE'))
        self.add_btn(l_char, "Word Cloud (Bike Model)", self.run_viz, functools.partial(visualization.plot_word_cloud, text_col='BIKE_MODEL'))
        self.add_btn(l_char, "Static Bubble Chart (Top N)", self.run_viz, functools.partial(visualization.plot_static_bubble_chart, x_col='BIKE_MAKE', y_col='LOCATION_TYPE', color_col='STATUS'), use_cost_df=False)
        self.add_btn(l_char, "3D Scatter Plot (Cost, Speed, Hour)", self.run_viz, functools.partial(visualization.plot_3d_scatter, x_col='BIKE_COST', y_col='BIKE_SPEED', z_col='OCC_HOUR', color_col='STATUS'), use_cost_df=True, is_html=True)
        self.add_btn(l_char, "Status Pie Chart", self.run_viz, visualization.plot_status_pie)
        self.add_btn(l_char, "Recovery Rate by Delay", self.run_viz, visualization.plot_recovery_rate_by_delay)
        tab_char.setLayout(l_char); self.tabs.addTab(tab_char, "Characteristics")

        # ------------------------------------------------- 
        # 5. ML
        # ------------------------------------------------- 
        
        tab_ml = QWidget(); l_ml = QVBoxLayout()
        self.btn_train = QPushButton("Train Models"); self.btn_train.clicked.connect(self.run_ml_pipeline)
        l_ml.addWidget(self.btn_train)
        self.add_btn(l_ml, "Model Metrics", self.run_viz_ml, visualization.plot_ml_metrics_heatmap, enabled=False)
        self.add_btn(l_ml, "ROC Curves", self.run_viz_ml, visualization.plot_roc_curves, enabled=False)
        self.add_btn(l_ml, "Calibration Curves", self.run_viz_ml, visualization.plot_calibration_curves, enabled=False)
        tab_ml.setLayout(l_ml); self.tabs.addTab(tab_ml, "ML")

    def add_btn(self, layout, text, handler, func, use_cost_df=False, is_html=False, enabled=True):
        btn = QPushButton(text)
        btn.setEnabled(enabled)
        btn.clicked.connect(functools.partial(handler, func, use_cost_df, is_html))
        layout.addWidget(btn)
        if not enabled:
            if not hasattr(self, 'ml_buttons'): self.ml_buttons = []
            self.ml_buttons.append(btn)

    def generate_filename(self, func, is_html):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = func.func.__name__ if isinstance(func, functools.partial) else func.__name__ if hasattr(func, '__name__') else "plot"
        ext = "html" if is_html else "png"
        return f"{self.output_dir}/gui_{name}_{ts}.{ext}"

    def run_viz(self, func, use_cost_df, is_html):
        try:
            filename = self.generate_filename(func, is_html)
            df_to_use = self.df_cost_filtered if use_cost_df else self.df
            
            is_subset_required = (
                (isinstance(func, functools.partial) and func.func == _subset_decomposition_helper) or 
                (func == visualization.plot_anomaly_detection)
            )
            
            if is_subset_required:
                # Subset data
                df_to_use = df_to_use[
                    (df_to_use['OCC_DATE'] >= '2017-01-01') & 
                    (df_to_use['OCC_DATE'] <= '2020-12-31')
                ].copy()
                
                # Execute the correct underlying function with subsetted data
                if isinstance(func, functools.partial) and func.func == _subset_decomposition_helper:
                    # Execute the real decomposition function using the arguments passed in functools.partial
                    visualization.plot_time_series_decomposition(
                        df_to_use, 
                        date_col='OCC_DATE', 
                        freq=func.keywords.get('freq', 'W'), # Get freq from partial or use default
                        filename=filename
                    )
                else:
                    # Execute anomaly detection
                    func(df_to_use, date_col='OCC_DATE', filename=filename)
            else:
                # Default call for all other functions
                func(df_to_use, filename=filename) 
                
            self.lbl_status.setText(f"Saved & Shown: {os.path.basename(filename)}")
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))
            
    def run_viz_ml(self, func, _u1, _u2):
        try:
            data_map = {visualization.plot_ml_metrics_heatmap: self.ml_results,
                        visualization.plot_roc_curves: self.roc_data,
                        visualization.plot_calibration_curves: self.cal_data}
            filename = self.generate_filename(func, False)
            func(data_map[func], filename=filename)
            self.lbl_status.setText(f"Saved & Shown: {os.path.basename(filename)}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def load_data(self):
        import io
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '.', 'CSV Files (*.csv)')
        if fname:
            # Temporarily redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                self.lbl_status.setText("Processing...")
                QApplication.processEvents()
                
                # --- Execute preprocessing and EDA (output captured) ---
                raw_df = preprocessing.load_data(fname)
                self.df = preprocessing.clean_data(raw_df) 
                self.df_cost_filtered = preprocessing.treat_cost_outliers_iqr(self.df)
                eda.print_eda_report(self.df)
                
                # --- Restore stdout and capture text ---
                sys.stdout = old_stdout
                report_text = buffer.getvalue()
                
                # --- Display report in pop-up ---
                msg = QMessageBox()
                msg.setWindowTitle("Preprocessing and EDA Report")
                msg.setText("Data successfully loaded and processed. See report below:")
                # Using setDetailedText for scrollable, multi-line output
                msg.setDetailedText(report_text) 
                msg.setIcon(QMessageBox.Information)
                msg.exec_()

                # --- Finalize GUI state ---
                self.lbl_status.setText("Data Loaded"); 
                self.tabs.setEnabled(True)

            except Exception as e:
                # Ensure stdout is restored even if an error occurs
                sys.stdout = old_stdout 
                self.lbl_status.setText("Error"); 
                QMessageBox.critical(self, "Error", str(e))

    def run_ml_pipeline(self):
        try:
            self.lbl_status.setText("Training...")
            QApplication.processEvents()
            self.ml_results, self.roc_data, self.cal_data = modeling.train_and_evaluate_models(self.df)
            for btn in self.ml_buttons: btn.setEnabled(True)
            self.lbl_status.setText("Training Complete")
            QMessageBox.information(self, "Done", "Models Trained!")
        except Exception as e:
            self.lbl_status.setText("Error"); QMessageBox.critical(self, "Error", str(e))

def launch_dashboard():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    launch_dashboard()

