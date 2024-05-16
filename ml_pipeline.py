import streamlit as st
import pandas as pd
import sweetviz as sv
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save, pull as cls_pull, plot_model as cls_plot
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, pull as reg_pull, plot_model as reg_plot
from pycaret.clustering import setup as clu_setup, create_model as clu_create, plot_model as clu_plot, save_model as clu_save, pull as clu_pull
from pycaret.anomaly import setup as ano_setup, create_model as ano_create, plot_model as ano_plot, save_model as ano_save, pull as ano_pull
from pycaret.time_series import setup as ts_setup, compare_models as ts_compare, save_model as ts_save, pull as ts_pull, plot_model as ts_plot
from pycaret.datasets import get_data
import streamlit.components.v1 as components
import traceback
from ydata_profiling import ProfileReport


def get_all_datasets():
    df = get_data('index')
    return df['Dataset'].to_list()
    
def data_profile(df):
    profile = ProfileReport(df)
    profile.to_file("profile_report.html")
    with open('profile_report.html', 'r') as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
    
def update_progress(progress_bar, step, max_steps):
    progress = int((step / max_steps) * 100)
    t = "Processing...."
    if step == max_steps:
        t="Process Completed"
    progress_bar.progress(progress, text=t)

def display_sweetviz_report(dataframe):
    report = sv.analyze(dataframe)
    report.show_html('sweetviz_report.html', open_browser=False)
    with open('sweetviz_report.html', 'r') as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)

def handle_exception(e):
    st.error(f"An error occurred: {str(e)}")
    with st.expander("See details"):
        st.error(traceback.format_exc())

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.write("## Dataset")
        st.write(df.head())
        st.session_state['dataframe'] = df
    except Exception as e:
        handle_exception(e)

def load_pycaret_dataset(dataset_name):
    try:
        df = get_data(dataset_name)
        st.write("## Dataset")
        st.write(df.head())
        st.session_state['dataframe'] = df
    except Exception as e:
        handle_exception(e)


def generate_eda_report():
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        pb = st.progress(0, text="Generating Report")
        try:
            if st.sidebar.button("Generate EDA Report"):
                update_progress(pb,1,4)
                data_profile(df)
                update_progress(pb,2,4)
                display_sweetviz_report(df)
                update_progress(pb,4,4)

        except Exception as e:
            handle_exception(e)

def build_model():
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        task = st.sidebar.selectbox("Select ML task", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series Forecasting"])
        target_column = st.sidebar.selectbox("Select target column", df.columns) if task in ["Classification", "Regression", "Time Series Forecasting"] else None
        numerical_columns = st.sidebar.multiselect("Select numerical columns", df.columns)
        categorical_columns = st.sidebar.multiselect("Select categorical columns", df.columns)

        # Data Preparation
        handle_missing_data = st.sidebar.checkbox("Automatic Missing Data Handling", value=True)
        handle_outliers = st.sidebar.checkbox("Automatic Outlier Handling", value=True)
        
        # Scale and Transform
        normalize = st.sidebar.checkbox("Normalize", value=False)
        normalize_method = st.sidebar.selectbox("Normalize Method", ["zscore", "minmax", "maxabs", "robust"], index=0 if normalize else -1) if normalize else None
        transformation = st.sidebar.checkbox("Apply Transformation", value=False)
        transformation_method = st.sidebar.selectbox("Transformation Method", ["yeo-johnson", "quantile"], index=0 if transformation else -1) if transformation else None
        
        # Feature Engineering
        polynomial_features = st.sidebar.checkbox("Polynomial Features", value=False)
        polynomial_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2) if polynomial_features else None
        
        # Feature Selection
        remove_multicollinearity = st.sidebar.checkbox("Remove Multicollinearity", value=False)
        multicollinearity_threshold = st.sidebar.slider("Multicollinearity Threshold", 0.5, 1.0, 0.9) if remove_multicollinearity else None
        
        if not (task == "Anomaly Detection" or task == "Clustering") :
            feature_selection = st.sidebar.checkbox("Feature Selection", value=False)
            feature_selection_method = st.sidebar.selectbox("Feature Selection Method", ["classic", "exhaustive"], index=0 if feature_selection else -1) if feature_selection else None
        else:
            feature_selection = None
            feature_selection_method = None
        
        pb = st.progress(0, text="Building Model")
        try:
            # Setup arguments for PyCaret
            setup_kwargs = {
                'data': df[numerical_columns + categorical_columns + ([target_column] if target_column else [])],
                'categorical_features': categorical_columns,
                'numeric_features': numerical_columns,
                'target': target_column,
                'preprocess': handle_missing_data,
                'remove_outliers': handle_outliers,
                'normalize': normalize,
                'normalize_method': normalize_method,
                'transformation': transformation,
                'transformation_method': transformation_method,
                'polynomial_features': polynomial_features,
                'polynomial_degree': polynomial_degree,
                'remove_multicollinearity': remove_multicollinearity,
                'multicollinearity_threshold': multicollinearity_threshold,
                'feature_selection': feature_selection,
                'feature_selection_method': feature_selection_method
            }
            
            if task == "Classification" and st.sidebar.button("Run Classification"):
                
                df[target_column] = df[target_column].astype('category')
                
                df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                
                if len(df) < 2:
                    st.error("Not enough data to split into train and test sets.")
                    return
                update_progress(pb,1,7)
                exp = cls_setup(**setup_kwargs)
                update_progress(pb,2,7)
                best_model = cls_compare()
                update_progress(pb,3,7)
                st.dataframe(cls_pull())
                update_progress(pb,4,7)
                cls_plot(best_model, plot='auc',display_format="streamlit")
                cls_plot(best_model, plot='confusion_matrix',display_format="streamlit")
                update_progress(pb,5,7)
                st.image(cls_plot(best_model, plot='pr',save=True))
                update_progress(pb,6,7)
                cls_save(best_model, 'best_classification_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)

            elif task == "Regression" and st.sidebar.button("Run Regression"):
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                
                df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                
                if len(df) < 2:
                    st.error("Not enough data to split into train and test sets.")
                    return
                
                exp = reg_setup(**setup_kwargs)
                best_model = reg_compare()
                st.dataframe(reg_pull())
                st.image(reg_plot(best_model, plot='residuals', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                reg_save(best_model, 'best_regression_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
            elif task == "Clustering" and st.sidebar.button("Run Clustering"):
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')  
                exp = clu_setup(**setup_kwargs)
                best_model = clu_create('kmeans')
                clu_plot(best_model, plot='cluster', display_format='streamlit')
                clu_plot(best_model, plot='elbow', display_format='streamlit')
                st.write(best_model)
                st.dataframe(clu_pull())
                clu_save(best_model, 'best_clustering_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)

            elif task == "Anomaly Detection" and st.sidebar.button("Run Anomaly Detection"):
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')        
                exp = ano_setup(**setup_kwargs)
                best_model = ano_create('iforest')
                ano_plot(best_model, plot='tsne', display_format='streamlit')
                #ano_plot(best_model, plot='umap', display_format='streamlit')
                st.write(best_model)
                st.dataframe(ano_pull())
                ano_save(best_model, 'best_anomaly_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                
            elif task == "Time Series Forecasting" :
                date_column = st.sidebar.selectbox("Select date column", df.columns)
                if st.sidebar.button("Run Time Series Forecasting"):
                    df[date_column] = pd.to_datetime(df[date_column])
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                    df.dropna(subset=[target_column], inplace=True)
                    df = df.set_index(date_column).asfreq('D')
                    exp = ts_setup(df, target=target_column, numeric_imputation_target='mean', numeric_imputation_exogenous='mean')
                    best_model = ts_compare()
                    st.dataframe(ts_pull())
                    ts_plot(best_model, plot='forecast', display_format="streamlit")
                    ts_save(best_model, 'best_timeseries_model')
                    st.write('Best Model based on metrics - ')
                    st.write(best_model)

        except Exception as e:
            handle_exception(e)

def download_model():
    model_file = None
    task = st.sidebar.selectbox("Select ML task", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series Forecasting"])
    if task == "Classification":
        model_file = 'best_classification_model.pkl'
    elif task == "Regression":
        model_file = 'best_regression_model.pkl'
    elif task == "Clustering":
        model_file = 'best_clustering_model.pkl'
    elif task == "Anomaly Detection":
        model_file = 'best_anomaly_model.pkl'
    elif task == "Time Series Forecasting":
        model_file = 'best_timeseries_model.pkl'
    
    if model_file:
        try:
            with open(model_file, 'rb') as f:
                st.download_button('Download Model', f, file_name=model_file)
        except Exception as e:
            handle_exception(e)
