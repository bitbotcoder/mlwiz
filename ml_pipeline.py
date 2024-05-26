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
import os



def get_all_datasets():
    df = get_data('index')
    return df['Dataset'].to_list()

def show_profile_reports(container):
    if os.path.exists("profile_report.html"):
        with open('profile_report.html', 'r') as f:
            html_content = f.read()
        with container:
            components.html(html_content, height=800, scrolling=True)
    if os.path.exists("sweetviz_report.html"):
        with open('sweetviz_report.html', 'r') as f:
            html_content = f.read()
        with container:
            components.html(html_content, height=800, scrolling=True)

def data_profile(df,container):
    profile = ProfileReport(df)
    profile.to_file("profile_report.html")
    with open('profile_report.html', 'r') as f:
        html_content = f.read()
    with container:
        components.html(html_content, height=800, scrolling=True)
    
def update_progress(progress_bar, step, max_steps):
    progress = int((step / max_steps) * 100)
    t = f"Processing....Step {step}/{max_steps}"
    if step == max_steps:
        t="Process Completed"
    progress_bar.progress(progress, text=t)

def display_sweetviz_report(dataframe,container):
    report = sv.analyze(dataframe)
    report.show_html('sweetviz_report.html', open_browser=False)
    with open('sweetviz_report.html', 'r') as f:
        html_content = f.read()
    with container:
        components.html(html_content, height=800, scrolling=True)

def handle_exception(e):
    st.error(
        f"""The app has encountered an error:  
            **{e}**  
            Please check settings - columns selections and model parameters  
            Or
            Create an issue [here](https://github.com/bitbotcoder/mlwiz/issues/new) with the below error details
        """,
        icon="🥺",
    )
    with st.expander("See Error details"):
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


def eda_report():
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        col1,col2 = st.columns([0.6,0.4])
        new_report = col1.toggle(":blue[Generate New]", value=True)
        show_button = col2.button("Show Report")
        pb = st.progress(0, text="Generating Report")
        cont = st.container(border=False)
        try:
            if show_button:
                if new_report:
                    update_progress(pb,1,4)
                    data_profile(df, cont)
                    update_progress(pb,2,4)
                    display_sweetviz_report(df, cont)
                    update_progress(pb,4,4)
                else:
                    show_profile_reports(cont)

        except Exception as e:
            handle_exception(e)


def build_model(task, container):
    
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        feature_expander = container.expander("Select Columns")
        target_column = feature_expander.selectbox("Select target column", df.columns) if task in ["Classification", "Regression", "Time Series Forecasting"] else None
        numerical_columns = feature_expander.multiselect("Select numerical columns", df.columns)
        categorical_columns = feature_expander.multiselect("Select categorical columns", df.columns)

        params_expander = container.expander("Tune Parameters")
        # Data Preparation
        handle_missing_data = params_expander.toggle("Handle Missing Data", value=True)
        handle_outliers = params_expander.toggle("Handle Outliers", value=True)
        
        # Scale and Transform
        normalize = params_expander.checkbox("Normalize", value=False)
        normalize_method = params_expander.selectbox("Normalize Method", ["zscore", "minmax", "maxabs", "robust"], index=0 if normalize else -1) if normalize else None
        transformation = params_expander.checkbox("Apply Transformation", value=False)
        transformation_method = params_expander.selectbox("Transformation Method", ["yeo-johnson", "quantile"], index=0 if transformation else -1) if transformation else None
        
        # Feature Engineering
        polynomial_features = params_expander.checkbox("Polynomial Features", value=False)
        polynomial_degree = params_expander.slider("Polynomial Degree", 2, 5, 2) if polynomial_features else None
        
        # Feature Selection
        remove_multicollinearity = params_expander.checkbox("Remove Multicollinearity", value=False)
        multicollinearity_threshold = params_expander.slider("Multicollinearity Threshold", 0.5, 1.0, 0.9) if remove_multicollinearity else None
        
        if not (task == "Anomaly Detection" or task == "Clustering") :
            feature_selection = params_expander.checkbox("Feature Selection", value=False)
            feature_selection_method = params_expander.selectbox("Feature Selection Method", ["classic", "exhaustive"], index=0 if feature_selection else -1) if feature_selection else None
        else:
            feature_selection = None
            feature_selection_method = None
                
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
            pb = st.progress(0, text="Building Model...")

            if task == "Classification" and st.button("Run Classification"):
                
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

            elif task == "Regression" and st.button("Run Regression"):
                update_progress(pb,1,7)
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                update_progress(pb,2,7)
                df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,3,7)                
                if len(df) < 2:
                    st.error("Not enough data to split into train and test sets.")
                    return
                
                exp = reg_setup(**setup_kwargs)
                best_model = reg_compare()
                update_progress(pb,4,7)
                st.dataframe(reg_pull())
                update_progress(pb,5,7)
                st.image(reg_plot(best_model, plot='residuals', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                update_progress(pb,6,7)
                reg_save(best_model, 'best_regression_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)
            elif task == "Clustering" and st.button("Run Clustering"):
                update_progress(pb,1,7)
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,2,7)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')  
                update_progress(pb,3,7)
                exp = clu_setup(**setup_kwargs)
                best_model = clu_create('kmeans')
                update_progress(pb,4,7)
                clu_plot(best_model, plot='cluster', display_format='streamlit')
                clu_plot(best_model, plot='elbow', display_format='streamlit')
                update_progress(pb,5,7)
                st.write(best_model)
                st.dataframe(clu_pull())
                update_progress(pb,6,7)
                clu_save(best_model, 'best_clustering_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)

            elif task == "Anomaly Detection" and st.button("Run Anomaly Detection"):
                update_progress(pb,1,7)
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,2,7)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')        
                update_progress(pb,3,7)
                exp = ano_setup(**setup_kwargs)
                best_model = ano_create('iforest')
                update_progress(pb,4,7)
                ano_plot(best_model, plot='tsne', display_format='streamlit')
                update_progress(pb,5,7)                
                st.write(best_model)
                st.dataframe(ano_pull())
                update_progress(pb,6,7)
                ano_save(best_model, 'best_anomaly_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)
            elif task == "Time Series Forecasting" :
                date_column = feature_expander.selectbox("Select date column", df.columns)
                if st.button("Run Time Series Forecasting"):
                    update_progress(pb,1,5)
                    df[date_column] = pd.to_datetime(df[date_column])
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                    df.dropna(subset=[target_column], inplace=True)
                    update_progress(pb,2,5)                
                    df = df.set_index(date_column).asfreq('D')
                    exp = ts_setup(df, target=target_column, numeric_imputation_target='mean', numeric_imputation_exogenous='mean')
                    best_model = ts_compare()
                    update_progress(pb,3,5)                    
                    st.dataframe(ts_pull())
                    ts_plot(best_model, plot='forecast', display_format="streamlit")
                    ts_save(best_model, 'best_timeseries_model')
                    update_progress(pb,4,5)
                    st.write('Best Model based on metrics - ')
                    st.write(best_model)
                    update_progress(pb,5,5)
        except Exception as e:
            handle_exception(e)

def download_model(task):
    model_file = None
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
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    st.download_button('Download Model', f, file_name=model_file)
            except Exception as e:
                handle_exception(e)
        else:
            st.error("❗No File Found | First Build A ML Model ")
