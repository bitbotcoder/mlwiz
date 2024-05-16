import streamlit as st
from ml_pipeline import load_data, generate_eda_report, build_model, download_model,load_pycaret_dataset, get_all_datasets

# Title and description
st.title("🤖 MLWiz ")
st.info("Streamline your ML pipelines with our low-code app, automating EDA, data profiling, and ML model building effortlessly.")
st.divider()
st.sidebar.image('logo.png', width=200)
st.sidebar.divider()
# Sidebar for navigation
st.sidebar.title("Navigation")

option = st.sidebar.selectbox("Choose a step", ["Choose Dataset", "Perform EDA", "Build Model", "Download Model"])

# Steps based on user selection
if option == "Choose Dataset":
    dataset_source = st.radio("Select Dataset Source", ["Upload", "PyCaret"])
    if dataset_source == "Upload":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            load_data(uploaded_file)
    elif dataset_source == "PyCaret":
        pycaret_datasets = get_all_datasets()#["index", "boston", "diabetes", "blood", "bupa", "diamond"]
        selected_dataset = st.selectbox("Select a dataset", pycaret_datasets)
        if st.button("Load Dataset"):
            load_pycaret_dataset(selected_dataset)
elif option == "Perform EDA":
    generate_eda_report()
elif option == "Build Model":
    build_model()
elif option == "Download Model":
    download_model()
