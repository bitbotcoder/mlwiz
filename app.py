try:
    import streamlit as st
    from ml_pipeline import load_data, eda_report, build_model, download_model,load_pycaret_dataset, get_all_datasets, handle_exception
    from st_social_media_links import SocialMediaIcons
    import streamlit.components.v1 as components
    import traceback
    
    VERSION = "0.5.4"

    # Title and description
    st.set_page_config(
        page_title="MLWiz - AutoML WorkBench",
        page_icon="🤖",
        menu_items={
            "About": f"MLWize v{VERSION}"
            f"\nApp contact: [Sumit Khanna](https://github.com/bitbotcoder/)",
            "Report a Bug": "https://github.com/bitbotcoder/mlwiz/issues/new",
            "Get help": None,
        },
        layout="wide",
    )
    st.subheader("🤖 MLWiz - Automating ML Tasks")
    st.divider()
    with st.sidebar: 
        st.image('logo.png', width=150)
        st.write("🔠 Supported Features")
        st.caption("""
                   - ✅ Datasets (Custom, PyCaret(disabled))
                   - ✅ Data Profiling & EDA
                   - ✅ Build ML Models 
                   - ✅ Regression 
                   - ✅ Classification 
                   - ✅ Clustering 
                   - ✅ Time Series Forecasting 
                   - ✅ Anamoly Detection 
                   - ✅ Download Models
                   """)
        st.divider()    
        st.write("📢 Share with wider community")
        social_media_links = [
                "https://x.com/intent/tweet?hashtags=streamlit%2Cpython&text=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app",
                "https://www.linkedin.com/sharing/share-offsite/?summary=https%3A%2F%2Fautoml-wiz.streamlit.app%20%23streamlit%20%23python&title=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app",
                "https://www.facebook.com/sharer/sharer.php?kid_directed_site=0&u=https%3A%2F%2Fautoml-wiz.streamlit.app",
                "https://github.com/bitbotcoder/mlwiz"
            ]

        social_media_icons = SocialMediaIcons(social_media_links, colors=["white"] * len(social_media_links))

        social_media_icons.render(sidebar=True)
        
    #Tasks based on user selection
    tab1, tab2, tab3, tab4, = st.tabs(["📑Choose Dataset", "📊Perform EDA", "🧠Build Model", "📩Download Model"])

    with tab1:
        c1, c2 = st.columns([1,2])
        c1.write("Upload Custom Dataset files")
        #dataset_source = c1.radio("Select Dataset Source", options=["PyCaret", "Upload"], captions=["Load PyCaret Datasets", "Upload Custom Dataset files"])
        #if dataset_source == "PyCaret":
        #    pycaret_datasets = get_all_datasets()
        #    selected_dataset = c2.selectbox("Select a dataset", pycaret_datasets)
        #    if c2.button("Load Dataset"):
        #        load_pycaret_dataset(selected_dataset)
        #elif dataset_source == "Upload":
        uploaded_file = c2.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            load_data(uploaded_file)
        
    with tab2:
        eda_report()
    with tab3:
        st.write("**Configure ML Model**")
        col1,col2 = st.columns([0.4,0.7])
        task = col1.selectbox("Select ML task", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series Forecasting"])
        build_model(task,col2)
    with tab4:
        download_model(task)
except Exception as e:
        handle_exception(e)

st.success(
    "Show your 💘 ➡️ [Star the repo](https://github.com/bitbotcoder/mlwiz/)",
    icon="ℹ️",
)