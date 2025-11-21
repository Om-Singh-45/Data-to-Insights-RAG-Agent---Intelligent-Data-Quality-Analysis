# app.py - Enhanced UI with Intelligent Data Cleaning & Analyst Dashboard
import os
from pathlib import Path
import shutil
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import io
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots

from rag_utils import (
    build_or_load_index,
    answer_question,
    save_dataframe,
    load_dataframe,
    clean_dataset,
    create_data_quality_heatmap,
    create_missing_value_pattern,
    create_kpi_dashboard,
    create_distribution_pie_chart,
    create_trend_line_chart,
    create_box_plot,
    create_histogram_analysis,
)

# Load env
load_dotenv()

# Custom CSS
st.set_page_config(page_title="Data-to-Insights RAG Agent", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .main-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1.1em;
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 25px 0 15px 0;
        color: white;
        font-weight: bold;
        font-size: 1.2em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Data-to-Insights RAG Agent</h1>
    <p>Transform your data into actionable insights with AI-powered analysis</p>
    <p style="font-size: 0.9em; opacity: 0.8; margin-top: 10px;">Upload your dataset to automatically clean data, detect anomalies, and ask questions using natural language.</p>
</div>
""", unsafe_allow_html=True)

ROOT = Path.cwd()
UPLOAD_DIR = ROOT / "data" / "uploads"
PERSIST_DIR = ROOT / "chroma_store"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "index_created" not in st.session_state:
    st.session_state.index_created = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

# Add instructions for file upload
st.sidebar.info("💡 Tip: Upload a CSV or Excel file to get started, or use the sample dataset below.")

if st.sidebar.button("Load sample dataset"):
    sample_path = ROOT / "sample_data.csv"
    if sample_path.exists():
        uploaded_file = str(sample_path)
        st.sidebar.success("Sample dataset selected! Click 'Upload Dataset' to load it.")
    else:
        st.sidebar.warning(f"Sample dataset not found at: {sample_path}")

df = None
if uploaded_file:
    file_name = uploaded_file.name.lower() if not isinstance(uploaded_file, str) else uploaded_file.lower()
    is_new_file = file_name != st.session_state.current_file_name
    
    if is_new_file and st.session_state.index_created:
        st.info("New dataset detected. Clearing old index...")
        try:
            if PERSIST_DIR.exists():
                shutil.rmtree(PERSIST_DIR)
                PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            st.session_state.index_created = False
            st.session_state.df = None
        except Exception as e:
            st.warning(f"Could not delete old index: {e}")
    
    if is_new_file:
        st.session_state.current_file_name = file_name

    try:
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                except Exception:
                    df = pd.read_csv(uploaded_file, encoding="cp1252")

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        if df is not None:
            st.success("Dataset loaded successfully!")
            
            cleaning_report = None
            if is_new_file or st.session_state.df is None:
                # Create an expander to show the cleaning process
                cleaning_expander = st.expander("🧹 Data Cleaning Progress", expanded=True)
                cleaning_status = cleaning_expander.empty()
                
                with st.spinner("Intelligent data cleaning in progress..."):
                    cleaning_status.info("🔄 Initializing data cleaning process...")
                    
                    # We'll capture the print outputs from clean_dataset
                    import io
                    import contextlib
                    
                    # Capture print statements from clean_dataset
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        df, cleaning_report = clean_dataset(df)
                    
                    # Display the captured output in a more user-friendly way
                    output_lines = captured_output.getvalue().split('\n')
                    cleaning_steps = []
                    for line in output_lines:
                        if line.strip():
                            cleaning_steps.append(line.strip())
                    
                    # Display the steps with appropriate emojis
                    for i, step in enumerate(cleaning_steps):
                        if "INTELLIGENT DATA CLEANING" in step:
                            cleaning_status.info("🧹 " + step.replace("🧹 ", ""))
                        elif "Converting special symbols" in step:
                            cleaning_status.info("🔄 " + step.replace("🔄 ", ""))
                        elif "Treating zeros as null" in step:
                            cleaning_status.info("0️⃣ " + step.replace("0️⃣  ", ""))
                        elif "Profiling data quality" in step:
                            cleaning_status.info("📊 " + step.replace("📊 ", ""))
                        elif "Removing columns with" in step:
                            cleaning_status.info("🗑️ " + step.replace("🗑️  ", ""))
                        elif "Removing rows with" in step:
                            cleaning_status.info("🗑️ " + step.replace("🗑️  ", ""))
                        elif "Removing duplicates" in step:
                            cleaning_status.info("🗑️ " + step.replace("🗑️  ", ""))
                        elif "Standardizing column names" in step:
                            cleaning_status.info("📝 " + step.replace("📝 ", ""))
                        elif "Cleaning text columns" in step:
                            cleaning_status.info("✂️ " + step.replace("✂️  ", ""))
                        elif "Converting date columns" in step:
                            cleaning_status.info("📅 " + step.replace("📅 ", ""))
                        elif "CLEANING COMPLETE" in step:
                            cleaning_status.success("✅ " + step.replace("✅ ", ""))
                        elif "REMOVING:" in step:
                            cleaning_status.warning("⚠️ " + step)
                        elif "✓" in step:
                            cleaning_status.info("✅ " + step)
                    
                    st.session_state.df = df
                    st.session_state.cleaning_report = cleaning_report
                    cleaning_expander.empty()
            else:
                df = st.session_state.df
                cleaning_report = st.session_state.get('cleaning_report', None)
            
            if cleaning_report:
                st.markdown('<div class="section-header">Data Cleaning Report</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows Removed", f"{cleaning_report.get('rows_removed', 0):,}")
                with col2:
                    st.metric("Removal Rate", f"{cleaning_report.get('rows_removed_pct', 0):.1f}%")
                with col3:
                    st.metric("Cols Removed", cleaning_report.get('cols_removed', 0))
                with col4:
                    final_quality = 100 - cleaning_report.get('rows_removed_pct', 0)
                    st.metric("Data Retained", f"{final_quality:.1f}%")
                
                removed_cols = cleaning_report.get('removed_columns', [])
                if removed_cols:
                    with st.expander("Removed Columns (>60% missing/zeros)"):
                        for col in removed_cols:
                            quality = cleaning_report['quality_profile'][col]
                            st.warning(f"**{col}**: {quality['null_pct']:.1f}% nulls + {quality['zero_pct']:.1f}% zeros")
            
            st.markdown('<div class="section-header">Dataset Overview (After Cleaning)</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory (MB)", f"{memory_mb:.2f}")
            with col4:
                missing_count = df.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
            
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Export Your Cleaned Data**")
            with col2:
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"cleaned_{file_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col3:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"cleaned_{file_name.replace('.csv', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with st.expander("Preview First 10 Rows", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # ========== OPTIMIZED DASHBOARD - FAST & LIGHTWEIGHT ==========
            st.markdown('<div class="section-header">📊 Data Quality & Analytics Dashboard</div>', unsafe_allow_html=True)
            
            # Add progress indicator for dashboard creation
            dashboard_expander = st.expander("📈 Dashboard Creation Progress", expanded=True)
            dashboard_status = dashboard_expander.empty()
            dashboard_progress = dashboard_expander.progress(0, text="Initializing dashboard creation...")
            
            try:
                # ========== KPI METRICS (4 Key Indicators) ==========
                dashboard_progress.progress(10, text="Calculating key performance indicators...")
                dashboard_status.info("📊 Calculating key performance indicators...")
                st.markdown("### 📊 Key Performance Indicators")
                kpis = create_kpi_dashboard(df)
                
                if kpis:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_records = int(str(kpis['Total Records']).replace(',', ''))
                        st.metric("Total Records", total_records)
                    with col2:
                        completeness = float(str(kpis['Data Completeness']).replace('%', '').strip())
                        st.metric("Data Completeness", f"{completeness:.1f}%")
                    with col3:
                        numeric_cols = int(str(kpis['Numeric Columns']).replace(',', ''))
                        st.metric("Numeric Columns", numeric_cols)
                    with col4:
                        duplicates = int(str(kpis['Duplicates']).replace(',', ''))
                        st.metric("Duplicates", duplicates)
                
                st.markdown("---")
                
                # ========== DATA QUALITY INSIGHTS (2 Charts) ==========
                dashboard_progress.progress(30, text="Analyzing data quality patterns...")
                dashboard_status.info("🔍 Analyzing data quality patterns...")
                st.markdown("### 🔍 Data Quality Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Data Quality Heatmap**")
                    dashboard_status.info("📊 Generating data quality heatmap...")
                    quality_fig = create_data_quality_heatmap(df, cleaning_report)
                    if quality_fig is not None:
                        st.plotly_chart(quality_fig, use_container_width=True)
                    else:
                        st.info("Quality analysis not available for this dataset")
                
                with col2:
                    st.markdown("**Missing Values Pattern**")
                    dashboard_status.info("📈 Analyzing missing values pattern...")
                    missing_fig = create_missing_value_pattern(df)
                    if missing_fig is not None:
                        st.plotly_chart(missing_fig, use_container_width=True)
                    else:
                        st.info("Missing values not found in this dataset")
                
                st.markdown("---")
                
                # ========== DATA ANALYSIS (4 Key Charts) ==========
                dashboard_progress.progress(50, text="Creating data distribution charts...")
                dashboard_status.info("🥧 Creating distribution charts...")
                st.markdown("### 📈 Data Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribution (Pie Chart)**")
                    dashboard_status.info("🥧 Generating categorical distribution chart...")
                    pie_fig = create_distribution_pie_chart(df)
                    if pie_fig is not None:
                        st.plotly_chart(pie_fig, use_container_width=True)
                    else:
                        st.info("Need categorical data for distribution")
                
                with col2:
                    st.markdown("**Trend Analysis (Line Chart)**")
                    dashboard_status.info("📈 Generating trend analysis chart...")
                    line_fig = create_trend_line_chart(df)
                    if line_fig is not None:
                        st.plotly_chart(line_fig, use_container_width=True)
                    else:
                        st.info("Need numeric data for trends")
                
                st.markdown("---")
                
                dashboard_progress.progress(70, text="Detecting outliers and patterns...")
                dashboard_status.info("📦 Detecting outliers and patterns...")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Outlier Detection (Box Plot)**")
                    dashboard_status.info("🔍 Generating outlier detection chart...")
                    box_fig = create_box_plot(df)
                    if box_fig is not None:
                        st.plotly_chart(box_fig, use_container_width=True)
                    else:
                        st.info("Need numeric data for outlier detection")
                
                with col2:
                    st.markdown("**Frequency Distribution (Histogram)**")
                    dashboard_status.info("📊 Generating frequency distribution chart...")
                    hist_fig = create_histogram_analysis(df)
                    if hist_fig is not None:
                        st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        st.info("Need numeric data for frequency distribution")
                
                dashboard_progress.progress(100, text="Dashboard generation complete!")
                dashboard_status.success("✅ Dashboard loaded successfully!")
                st.success("✅ Dashboard loaded successfully!")
                dashboard_expander.empty()
                
            except Exception as e:
                dashboard_expander.empty()
                st.error(f"Error creating dashboard: {e}")
                import traceback
                st.write(traceback.format_exc())
            
            # ========== VECTOR INDEX CREATION ==========
            st.markdown("---")
            st.markdown('<div class="section-header">Creating Vector Index</div>', unsafe_allow_html=True)
            
            # Create an expander to show the indexing process
            indexing_expander = st.expander("🔍 Index Creation Progress", expanded=True)
            indexing_status = indexing_expander.empty()
            
            with st.spinner("Creating vector index..."):
                try:
                    indexing_status.info("💾 Saving dataframe for indexing...")
                    save_dataframe(df, persist_dir=str(PERSIST_DIR))  # Convert Path to string
                    
                    indexing_status.info("🧮 Building vector index with data summaries...")
                    build_or_load_index(df, persist_dir=str(PERSIST_DIR))
                    
                    st.session_state.index_created = True
                    indexing_status.success("✅ Vector index created! Ready to ask questions.")
                    indexing_expander.empty()
                    
                    st.success("Vector index created! Ready to ask questions.")
                except Exception as e:
                    indexing_expander.empty()
                    st.error(f"Index creation failed: {e}")
            
            st.sidebar.markdown("---")
            st.sidebar.header("Dataset Info")
            st.sidebar.metric("Rows", f"{df.shape[0]:,}")
            st.sidebar.metric("Columns", df.shape[1])
            st.sidebar.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

if st.session_state.index_created and st.session_state.df is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">Ask Questions from Your Data</div>', unsafe_allow_html=True)
    
    # Add instructions for asking questions
    st.info("💡 Tip: Ask questions like 'What are the top 5 products by revenue?' or 'Show me the trend of sales over time'")
    
    question = st.text_input(
        "What would you like to know about your data?",
        placeholder="e.g., Top 5 products by revenue?"
    )
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col2:
        top_k = st.slider("Top-K Results", 1, 10, 4)
    with col3:
        run_btn = st.button("Ask Question", use_container_width=True)
    
    if run_btn:
        if not question.strip():
            st.error("Please type a question.")
        else:
            # Create an expander to show the analysis process
            analysis_expander = st.expander("🔍 Analysis Progress", expanded=True)
            analysis_status = analysis_expander.empty()
            
            with st.spinner("Analyzing data..."):
                try:
                    analysis_status.info("🔍 Searching for relevant information in your dataset...")
                    answer_text, contexts, chart_fig = answer_question(
                        question=question,
                        persist_dir=str(PERSIST_DIR),
                        top_k=int(top_k),
                    )

                    st.markdown("---")
                    st.subheader("Answer:")
                    st.info(answer_text)

                    if chart_fig is not None:
                        analysis_status.info("📊 Generating visualization based on your question...")
                        st.markdown("---")
                        st.subheader("Visualization:")
                        st.plotly_chart(chart_fig, use_container_width=True)
                    
                    analysis_status.success("✅ Analysis complete!")
                    
                    if contexts:
                        st.markdown("---")
                        with st.expander("Source Data", expanded=False):
                            for i, c in enumerate(contexts[:3], 1):
                                st.markdown(f"**Source {i}:**")
                                st.code(c[:300], language="text")

                except Exception as e:
                    analysis_expander.empty()
                    st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a dataset to get started!")
    
    # Add welcome message and instructions
    with st.expander("ℹ️ How to use this application", expanded=True):
        st.markdown("""
        ### Getting Started
        1. **Upload Data**: Use the file uploader in the sidebar to upload a CSV or Excel file
        2. **Sample Data**: Click "Load sample dataset" to try with example data
        3. **Automatic Analysis**: The system will automatically clean and analyze your data
        4. **Ask Questions**: Once processing is complete, ask natural language questions about your data
        
        ### What Happens During Processing
        - 🧹 **Data Cleaning**: Removes invalid data, standardizes formats, and handles missing values
        - 📊 **Analysis**: Generates visualizations and key metrics about your data
        - 🤖 **Indexing**: Creates a searchable index for answering your questions
        - 📈 **Dashboard**: Displays comprehensive analytics and insights
        
        ### Features
        - Natural language queries about your data
        - Automatic data quality assessment
        - Anomaly detection and outlier analysis
        - Interactive visualizations
        - Data export capabilities
        """)
