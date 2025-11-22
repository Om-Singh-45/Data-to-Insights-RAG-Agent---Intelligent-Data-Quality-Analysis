from dotenv import load_dotenv
load_dotenv()

import os
import re
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser

try:
    from llama_index.embeddings.gemini import GeminiEmbedding
    GEMINI_EMBEDDING_AVAILABLE = True
except ImportError:
    GEMINI_EMBEDDING_AVAILABLE = False


# -------- Config --------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found! Add it in .env file.")


# -------- Data Cleaning & Quality Profiling --------
def profile_data_quality(df: pd.DataFrame) -> dict:
    """Analyze data quality issues before cleaning."""
    profile = {}
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        # Count zeros (only for numeric columns)
        zero_count = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            zero_count = (df[col] == 0).sum()
        
        profile[col] = {
            'dtype': str(df[col].dtype),
            'nulls': null_count,
            'null_pct': null_pct,
            'zeros': zero_count,
            'zero_pct': (zero_count / len(df)) * 100 if pd.api.types.is_numeric_dtype(df[col]) else 0,
            'unique': df[col].nunique(),
            'cardinality_pct': (df[col].nunique() / len(df)) * 100,
        }
    
    return profile

def clean_dataset(df: pd.DataFrame) -> tuple:
    """
    Intelligently clean dataset:
    - Treat special symbols (!@#$%^&* etc.) as null values
    - Treat 0 as null value
    - Remove columns with >50% null values
    - Remove ALL rows with ANY null values
    - Remove duplicates
    - Handle remaining missing values
    
    Returns: (cleaned_df, cleaning_report)
    """
    df = df.copy()
    cleaning_report = {}
    
    print("🧹 INTELLIGENT DATA CLEANING...")
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # ===== STEP 0: Replace special symbols with NaN =====
    print("\n🔄 Converting special symbols to null...")
    special_chars_pattern = r'[!@#$%^&*()_+=\[\]{};:\'"\\|,.<>?/`~]'
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace special symbols with NaN
            df[col] = df[col].replace(special_chars_pattern, pd.NA, regex=True)
            # Replace empty strings with NaN
            df[col] = df[col].replace('', pd.NA)
            # Replace whitespace-only strings with NaN
            df[col] = df[col].replace(r'^\s+$', pd.NA, regex=True)
    
    # ===== STEP 1: Treat 0 as null in numeric columns =====
    print("\n0️⃣  Treating zeros as null values...")
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df.loc[df[col] == 0, col] = pd.NA
    
    # ===== PROFILE DATA QUALITY AFTER SYMBOL/ZERO CONVERSION =====
    print("\n📊 Profiling data quality...")
    quality_profile = profile_data_quality(df)
    
    # ===== STEP 2: Remove columns with >50% null values =====
    print("\n🗑️  Removing columns with >50% null values...")
    cols_to_remove = []
    
    for col, profile in quality_profile.items():
        null_pct = profile['null_pct']
        if null_pct > 50:
            cols_to_remove.append(col)
            print(f"  ✗ REMOVING: {col} ({null_pct:.1f}% null)")
    
    df = df.drop(columns=cols_to_remove)
    cleaning_report['removed_columns'] = cols_to_remove
    print(f"  ✓ Removed {len(cols_to_remove)} columns")
    
    # ===== STEP 3: Remove rows with MORE THAN 2 null values =====
    print("\n🗑️  Removing rows with more than 2 null values...")
    null_count_per_row = df.isnull().sum(axis=1)
    null_rows_before = (null_count_per_row > 2).sum()
    df = df.loc[null_count_per_row <= 2]  # Keep rows with 0, 1, or 2 nulls; remove if >2
    print(f"  ✓ Removed {null_rows_before} rows with more than 2 null values")
    
    # ===== STEP 4: Remove duplicate rows =====
    print("\n🗑️  Removing duplicates...")
    removed_dups = len(df) - len(df.drop_duplicates())
    df = df.drop_duplicates()
    if removed_dups > 0:
        print(f"  ✓ Removed {removed_dups} duplicate rows")
    
    # ===== STEP 5: Standardize column names =====
    print("\n📝 Standardizing column names...")
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    print(f"  ✓ Standardized column names to lowercase/snake_case")
    
    # ===== STEP 6: Clean text columns =====
    print("\n✂️  Cleaning text columns...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    print(f"  ✓ Trimmed whitespace")
    
    # ===== STEP 7: Convert date columns =====
    print("\n📅 Converting date columns...")
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"  ✓ Converted '{col}' to datetime")
            except:
                pass
    
    # ===== GENERATE CLEANING REPORT =====
    rows_removed = initial_rows - len(df)
    cols_removed = initial_cols - len(df.columns)
    
    cleaning_report.update({
        'initial_rows': initial_rows,
        'final_rows': len(df),
        'rows_removed': rows_removed,
        'rows_removed_pct': (rows_removed / initial_rows) * 100 if initial_rows > 0 else 0,
        'initial_cols': initial_cols,
        'final_cols': len(df.columns),
        'cols_removed': cols_removed,
        'quality_profile': quality_profile,
    })
    
    print(f"\n✅ CLEANING COMPLETE:")
    print(f"   Rows: {initial_rows:,} → {len(df):,} ({rows_removed:,} removed, {(rows_removed/initial_rows)*100:.1f}%)")
    print(f"   Cols: {initial_cols} → {len(df.columns)} ({cols_removed} removed)")
    print(f"   Final dataset: {len(df):,} records × {len(df.columns)} columns")
    
    return df, cleaning_report

# -------- ANALYST-FOCUSED VISUALIZATIONS --------

def create_data_quality_heatmap(df: pd.DataFrame, cleaning_report: dict) -> go.Figure:
    """Heatmap showing data quality issues per column (nulls vs zeros vs valid)."""
    quality_profile = cleaning_report.get('quality_profile', {})
    
    columns = list(quality_profile.keys())
    nulls = [quality_profile[col]['null_pct'] for col in columns]
    zeros = [quality_profile[col]['zero_pct'] for col in columns]
    valid = [100 - quality_profile[col]['null_pct'] - quality_profile[col]['zero_pct'] for col in columns]
    
    fig = go.Figure(data=[
        go.Bar(y=columns, x=valid, name='Valid Data', orientation='h', marker=dict(color='#48bb78')),
        go.Bar(y=columns, x=zeros, name='Zeros', orientation='h', marker=dict(color='#f56565')),
        go.Bar(y=columns, x=nulls, name='Missing', orientation='h', marker=dict(color='#ecc94b')),
    ])
    fig.update_layout(
        title="📊 Data Quality by Column (% Distribution)",
        barmode='stack',
        xaxis_title="Percentage (%)",
        yaxis_title="Column Name",
        height=max(300, len(columns) * 20),
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_anomaly_detection_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Detect and visualize outliers using IQR method for numeric columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return None
    
    col = numeric_cols[0]  # Analyze first numeric column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    normal = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(normal))), y=normal[col].tolist(),
        mode='markers', name='Normal', 
        marker=dict(size=5, color='#667eea', opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(normal), len(normal) + len(outliers))), y=outliers[col].tolist(),
        mode='markers', name='Outliers',
        marker=dict(size=8, color='#f56565', symbol='diamond')
    ))
    
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="#ecc94b", name="Upper Bound")
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="#ecc94b", name="Lower Bound")
    
    fig.update_layout(
        title=f"🔍 Anomaly Detection: {col} ({len(outliers)} outliers detected)",
        xaxis_title="Row Index",
        yaxis_title=col,
        height=400,
        template='plotly_white',
        hovermode='closest'
    )
    return fig

def create_missing_value_pattern(df: pd.DataFrame) -> Optional[go.Figure]:
    """Visualize missing value patterns - which rows have the most missing data."""
    missing_per_row = df.isnull().sum(axis=1)
    
    if missing_per_row.max() == 0:
        return None
    
    fig = px.histogram(
        missing_per_row,
        nbins=int(min(30, missing_per_row.max() + 1)),  # Convert to Python int to avoid numpy.int64 issue
        title="📈 Missing Value Distribution Across Rows",
        labels={'value': 'Count', 'count': 'Number of Rows'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_xaxes(title_text="Number of Missing Values per Row")
    fig.update_yaxes(title_text="Number of Rows")
    fig.update_layout(height=350, template='plotly_white')
    
    return fig

def create_cardinality_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Show cardinality (uniqueness) of columns - identifies categorical vs high-variety columns."""
    cardinality_data = []
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality_pct = (unique_count / len(df)) * 100
        cardinality_data.append({
            'column': col,
            'unique': unique_count,
            'cardinality_pct': cardinality_pct,
            'type': 'High Variety' if cardinality_pct > 50 else 'Categorical'
        })
    
    cardinality_df = pd.DataFrame(cardinality_data)
    
    fig = px.bar(
        cardinality_df,
        x='column',
        y='cardinality_pct',
        color='type',
        title="🎯 Column Cardinality Analysis (Uniqueness)",
        labels={'cardinality_pct': 'Cardinality (%)', 'column': 'Column'},
        color_discrete_map={'Categorical': '#764ba2', 'High Variety': '#667eea'}
    )
    fig.update_layout(height=350, template='plotly_white', xaxis_tickangle=-45)
    
    return fig

def create_value_distribution_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Show distribution of numeric values (excluding zeros for clarity)."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return None
    
    col = numeric_cols[0]
    # Exclude zeros and negative values for better distribution view
    values = df[col][(df[col] > 0) & (df[col].notna())]
    
    if len(values) == 0:
        return None
    
    fig = px.histogram(
        values,
        nbins=40,
        title=f"📊 {str(col).title()} Distribution (Excluding Zeros)",
        labels={'value': col, 'count': 'Frequency'},
        color_discrete_sequence=['#48bb78']
    )
    fig.update_layout(height=350, template='plotly_white', hovermode='x unified')
    
    return fig

def create_zero_percentage_chart(df: pd.DataFrame, cleaning_report: dict) -> Optional[go.Figure]:
    """Show which columns have excessive zeros (potential data quality issues)."""
    quality_profile = cleaning_report.get('quality_profile', {})
    
    numeric_cols = [col for col in quality_profile if quality_profile[col]['zero_pct'] > 0]
    zero_data = [{
        'column': col,
        'zero_pct': quality_profile[col]['zero_pct'],
        'severity': 'Critical' if quality_profile[col]['zero_pct'] > 50 else 'Warning' if quality_profile[col]['zero_pct'] > 20 else 'Info'
    } for col in numeric_cols]
    
    if not zero_data:
        return None
    
    zero_df = pd.DataFrame(zero_data)
    
    fig = px.bar(
        zero_df,
        x='column',
        y='zero_pct',
        color='severity',
        title="⚠️ Zero Value Problem Analysis",
        labels={'zero_pct': 'Zeros (%)', 'column': 'Column'},
        color_discrete_map={'Critical': '#f56565', 'Warning': '#ecc94b', 'Info': '#667eea'}
    )
    fig.update_layout(height=350, template='plotly_white', xaxis_tickangle=-45)
    
    return fig

def create_cleaning_impact_report(cleaning_report: dict) -> Optional[go.Figure]:
    """Visual summary of cleaning impact."""
    initial = cleaning_report.get('initial_rows', 0)
    final = cleaning_report.get('final_rows', 0)
    removed = cleaning_report.get('rows_removed', 0)
    
    # Handle case where no data is available
    if initial == 0 and final == 0 and removed == 0:
        return None
    
    fig = go.Figure(data=[
        go.Bar(x=['Data Retained', 'Data Removed'], y=[final, removed],
               marker=dict(color=['#48bb78', '#f56565']),
               text=[f'{final:,}<br>({(final/initial)*100:.1f}%)', f'{removed:,}<br>({(removed/initial)*100:.1f}%)'],
               textposition='inside',
               textfont=dict(color='white', size=12),
               hovertemplate='<b>%{x}</b><br>Records: %{y:,}<extra></extra>')
    ])
    fig.update_layout(
        title=f"🧹 Data Cleaning Impact: {initial:,} → {final:,} Records",
        yaxis_title="Number of Records",
        height=350,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def get_embeddings():
    """Return an embeddings object for LlamaIndex.

    By default this tries to use local HuggingFace embeddings (free, offline).
    You can override by setting environment variable `EMBEDDING_PROVIDER` to
    'gemini' to use Google Gemini embeddings (requires GEMINI_API_KEY).
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "hf").lower()

    if provider == "gemini":
        if not GEMINI_EMBEDDING_AVAILABLE:
            raise RuntimeError(
                "Google Generative AI / Gemini embedding support is not available.\n"
                "Install google-generativeai: pip install google-generativeai llama-index-embeddings-gemini\n"
                "Then set GEMINI_API_KEY in your .env file."
            )

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY not found in environment.\n"
                "Please set GEMINI_API_KEY in your .env file.\n"
                "Get your key from: https://makersuite.google.com/app/apikey"
            )

        # Check if GeminiEmbedding is available
        if GEMINI_EMBEDDING_AVAILABLE:
            from llama_index.embeddings.gemini import GeminiEmbedding
            return GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")
        else:
            raise RuntimeError("Gemini embeddings are not available. Please install the required packages.")

    if provider in ("hf", "huggingface", "local"):
        try:
            return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(
                "Failed to load HuggingFace embeddings.\n"
                "Install sentence-transformers and torch: pip install sentence-transformers torch\n"
                f"Original error: {e}"
            ) from e

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}. Use 'hf' (default) or 'gemini'.")


# -------- DATA ANALYSIS DASHBOARD - TREND ANALYSIS & KPIs --------

def get_kpi_metrics(df: pd.DataFrame) -> dict:
    """Calculate key performance indicators from dataset."""
    metrics = {}
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Basic metrics
    metrics['total_records'] = len(df)
    metrics['total_columns'] = len(df.columns)
    metrics['numeric_columns'] = len(numeric_cols)
    metrics['categorical_columns'] = len(cat_cols)
    
    # Data quality metrics
    metrics['completeness'] = ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])) * 100
    metrics['duplicates'] = df.duplicated().sum()
    metrics['duplicate_rate'] = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    
    # Numeric statistics
    if len(numeric_cols) > 0:
        # Convert to numpy values to avoid attribute access issues
        numeric_data = df[numeric_cols]
        # Fix attribute access issues by converting to list
        if len(numeric_data) > 0:
            metrics['avg_value'] = float(numeric_data.mean().mean())
            metrics['max_value'] = float(numeric_data.max().max())
            metrics['min_value'] = float(numeric_data.min().min())
            metrics['std_dev'] = float(numeric_data.std().mean())
        else:
            metrics['avg_value'] = metrics['max_value'] = metrics['min_value'] = metrics['std_dev'] = 0.0
    else:
        metrics['avg_value'] = metrics['max_value'] = metrics['min_value'] = metrics['std_dev'] = 0
    
    return metrics


def create_kpi_dashboard(df: pd.DataFrame) -> dict:
    """Create KPI cards for dashboard display."""
    metrics = get_kpi_metrics(df)
    
    kpis = {
        'Total Records': f"{metrics['total_records']:,}",
        'Data Completeness': f"{metrics['completeness']:.1f}%",
        'Duplicates': f"{metrics['duplicates']:,}",
        'Numeric Columns': f"{metrics['numeric_columns']}",
        'Avg Value': f"{metrics['avg_value']:.2f}",
        'Max Value': f"{metrics['max_value']:.2f}",
    }
    
    return kpis


# -------- Generate Smart Data Summaries --------
def generate_data_summaries(df: pd.DataFrame) -> List[str]:
    """Generate pre-aggregated insights for better RAG responses."""
    summaries = []
    
    # Basic dataset info
    summaries.append(f"Dataset: {len(df)} records, {len(df.columns)} columns: {', '.join(df.columns)}")
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summaries.append(
            f"[STAT] {col}: min={df[col].min()}, max={df[col].max()}, "
            f"mean={df[col].mean():.2f}, median={df[col].median():.2f}, sum={df[col].sum():.0f}"
        )
    
    # Categorical distributions
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        top_values = df[col].value_counts().head(10)
        summary = f"[CATEGORY] {col}: " + ", ".join(f"{k}({v})" for k, v in top_values.items())
        summaries.append(summary)
    
    # Pre-compute common aggregations for better answers
    # Top items by numeric columns
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            try:
                top_agg = df.groupby(cat_col)[num_col].sum().nlargest(n=10)
                summary = f"[TOP {num_col}] Top {cat_col}: " + ", ".join(
                    f"{k}({v:.0f})" for k, v in top_agg.items()
                )
                summaries.append(summary)
            except:
                pass
    
    # If there's an ID column with quantities, add top IDs
    id_cols = [c for c in df.columns if 'id' in c.lower() or 'order' in c.lower()]
    qty_cols = [c for c in df.columns if 'quantity' in c.lower() or 'qty' in c.lower()]
    
    if id_cols and qty_cols:
        id_col = id_cols[0]
        qty_col = qty_cols[0]
        try:
            top_by_qty = df.nlargest(n=20, columns=qty_col)[[id_col, qty_col]].to_dict()
            summary = f"[TOP {qty_col}] Top Order IDs by {qty_col}: " + ", ".join(
                f"{row[id_col]}({row[qty_col]})" for row in top_by_qty
            )
            summaries.append(summary)
        except:
            pass
    
    return summaries


# -------- Convert DF to Vector-Readable Docs --------
def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Convert dataframe into documents for RAG using summaries + actual rows."""
    docs = []
    
    # Add pre-computed summaries (these help RAG understand the data)
    summaries = generate_data_summaries(df)
    for summary in summaries:
        docs.append(Document(text=summary))
    
    # Add sample rows with all their data (important for specific queries)
    # For larger datasets, sample strategically
    if len(df) > 200:
        # Sample: top 100 + random 100
        numeric_columns = df.select_dtypes(include=['number']).columns
        sort_column = numeric_columns[0] if len(numeric_columns) > 0 else df.columns[0]
        top_100 = df.nlargest(n=100, columns=sort_column)
        random_100 = df.sample(min(100, len(df)))
        sample_df = pd.concat([top_100, random_100]).drop_duplicates()
    else:
        sample_df = df
    
    # Convert rows to readable text for RAG
    for idx, row in sample_df.iterrows():
        row_dict = row.to_dict()
        # Create a readable format
        text = " | ".join(f"{k}: {v}" for k, v in row_dict.items())
        docs.append(Document(text=text))
    
    return docs


# -------- Build or Load Vector Store --------
def build_or_load_index(df: pd.DataFrame, persist_dir="chroma_db"):
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embed_model = get_embeddings()

    if (persist_dir / "docstore.json").exists():
        print("🔄 Loading existing index...")
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage, embed_model=embed_model)

    print("📌 Creating new index with data summaries...")
    docs = df_to_documents(df)
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print("💾 Index Stored Successfully")

    return index


# -------- Persistent DF for Charts --------
def save_dataframe(df: pd.DataFrame, persist_dir="chroma_db"):
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p / "persist_df.pkl")


def load_dataframe(persist_dir="chroma_db") -> Optional[pd.DataFrame]:
    file = Path(persist_dir) / "persist_df.pkl"
    if file.exists():
        result = pd.read_pickle(file)
        if isinstance(result, pd.DataFrame):
            return result
        return None
    return None


# -------- Answer Using RAG --------
def answer_question(question: str, persist_dir="chroma_db", top_k: int = 4):
    df = load_dataframe(persist_dir)
    if df is None:
        return "⚠ No data indexed. Upload & index a dataset first!", [], None

    index = build_or_load_index(df, persist_dir)

    llm = OpenRouter(
        model="anthropic/claude-3.5-sonnet",
        api_key=OPENROUTER_API_KEY
    )

    try:
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)
    except TypeError:
        query_engine = index.as_query_engine(llm=llm)

    # Enhance the question with context
    enhanced_question = (
        f"{question}\n\n"
        "Please provide a specific, factual answer with concrete numbers and examples from the data. "
        "Do not apologize or say you cannot find information - extract and analyze from the context."
    )

    try:
        response = query_engine.query(enhanced_question, similarity_top_k=top_k)
    except TypeError:
        response = query_engine.query(enhanced_question)

    # Extract source snippets
    contexts = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for n in response.source_nodes:
            try:
                node_obj = getattr(n, "node", n)
                node_text = getattr(node_obj, "get_text", lambda: str(node_obj))()
                snippet = str(node_text)[:250]
                if snippet not in contexts:
                    contexts.append(snippet)
            except Exception:
                pass
    
    if not contexts:
        contexts = ["Data retrieved from dataset aggregations"]

    # Generate appropriate chart
    chart = generate_smart_chart(df, question, str(response), top_k)

    return str(response), contexts, chart


# -------- Intelligent Chart Generation --------
def generate_smart_chart(df: pd.DataFrame, question: str, answer_text: str, top_k: int = 5):
    """Generate appropriate chart based on question, answer, and actual data."""
    
    # First, ask the LLM which visualization would be most suitable
    llm = OpenRouter(
        model="anthropic/claude-3.5-sonnet",
        api_key=OPENROUTER_API_KEY
    )
    
    # Create a prompt to ask which visualization is most suitable
    viz_prompt = f"""
    Based on this question and answer about a dataset, what type of visualization would be most suitable to display?
    
    Question: {question}
    Answer: {answer_text}
    
    Available chart types:
    1. Bar chart (for comparisons, rankings, categories)
    2. Line chart (for trends over time/sequence)
    3. Pie chart (for proportions, distributions)
    4. Scatter plot (for correlations, relationships)
    5. Histogram (for frequency distributions)
    6. Box plot (for statistical distributions, outliers)
    7. Heatmap (for correlation matrices, cross-tabulations)
    
    Respond with ONLY ONE of these words: bar, line, pie, scatter, histogram, box, heatmap
    """
    
    try:
        # Get the visualization recommendation from the LLM
        viz_response = llm.complete(viz_prompt)
        recommended_viz = str(viz_response).strip().lower()
        
        # Validate the recommendation
        valid_viz_types = ['bar', 'line', 'pie', 'scatter', 'histogram', 'box', 'heatmap']
        if recommended_viz not in valid_viz_types:
            recommended_viz = 'bar'  # Default to bar chart if invalid
            
        # Generate the appropriate chart based on the recommendation
        if recommended_viz == 'bar':
            return create_top_values_bar(df)
        elif recommended_viz == 'line':
            return create_trend_line_chart(df)
        elif recommended_viz == 'pie':
            return create_distribution_pie_chart(df)
        elif recommended_viz == 'scatter':
            return create_scatter_analysis(df)
        elif recommended_viz == 'histogram':
            return create_histogram_analysis(df)
        elif recommended_viz == 'box':
            return create_box_plot(df)
        elif recommended_viz == 'heatmap':
            return create_heatmap_correlation(df)
        else:
            # Fallback to bar chart
            return create_top_values_bar(df)
            
    except Exception as e:
        # If there's any error in the LLM call, fallback to a default visualization
        print(f"Error getting visualization recommendation: {e}")
        return create_top_values_bar(df)

def create_top_values_bar(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create bar chart for top values analysis."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) == 0 or len(cat_cols) == 0:
        return None
    
    num_col = numeric_cols[0]
    cat_col = cat_cols[0]
    
    top_data = df.groupby(cat_col)[num_col].sum().nlargest(n=10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_data.index,
            y=top_data.values,
            marker=dict(
                color=top_data.values,
                colorscale='Blues',
                showscale=True
            ),
            text=top_data.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"🏆 Top 10: {num_col} by {cat_col}",
        xaxis_title=cat_col,
        yaxis_title=num_col,
        height=400,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig

def create_trend_line_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create line chart for trend analysis."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        return None
    
    # Use first numeric column and show cumulative trend
    col = numeric_cols[0]
    data = df[col].dropna().head(100)  # First 100 records
    
    if len(data) < 2:
        return None
    
    fig = go.Figure()
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data.values,
        mode='lines+markers',
        name=f'{col} Trend',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    # Moving average
    ma = pd.Series(data.values).rolling(window=5, center=True).mean()
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=ma.tolist(),
        mode='lines',
        name='5-Point MA',
        line=dict(color='#f56565', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"📈 Trend Analysis: {col}",
        xaxis_title='Record Index',
        yaxis_title='Value',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_distribution_pie_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create pie chart for categorical distribution analysis."""
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(cat_cols) == 0:
        return None
    
    col = cat_cols[0]
    value_counts = df[col].value_counts().head(10)
    
    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index,
        values=value_counts.values,
        marker=dict(colors=px.colors.qualitative.Set3),
        textposition='inside',
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title=f"📊 Distribution Analysis: {col}",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_scatter_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create scatter plot for relationship analysis."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=df[x_col].dropna(),
            y=df[y_col].dropna(),
            mode='markers',
            marker=dict(
                size=6,
                color=df[x_col].dropna(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Value')
            ),
            name=f'{x_col} vs {y_col}'
        )
    ])
    
    fig.update_layout(
        title=f"🔗 Relationship: {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_histogram_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create histogram for frequency distribution analysis."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        return None
    
    col = numeric_cols[0]
    data = df[col].dropna().astype(float)
    
    # Calculate appropriate number of bins
    nbins = int(min(30, max(5, len(data) // 20)))
    
    fig = go.Figure(data=[
        go.Histogram(
            x=data,
            nbinsx=nbins,
            name=col,
            marker_color='#667eea',
            opacity=0.75
        )
    ])
    
    fig.update_layout(
        title=f"📊 Frequency Distribution: {col}",
        xaxis_title=col,
        yaxis_title='Frequency',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_box_plot(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create box plot for distribution and outlier analysis."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        return None
    
    # Use top 4 numeric columns
    cols_to_plot = numeric_cols[:4]
    
    fig = go.Figure()
    
    for col in cols_to_plot:
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="📦 Statistical Distribution & Outliers",
        yaxis_title='Value',
        height=400,
        template='plotly_white',
        hovermode='y'
    )
    
    return fig

def create_heatmap_correlation(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create correlation heatmap for numeric columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title="🔥 Correlation Matrix: Numeric Columns",
        height=400,
        xaxis_title='Columns',
        yaxis_title='Columns',
        template='plotly_white'
    )
    
    return fig
