# ANALYST-FOCUSED VISUALIZATION FUNCTIONS
# Replace generic charts with actionable, meaningful insights

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def get_numeric_columns(df):
    """Get numeric columns for analysis"""
    return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_columns(df):
    """Get categorical columns for analysis"""
    return df.select_dtypes(include=['object']).columns.tolist()

# ============= ANALYST-GRADE VISUALIZATIONS =============

def create_key_metrics(df):
    """📊 KEY METRICS: Statistical summary that matters"""
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return None
    
    metrics = {}
    for col in numeric_cols[:5]:  # Top 5 numeric columns
        data = df[col].dropna()
        if len(data) > 0:
            metrics[col] = {
                'Mean': f"{data.mean():.2f}",
                'Median': f"{data.median():.2f}",
                'Std Dev': f"{data.std():.2f}",
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}",
                'Skewness': f"{stats.skew(data):.2f}"
            }
    
    return metrics

def create_outlier_detection(df):
    """🎯 OUTLIER DETECTION: Find anomalies in key metrics"""
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return None
    
    col = numeric_cols[0]
    data = df[col].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # Box plot with outliers highlighted
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name=col,
        marker_color='#667eea',
        boxmean='sd',
        hovertext=f'Q1: {Q1:.2f}<br>Q3: {Q3:.2f}<br>IQR: {IQR:.2f}',
        hoverinfo='text'
    ))
    
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            y=outliers[col],
            mode='markers',
            marker=dict(size=10, color='#f56565', symbol='diamond'),
            name=f'Outliers ({len(outliers)})',
            text=[f'Outlier: {v:.2f}' for v in outliers[col]],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f"🎯 Outlier Detection: {col} ({len(outliers)} anomalies)",
        yaxis_title=col,
        height=350,
        template='plotly_white',
        showlegend=True
    )
    return fig

def create_trend_analysis(df):
    """📈 TREND ANALYSIS: Growth/decline patterns over index"""
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols or len(df) < 10:
        return None
    
    col = numeric_cols[0]
    df_sorted = df[[col]].copy()
    df_sorted['index'] = range(len(df_sorted))
    
    # Calculate trend line
    z = np.polyfit(df_sorted['index'], df_sorted[col].dropna(), 2)
    p = np.poly1d(z)
    trend_line = p(df_sorted['index'])
    
    # Calculate growth rate
    first_val = df_sorted[col].iloc[0]
    last_val = df_sorted[col].iloc[-1]
    growth_rate = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['index'],
        y=df_sorted[col],
        mode='lines',
        name='Actual Data',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['index'],
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='#f56565', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f"📈 Trend Analysis: {col} (Growth: {growth_rate:+.1f}%)",
        xaxis_title='Time Index',
        yaxis_title=col,
        hovermode='x unified',
        height=350,
        template='plotly_white'
    )
    return fig

def create_distribution_insights(df):
    """📊 DISTRIBUTION ANALYSIS: Identify skewness & kurtosis"""
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return None
    
    col = numeric_cols[0]
    data = df[col].dropna()
    
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=int(30),  # Convert to Python int to avoid numpy.int64 issue
        name='Frequency',
        marker_color='#667eea',
        opacity=0.7
    ))
    
    # Add normal distribution overlay
    mu = data.mean()
    sigma = data.std()
    x_range = np.linspace(data.min(), data.max(), 100)
    y_normal = stats.norm.pdf(x_range, mu, sigma) * len(data) * (data.max() - data.min()) / 30
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_normal,
        mode='lines',
        name='Normal Dist',
        line=dict(color='#f56565', width=3)
    ))
    
    fig.update_layout(
        title=f"📊 Distribution: {col} (Skew: {skewness:.2f}, Kurt: {kurtosis:.2f})",
        xaxis_title=col,
        yaxis_title='Frequency',
        height=350,
        template='plotly_white',
        barmode='overlay'
    )
    return fig

def create_segmentation_analysis(df):
    """🎯 SEGMENTATION: Group by categorical feature & compare metrics"""
    cat_cols = get_categorical_columns(df)
    num_cols = get_numeric_columns(df)
    
    if not cat_cols or not num_cols:
        return None
    
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    
    # Group and calculate mean, median, count
    grouped = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'median', 'std'])
    grouped = grouped.sort_values('mean', ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped.index,
        y=grouped['mean'],
        name='Mean',
        marker_color='#667eea',
        error_y=dict(type='data', array=grouped['std'])
    ))
    
    fig.update_layout(
        title=f"🎯 Segment Analysis: {num_col} by {cat_col}",
        xaxis_title=cat_col,
        yaxis_title=f"Mean {num_col}",
        height=350,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_correlation_pairs(df):
    """🔗 CORRELATION PAIRS: Identify relationships worth investigating"""
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{p['var1'][:10]} ↔ {p['var2'][:10]}" for p in corr_pairs],
            y=[p['correlation'] for p in corr_pairs],
            marker=dict(
                color=[p['correlation'] for p in corr_pairs],
                colorscale='RdBu',
                showscale=False
            ),
            text=[f"{p['correlation']:.2f}" for p in corr_pairs],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="🔗 Top Correlations: Variables Worth Investigating",
        xaxis_title='Variable Pairs',
        yaxis_title='Correlation Coefficient',
        height=350,
        template='plotly_white',
        yaxis=dict(range=[-1, 1])
    )
    return fig

def create_top_performers(df):
    """🏆 TOP PERFORMERS: Key segments/categories driving results"""
    cat_cols = get_categorical_columns(df)
    num_cols = get_numeric_columns(df)
    
    if not cat_cols or not num_cols:
        return None
    
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    
    # Calculate total per category
    totals = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(15)
    
    fig = px.bar(
        x=totals.index,
        y=totals.values,
        title=f"🏆 Top Performers: {num_col} by {cat_col}",
        labels={'x': cat_col, 'y': f'Total {num_col}'},
        color=totals.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    return fig

def create_pareto_analysis(df):
    """📊 PARETO 80/20: Where do 80% of results come from?"""
    cat_cols = get_categorical_columns(df)
    num_cols = get_numeric_columns(df)
    
    if not cat_cols or not num_cols:
        return None
    
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    
    # Group and sort
    data = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
    cumsum = data.cumsum()
    cumsum_pct = (cumsum / cumsum.iloc[-1] * 100)
    
    # Find 80% threshold
    threshold_idx = (cumsum_pct >= 80).idxmax() if (cumsum_pct >= 80).any() else len(cumsum_pct)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=data.index[:20],
        y=data.values[:20],
        name='Value',
        marker_color='#667eea'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=cumsum_pct.index[:20],
        y=cumsum_pct.values[:20],
        mode='lines+markers',
        name='Cumulative %',
        line=dict(color='#f56565', width=3),
        marker=dict(size=8)
    ), secondary_y=True)
    
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#ecc94b",
        annotation_text="80% Target",
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"📊 Pareto 80/20: {threshold_idx} items = 80% of {num_col}",
        xaxis_title=cat_col,
        yaxis_title='Value',
        yaxis2_title='Cumulative %',
        height=350,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_data_quality_report(df):
    """📋 DATA QUALITY REPORT: Completeness, duplicates, anomalies"""
    report = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Duplicate Rows': df.duplicated().sum(),
        'Duplicate %': f"{(df.duplicated().sum() / len(df) * 100):.1f}%",
        'Missing Cells': df.isnull().sum().sum(),
        'Missing %': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%",
        'Complete Rows': len(df.dropna()),
        'Complete %': f"{(len(df.dropna()) / len(df) * 100):.1f}%",
    }
    
    # Columns with missing data
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        fig = px.bar(
            x=missing_by_col.index,
            y=missing_by_col.values,
            title="📋 Data Quality: Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'},
            color=missing_by_col.values,
            color_continuous_scale='Reds'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values detected!", x=0.5, y=0.5)
    
    fig.update_layout(height=350, template='plotly_white', showlegend=False)
    return report, fig

# ============= HELPER: Make subplots for Pareto
from plotly.subplots import make_subplots

# Export all functions
__all__ = [
    'create_key_metrics',
    'create_outlier_detection',
    'create_trend_analysis',
    'create_distribution_insights',
    'create_segmentation_analysis',
    'create_correlation_pairs',
    'create_top_performers',
    'create_pareto_analysis',
    'create_data_quality_report'
]
