#!/usr/bin/env python3
"""
CREATE Library Analysis App

A comprehensive Streamlit application for analyzing CREATE Library  data with:
- Circular genome plots using Circos-style visualization
- Mutation diversity analysis
- Sequence composition analysis (GC content, etc.)
- Interactive filtering and controls
- Modular architecture for easy expansion

Author: CREATE Design System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import io
import tempfile
import os
from collections import Counter
import re
import time

# ================================================================================
# CONFIGURATION AND SETUP
# ================================================================================

# Manus brand colors
MANUS_COLORS = {
    'primary_teal': '#4ECDC4',
    'dark_teal': '#2E8B8B',
    'light_teal': '#7FDBDA',
    'accent_green': '#96CEB4',
    'dark_bg': '#2C3E50',
    'light_bg': '#F8F9FA',
    'text_primary': '#2C3E50',
    'text_secondary': '#5A6C7D',
    'white': '#FFFFFF',
    'gradient_start': '#4ECDC4',
    'gradient_end': '#96CEB4'
}

# Create custom color palettes for plots
MANUS_PLOT_COLORS = [
    MANUS_COLORS['primary_teal'],
    MANUS_COLORS['dark_teal'],
    MANUS_COLORS['accent_green'],
    MANUS_COLORS['light_teal'],
    MANUS_COLORS['dark_bg'],
    MANUS_COLORS['text_secondary']
]

st.set_page_config(
    page_title="CREATE Library Design Analysis | Manus Bio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS with Manus branding
st.markdown(f"""
<style>
    /* Main app styling */
    .main > div {{
        padding-top: 1rem;
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {MANUS_COLORS['gradient_start']} 0%, {MANUS_COLORS['gradient_end']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {MANUS_COLORS['light_bg']};
    }}
    
    /* Form controls */
    .stSelectbox label, .stMultiSelect label {{
        font-weight: 600;
        color: {MANUS_COLORS['text_primary']};
        font-size: 14px;
    }}
    
    .stSelectbox > div > div {{
        border: 2px solid {MANUS_COLORS['light_teal']};
        border-radius: 8px;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {MANUS_COLORS['primary_teal']} 0%, {MANUS_COLORS['dark_teal']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {MANUS_COLORS['dark_teal']} 0%, {MANUS_COLORS['primary_teal']} 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
    }}
    
    /* Metrics */
    .metric-container {{
        background: linear-gradient(135deg, {MANUS_COLORS['white']} 0%, {MANUS_COLORS['light_bg']} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {MANUS_COLORS['primary_teal']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* Plot containers */
    .plot-container {{
        background: white;
        border: 1px solid {MANUS_COLORS['light_teal']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(78, 205, 196, 0.1);
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: linear-gradient(90deg, {MANUS_COLORS['light_teal']} 0%, {MANUS_COLORS['accent_green']} 100%);
        border-radius: 8px;
        color: {MANUS_COLORS['text_primary']};
        font-weight: 600;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {MANUS_COLORS['text_primary']};
        font-weight: 700;
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: rgba(78, 205, 196, 0.1);
        border-left: 4px solid {MANUS_COLORS['primary_teal']};
    }}
    
    /* Success boxes */
    .stSuccess {{
        background-color: rgba(150, 206, 180, 0.1);
        border-left: 4px solid {MANUS_COLORS['accent_green']};
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        border: 2px dashed {MANUS_COLORS['primary_teal']};
        border-radius: 10px;
        background: rgba(78, 205, 196, 0.05);
    }}
    
    /* Sidebar headers */
    .css-1lcbmhc {{
        color: {MANUS_COLORS['text_primary']};
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)

# ================================================================================
# PERFORMANCE OPTIMIZATION UTILITIES
# ================================================================================

@st.cache_data
def load_and_sample_data(file_data, file_name, sample_size=15000):
    """Load data with optional sampling for large datasets"""
    file_extension = file_name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        # First, get the total number of rows
        df_temp = pd.read_csv(io.BytesIO(file_data), nrows=0)
        total_rows = sum(1 for _ in pd.read_csv(io.BytesIO(file_data), chunksize=1000))
        
        if total_rows > sample_size:
            # Sample the data
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                              total_rows - sample_size, 
                                              replace=False))
            df = pd.read_csv(io.BytesIO(file_data), skiprows=skip_rows)
            return df, True, total_rows
        else:
            df = pd.read_csv(io.BytesIO(file_data))
            return df, False, total_rows
            
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(io.BytesIO(file_data))
        total_rows = len(df)
        
        if total_rows > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            return df, True, total_rows
        else:
            return df, False, total_rows
    
    raise ValueError(f"Unsupported file format: {file_extension}")

@st.cache_data
def preprocess_data_columns(df):
    """Pre-analyze data columns for efficient column detection"""
    column_info = {
        'position_columns': [],
        'sequence_columns': [],
        'distance_columns': [],
        'categorical_columns': [],
        'numeric_columns': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        dtype = df[col].dtype
        
        # Position columns
        if any(term in col_lower for term in ['start', 'position', 'pos', 'coord', 'genomic']):
            column_info['position_columns'].append(col)
        elif dtype in ['int64', 'float64'] and df[col].max() > 1000:
            column_info['position_columns'].append(col)
        
        # Sequence columns
        if any(term in col_lower for term in ['seq', 'spacer', 'cassette']):
            column_info['sequence_columns'].append(col)
        
        # Distance columns
        if col_lower == 'distance':
            column_info['distance_columns'].insert(0, col)  # Primary distance column first
        elif any(term in col_lower for term in ['dist', 'spacing', 'gap', 'length', 'size', 
                                               'pam_distance', 'target_distance', 'genomic_distance',
                                               'upstream', 'downstream', 'offset', 'interval']):
            column_info['distance_columns'].append(col)
        elif dtype in ['int64', 'float64']:
            numeric_vals = df[col].dropna()
            if len(numeric_vals) > 0:
                min_val, max_val = numeric_vals.min(), numeric_vals.max()
                if 1 <= min_val and max_val <= 10000000:
                    column_info['distance_columns'].append(col)
        
        # Categorical columns
        if dtype == 'object' and df[col].nunique() <= 20:
            column_info['categorical_columns'].append(col)
        
        # Numeric columns
        if dtype in ['int64', 'float64']:
            column_info['numeric_columns'].append(col)
    
    return column_info

@st.cache_data
def efficient_filter_data(df, filters_dict):
    """Efficiently filter data with minimal copying"""
    if not filters_dict:
        return df
    
    mask = pd.Series(True, index=df.index)
    
    for col, values in filters_dict.items():
        if col in df.columns and values:
            mask &= df[col].isin(values)
    
    return df[mask]

# Add performance warnings
def show_performance_info(df, is_sampled, original_size=None):
    """Show performance information to users"""
    if is_sampled:
        st.warning(f"⚡ **Performance Mode**: Displaying a sample of {len(df):,} rows from {original_size:,} total rows for faster performance.")
        st.info("💡 **Tip**: For full analysis, consider filtering your data externally or using a smaller dataset.")
    elif len(df) > 5000:
        st.info(f"📊 **Large Dataset**: Processing {len(df):,} rows. Some visualizations may take a moment to load.")

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

@st.cache_data
def calculate_gc_content(sequence):
    """Calculate GC content of a DNA sequence"""
    if pd.isna(sequence) or not sequence:
        return np.nan
    # gc_fraction returns 0-1, multiply by 100 to get percentage
    return gc_fraction(str(sequence).upper()) * 100

@st.cache_data
def parse_genbank_file(genbank_file):
    """Parse GenBank file and extract basic genome information"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gb') as tmp_file:
            tmp_file.write(genbank_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Parse the GenBank file
        record = SeqIO.read(tmp_file_path, "genbank")
        
        # Extract features
        features = []
        for feature in record.features:
            if feature.type in ['CDS', 'gene']:
                try:
                    gene_name = feature.qualifiers.get('gene', ['Unknown'])[0]
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    strand = feature.strand
                    features.append({
                        'gene': gene_name,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'length': end - start
                    })
                except:
                    continue
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            'genome_length': len(record.seq),
            'description': record.description,
            'features': pd.DataFrame(features)
        }
    except Exception as e:
        st.error(f"Error parsing GenBank file: {str(e)}")
        return None

@st.cache_data
def identify_unique_mutations(df, mutation_columns):
    """Identify unique mutations based on selected columns"""
    if not mutation_columns:
        return pd.DataFrame()
    
    # Create mutation signature by combining selected columns
    mutation_df = df[mutation_columns].copy()
    mutation_df['mutation_id'] = mutation_df.apply(
        lambda row: '_'.join([str(val) for val in row.values if pd.notna(val)]), 
        axis=1
    )
    
    # Count occurrences
    mutation_counts = mutation_df['mutation_id'].value_counts().reset_index()
    mutation_counts.columns = ['mutation_signature', 'count']
    
    # Add detailed information for top mutations
    detailed_mutations = []
    for _, row in mutation_counts.head(20).iterrows():
        mutation_id = row['mutation_signature']
        examples = df[mutation_df['mutation_id'] == mutation_id][mutation_columns].iloc[0]
        detailed_mutations.append({
            'mutation_signature': mutation_id,
            'count': row['count'],
            **examples.to_dict()
        })
    
    return pd.DataFrame(detailed_mutations)

# ================================================================================
# VISUALIZATION MODULES
# ================================================================================

class CircularGenomePlotter:
    """Module for creating circular genome plots"""
    
    @staticmethod
    def create_circular_plot(df, genome_info=None, position_col='genomic_start', 
                           color_col='name', filter_dict=None, max_gene_labels=12):
        """Create a professional Circos-style circular genome plot"""
        
        # Apply filters if provided
        plot_df = df.copy()
        if filter_dict:
            for col, values in filter_dict.items():
                if col in plot_df.columns and values:
                    plot_df = plot_df[plot_df[col].isin(values)]
        
        if plot_df.empty:
            st.warning("No data to plot after filtering")
            return None
        
        # Convert genomic positions to angles (0-360 degrees)
        if genome_info and 'genome_length' in genome_info:
            max_position = genome_info['genome_length']
        else:
            max_position = plot_df[position_col].max()
        
        # Ensure we have valid positions
        plot_df = plot_df[plot_df[position_col].notna()].copy()
        plot_df['angle'] = (plot_df[position_col] / max_position) * 360
        
        # Create the figure with dark background like professional Circos
        fig = go.Figure()
        
        # Define professional Circos-style colors
        circos_colors = {
            'outer_ring': '#2C3E50',
            'inner_ring': '#34495E',
            'tick_major': '#BDC3C7',
            'tick_minor': '#95A5A6',
            'background': '#1A252F',
            'text': '#ECF0F1',
            'grid': '#34495E'
        }
        
        # Create concentric circles for the genome structure
        theta = np.linspace(0, 2*np.pi, 1000)
        
        # Create solid colored band between inner and outer circles
        band_outer_radius = 1.0
        band_inner_radius = 0.85
        
        # Outer boundary of the band
        outer_x = band_outer_radius * np.cos(theta)
        outer_y = band_outer_radius * np.sin(theta)
        
        # Inner boundary of the band
        inner_x = band_inner_radius * np.cos(theta)
        inner_y = band_inner_radius * np.sin(theta)
        
        # Create the solid band using a filled area
        fig.add_trace(go.Scatter(
            x=np.concatenate([outer_x, inner_x[::-1], [outer_x[0]]]),
            y=np.concatenate([outer_y, inner_y[::-1], [outer_y[0]]]),
            fill='toself',
            fillcolor='rgba(44, 62, 80, 0.8)',  # Solid dark band
            line=dict(color='rgba(44, 62, 80, 1.0)', width=2),
            mode='lines',
            name='Genome Band',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add outer ring border
        fig.add_trace(go.Scatter(
            x=outer_x, y=outer_y,
            mode='lines',
            line=dict(color='#2C3E50', width=3),
            name='Outer Border',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add inner ring border
        fig.add_trace(go.Scatter(
            x=inner_x, y=inner_y,
            mode='lines',
            line=dict(color='#2C3E50', width=3),
            name='Inner Border',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add major tick marks every 500,000 bp
        label_interval = 500000
        num_labels = int(max_position / label_interval) + 1
        
        for i in range(num_labels):
            position = i * label_interval
            if position > max_position:
                break
                
            angle = (position / max_position) * 360
            rad = np.radians(angle)
            
            # Major tick marks
            tick_start = 1.0
            tick_end = 1.1
            tick_x = [tick_start * np.cos(rad), tick_end * np.cos(rad)]
            tick_y = [tick_start * np.sin(rad), tick_end * np.sin(rad)]
            
            fig.add_trace(go.Scatter(
                x=tick_x, y=tick_y,
                mode='lines',
                line=dict(color=circos_colors['tick_major'], width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Position labels
            label_radius = 1.2
            label_x = label_radius * np.cos(rad)
            label_y = label_radius * np.sin(rad)
            
            # Format position label
            if position >= 1000000:
                label_text = f"{position/1000000:.1f}"
            elif position >= 1000:
                label_text = f"{position/1000:.0f}k"
            else:
                label_text = f"{position}"
            
            fig.add_annotation(
                x=label_x, y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(size=12, color='#1A1A1A', family='Arial Black', weight='bold'),
                xanchor='center',
                yanchor='middle'
            )
        
        # Add minor tick marks every 100,000 bp
        minor_interval = 100000
        num_minor = int(max_position / minor_interval) + 1
        
        for i in range(num_minor):
            position = i * minor_interval
            if position > max_position or position % label_interval == 0:
                continue
                
            angle = (position / max_position) * 360
            rad = np.radians(angle)
            
            # Minor tick marks
            tick_start = 1.0
            tick_end = 1.05
            tick_x = [tick_start * np.cos(rad), tick_end * np.cos(rad)]
            tick_y = [tick_start * np.sin(rad), tick_end * np.sin(rad)]
            
            fig.add_trace(go.Scatter(
                x=tick_x, y=tick_y,
                mode='lines',
                line=dict(color=circos_colors['tick_minor'], width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add data points with Manus color scheme
        if color_col in plot_df.columns:
            unique_values = sorted(plot_df[color_col].unique())
            
            # Use Manus-inspired color palette matching their website
            manus_palette = [
                '#4ECDC4',  # Primary teal (main Manus color)
                '#96CEB4',  # Accent green
                '#2E8B8B',  # Dark teal
                '#7FDBDA',  # Light teal
                '#5FB3B3',  # Medium teal
                '#3ABAB4',  # Vibrant teal
                '#52C4B0',  # Teal-green
                '#6BC4BC',  # Soft teal
                '#45A29E',  # Deep teal
                '#5CDB95',  # Fresh green
                '#8EE4AF',  # Light green
                '#70C1B3',  # Mint teal
                '#247BA0',  # Ocean blue
                '#4ECDC4',  # Repeat primary for consistency
                '#96CEB4'   # Repeat accent for consistency
            ]
            
            for i, value in enumerate(unique_values):
                mask = plot_df[color_col] == value
                subset = plot_df[mask]
                
                if len(subset) == 0:
                    continue
                
                # Position data points on the solid band
                data_radius = 0.925  # Middle of the band
                subset_x = data_radius * np.cos(np.radians(subset['angle']))
                subset_y = data_radius * np.sin(np.radians(subset['angle']))
                
                # Create enhanced hover text
                hover_text = []
                for _, row in subset.iterrows():
                    text_parts = [
                        f"<b style='color: {manus_palette[i % len(manus_palette)]}'>{row.get('name', 'Unknown')}</b>",
                        f"<b>Position:</b> {row[position_col]:,} bp",
                        f"<b>{color_col}:</b> {row[color_col]}",
                    ]
                    
                    # Add library information if available
                    if 'libname' in row and pd.notna(row['libname']):
                        text_parts.append(f"<b>Library:</b> {row['libname']}")
                    
                    # Add additional info if available
                    if 'strand' in row and pd.notna(row['strand']):
                        text_parts.append(f"<b>Strand:</b> {row['strand']}")
                    if 'mutation_pattern' in row and pd.notna(row['mutation_pattern']):
                        text_parts.append(f"<b>Mutation:</b> {row['mutation_pattern']}")
                    
                    hover_text.append("<br>".join(text_parts))
                
                fig.add_trace(go.Scatter(
                    x=subset_x, y=subset_y,
                    mode='markers',
                    marker=dict(
                        size=14,  # Larger dots
                        color=manus_palette[i % len(manus_palette)],
                        line=dict(width=2, color='white'),
                        opacity=1.0,
                        symbol='circle'
                    ),
                    name=str(value),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=str(value)
                ))
        
        # Add gene names with intelligent positioning and overlap avoidance
        if 'name' in plot_df.columns:
            # Get all individual data points with their exact positions
            gene_positions = []
            for _, row in plot_df.iterrows():
                gene = row['name']
                exact_position = row[position_col]
                exact_angle = (exact_position / max_position) * 360
                
                gene_positions.append({
                    'gene': gene,
                    'angle': exact_angle,
                    'position': exact_position,
                    'row_data': row
                })
            
            # Group by gene and select one representative position per gene (preferably the first occurrence)
            gene_representatives = {}
            for gene_pos in gene_positions:
                gene = gene_pos['gene']
                if gene not in gene_representatives:
                    gene_representatives[gene] = gene_pos
            
            # Convert back to list and sort by count of variants per gene, then by angle
            gene_counts = plot_df['name'].value_counts().to_dict()
            selected_positions = []
            for gene, gene_pos in gene_representatives.items():
                gene_pos['count'] = gene_counts[gene]
                selected_positions.append(gene_pos)
            
            # Sort by count (most variants first) then by angle
            selected_positions.sort(key=lambda x: (-x['count'], x['angle']))
            
            # Function to check if two angles are too close (within min_separation degrees)
            def angles_too_close(angle1, angle2, min_separation=25):
                diff = abs(angle1 - angle2)
                return min(diff, 360 - diff) < min_separation
            
            # Select genes with overlap avoidance
            selected_genes = []
            for gene_info in selected_positions:
                angle = gene_info['angle']
                # Check if this gene's angle conflicts with already selected genes
                conflict = any(angles_too_close(angle, selected['angle']) for selected in selected_genes)
                
                if not conflict:
                    selected_genes.append(gene_info)
            
            # Add gene labels with smart radial positioning for optimal readability
            for gene_info in selected_genes[:max_gene_labels]:
                gene = gene_info['gene']
                angle = gene_info['angle']
                genomic_pos = gene_info['position']
                
                # Convert angle to radians
                rad = np.radians(angle)
                
                # Normalize angle to 0-360 range
                norm_angle = angle % 360
                
                # Smart positioning based on quadrant for optimal readability
                if 0 <= norm_angle <= 90:  # Top-right quadrant
                    label_radius = 0.6
                    text_angle = 0  # Horizontal text
                    anchor_x, anchor_y = 'left', 'bottom'
                elif 90 < norm_angle <= 180:  # Top-left quadrant  
                    label_radius = 0.6
                    text_angle = 0  # Horizontal text
                    anchor_x, anchor_y = 'right', 'bottom'
                elif 180 < norm_angle <= 270:  # Bottom-left quadrant
                    label_radius = 0.6  
                    text_angle = 0  # Horizontal text
                    anchor_x, anchor_y = 'right', 'top'
                else:  # Bottom-right quadrant (270-360)
                    label_radius = 0.6
                    text_angle = 0  # Horizontal text
                    anchor_x, anchor_y = 'left', 'top'
                
                # Calculate label position
                label_x = label_radius * np.cos(rad)
                label_y = label_radius * np.sin(rad)
                
                # Position for the arrow end (pointing to data point on the band)
                arrow_radius = 0.925  # Same as data points
                arrow_x = arrow_radius * np.cos(rad)
                arrow_y = arrow_radius * np.sin(rad)
                
                # Add the gene label with smart positioning
                fig.add_annotation(
                    x=label_x, y=label_y,
                    text=f"<b>{gene}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=MANUS_COLORS['primary_teal'],
                    ax=arrow_x,
                    ay=arrow_y,
                    font=dict(size=11, color=MANUS_COLORS['primary_teal'], family='Arial', weight='bold'),
                    textangle=text_angle,  # Always horizontal for readability
                    xanchor=anchor_x,
                    yanchor=anchor_y
                )
        
        # Professional title with library context
        title_text = f"<b>Circular Genome Map</b><br><sub style='color: {MANUS_COLORS['text_secondary']}'>CREATE Library  Elements"
        
        # Add library information if filtered
        if filter_dict and 'libname' in filter_dict and filter_dict['libname']:
            if len(filter_dict['libname']) == 1:
                title_text += f" | Library: {filter_dict['libname'][0]}"
            elif len(filter_dict['libname']) <= 3:
                title_text += f" | Libraries: {', '.join(filter_dict['libname'])}"
            else:
                title_text += f" | {len(filter_dict['libname'])} Libraries"
        
        if genome_info and 'genome_length' in genome_info:
            title_text += f" | {genome_info['genome_length']:,} bp genome</sub>"
        else:
            title_text += f" | {max_position:,} bp range</sub>"
        
        # Professional layout with transparent background
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=18, color=MANUS_COLORS['text_primary'], family='Arial Black')
            ),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.4, 1.4],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.4, 1.4]
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=MANUS_COLORS['primary_teal'],
                borderwidth=2,
                font=dict(size=11, family='Arial')
            ),
            width=800,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            margin=dict(l=50, r=180, t=100, b=50)
        )
        
        return fig

class MutationDiversityAnalyzer:
    """Module for mutation diversity analysis"""
    
    @staticmethod
    def create_mutation_heatmap(df, mutation_cols):
        """Create heatmap of mutation patterns"""
        if not mutation_cols:
            return None
        
        # Create mutation matrix
        available_cols = [col for col in mutation_cols if col in df.columns]
        if not available_cols:
            st.warning("Selected mutation columns not found in data")
            return None
        
        mutation_df = df[available_cols].copy()
        
        # For categorical data, create co-occurrence matrix
        if len(available_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation matrix for numerical data or co-occurrence for categorical
            try:
                corr_matrix = mutation_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax, 
                           cbar_kws={'shrink': 0.8})
                ax.set_title("Mutation Pattern Correlation Matrix")
            except:
                # For categorical data, create value counts heatmap
                value_counts = {}
                for col in available_cols:
                    value_counts[col] = mutation_df[col].value_counts().head(10)
                
                # Convert to matrix format
                all_values = set()
                for counts in value_counts.values():
                    all_values.update(counts.index)
                
                matrix_data = []
                for value in list(all_values)[:15]:  # Limit to top 15 values
                    row = []
                    for col in available_cols:
                        count = value_counts[col].get(value, 0)
                        row.append(count)
                    matrix_data.append(row)
                
                matrix_df = pd.DataFrame(matrix_data, 
                                       index=list(all_values)[:15], 
                                       columns=available_cols)
                
                # Create custom colormap from MANUS colors
                from matplotlib.colors import LinearSegmentedColormap
                manus_cmap = LinearSegmentedColormap.from_list(
                    'manus', 
                    [MANUS_COLORS['light_teal'], MANUS_COLORS['primary_teal'], MANUS_COLORS['dark_teal']]
                )
                sns.heatmap(matrix_df, annot=True, cmap=manus_cmap, ax=ax, 
                           cbar_kws={'shrink': 0.8})
                ax.set_title("Mutation Value Frequency Heatmap")
            
            plt.tight_layout()
            return fig
        
        return None
    
    @staticmethod
    def create_diversity_metrics(df, mutation_cols):
        """Calculate diversity metrics"""
        metrics = {}
        
        for col in mutation_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                total_count = df[col].count()
                most_common = df[col].value_counts().head(1)
                
                metrics[col] = {
                    'unique_mutations': unique_count,
                    'total_designs': total_count,
                    'diversity_ratio': unique_count / total_count if total_count > 0 else 0,
                    'most_common': most_common.index[0] if len(most_common) > 0 else 'N/A',
                    'most_common_count': most_common.iloc[0] if len(most_common) > 0 else 0
                }
        
        return metrics
    
    @staticmethod
    def create_mutation_length_analysis(df, mutation_col):
        """Analyze mutation length distribution"""
        if mutation_col not in df.columns:
            return None, None
        
        # Get mutation lengths
        mutation_lengths = df[mutation_col].dropna().astype(str).str.len()
        length_counts = mutation_lengths.value_counts().sort_index()
        
        return mutation_lengths, length_counts
    
    @staticmethod
    def create_mutation_frequency_plot(df, mutation_col, top_n=20):
        """Create bar chart of mutation frequency distribution"""
        if mutation_col not in df.columns:
            return None
        
        # Get value counts for mutations
        mutation_counts = df[mutation_col].value_counts().head(top_n)
        
        if len(mutation_counts) == 0:
            return None
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use custom color palette
        colors = (MANUS_PLOT_COLORS * ((len(mutation_counts) // len(MANUS_PLOT_COLORS)) + 1))[:len(mutation_counts)]
        
        bars = ax.bar(range(len(mutation_counts)), mutation_counts.values, 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
        
        # Customize the plot
        ax.set_xlabel("Mutation", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Top {len(mutation_counts)} Most Frequent Mutations", fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(range(len(mutation_counts)))
        ax.set_xticklabels(mutation_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, mutation_counts.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mutation_counts.values)*0.01,
                   str(count), ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_mutation_length_histogram(mutation_lengths):
        """Create histogram of mutation length distribution"""
        if mutation_lengths is None or len(mutation_lengths) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        counts, bins, patches = ax.hist(mutation_lengths, bins=range(int(mutation_lengths.min()), 
                                                                   int(mutation_lengths.max()) + 2),
                                       alpha=0.8, color=MANUS_COLORS['primary_teal'], 
                                       edgecolor='white', linewidth=0.8)
        
        ax.set_xlabel("Mutation Length (characters)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Mutation Lengths", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_length = mutation_lengths.mean()
        median_length = mutation_lengths.median()
        ax.axvline(mean_length, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_length:.1f}')
        ax.axvline(median_length, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_length:.1f}')
        ax.legend()
        
        plt.tight_layout()
        return fig

class SequenceCompositionAnalyzer:
    """Module for sequence composition and distance analysis"""
    
    @staticmethod
    def calculate_expected_gc_content(df, sequence_cols):
        """Calculate expected GC content from the design sequences"""
        expected_gc = {}
        
        for col in sequence_cols:
            if col in df.columns:
                sequences = df[col].dropna().astype(str)
                gc_values = []
                
                for seq in sequences:
                    try:
                        gc_content = calculate_gc_content(seq)
                        if not pd.isna(gc_content):
                            gc_values.append(gc_content)
                    except:
                        continue
                
                if gc_values:
                    # Calculate mean expected GC content for this sequence type
                    expected_gc[col] = np.mean(gc_values)
        
        return expected_gc

    @staticmethod
    def calculate_gc_content_distribution(df, sequence_cols):
        """Calculate GC content for sequence columns"""
        gc_data = {}
        
        for col in sequence_cols:
            if col in df.columns:
                gc_values = []
                # Vectorized calculation for better performance
                sequences = df[col].dropna().astype(str)
                for seq in sequences:
                    try:
                        gc_content = calculate_gc_content(seq)
                        if not pd.isna(gc_content):
                            gc_values.append(gc_content)
                    except:
                        continue
                gc_data[col] = gc_values
        
        return gc_data
    
    @staticmethod
    def calculate_distance_distribution(df, distance_cols):
        """Calculate distance distributions for distance-related columns"""
        distance_data = {}
        
        for col in distance_cols:
            if col in df.columns:
                distance_values = []
                for value in df[col].dropna():
                    try:
                        # Convert to numeric if it's not already
                        if isinstance(value, str):
                            # Handle cases where distance might be in string format
                            # e.g., "100bp", "50", "distance: 75"
                            numeric_value = re.search(r'(\d+(?:\.\d+)?)', str(value))
                            if numeric_value:
                                distance_value = float(numeric_value.group(1))
                            else:
                                continue
                        else:
                            distance_value = float(value)
                        
                        # Only include reasonable distance values (0 to 10MB)
                        if 0 <= distance_value <= 10000000:
                            distance_values.append(distance_value)
                    except (ValueError, TypeError):
                        continue
                
                if distance_values:  # Only add if we have valid data
                    distance_data[col] = distance_values
        
        return distance_data
    
    @staticmethod
    def create_gc_content_plots(gc_data, plot_type='histogram', expected_gc=None, use_fraction=True):
        """Create GC content visualization with fraction counts and expected values"""
        if not gc_data:
            return None
        
        # Prepare data for plotting
        plot_data = []
        for col, values in gc_data.items():
            for value in values:
                plot_data.append({'sequence_type': col, 'gc_content': value})
        
        if not plot_data:
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        if plot_type == 'histogram':
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Create separate histograms for each sequence type
            sequence_types = plot_df['sequence_type'].unique()
            # Use custom Manus color palette
            colors = (MANUS_PLOT_COLORS * ((len(sequence_types) // len(MANUS_PLOT_COLORS)) + 1))[:len(sequence_types)]
            
            for i, seq_type in enumerate(sequence_types):
                subset = plot_df[plot_df['sequence_type'] == seq_type]
                
                # Create histogram with appropriate normalization
                counts, bins, patches = ax.hist(subset['gc_content'], bins=30, alpha=0.8, 
                       label=f'{seq_type} (n={len(subset)})', color=colors[i], 
                       edgecolor='white', linewidth=0.8, density=use_fraction)
            
            # Expected GC content lines removed for cleaner visualization
            
            # Set title and labels based on count mode
            if use_fraction:
                ax.set_title("GC Content vs Count Abundance (Fraction)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Fraction Read Count", fontsize=12)
                count_mode_text = 'Count mode: Fraction'
            else:
                ax.set_title("GC Content vs Count Abundance (Raw)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Total Counts", fontsize=12)
                count_mode_text = 'Count mode: Total Counts'
            
            ax.set_xlabel("GC Content (%)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add count mode toggle info
            ax.text(0.02, 0.98, count_mode_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plt.tight_layout()
            return fig
            
        elif plot_type == 'swarm':
            fig, ax = plt.subplots(figsize=(10, 6))
            # For large datasets, sample data to avoid overcrowding
            if len(plot_df) > 2000:
                plot_df_sample = plot_df.sample(n=2000, random_state=42)
            else:
                plot_df_sample = plot_df
            sns.swarmplot(data=plot_df_sample, x='sequence_type', y='gc_content', ax=ax, 
                         palette=MANUS_PLOT_COLORS, size=4, alpha=0.8)
            ax.set_title("GC Content Distribution by Sequence Type", fontsize=14, fontweight='bold')
            ax.set_ylabel("GC Content (%)", fontsize=12)
            ax.set_xlabel("Sequence Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif plot_type == 'box':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=plot_df, x='sequence_type', y='gc_content', ax=ax, palette=MANUS_PLOT_COLORS)
            ax.set_title("GC Content Distribution by Sequence Type", fontsize=14, fontweight='bold')
            ax.set_ylabel("GC Content (%)", fontsize=12)
            ax.set_xlabel("Sequence Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif plot_type == 'violin':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=plot_df, x='sequence_type', y='gc_content', ax=ax, palette=MANUS_PLOT_COLORS)
            ax.set_title("GC Content Distribution by Sequence Type", fontsize=14, fontweight='bold')
            ax.set_ylabel("GC Content (%)", fontsize=12)
            ax.set_xlabel("Sequence Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        
        return None
    
    @staticmethod
    def create_distance_plots(distance_data, plot_type='histogram'):
        """Create distance visualization"""
        if not distance_data:
            return None
        
        # Prepare data for plotting
        plot_data = []
        for col, values in distance_data.items():
            for value in values:
                plot_data.append({'distance_type': col, 'distance': value})
        
        if not plot_data:
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        # Determine appropriate units for display
        max_distance = plot_df['distance'].max()
        if max_distance > 1000000:
            plot_df['distance_display'] = plot_df['distance'] / 1000000
            unit_label = "Distance (Mb)"
        elif max_distance > 1000:
            plot_df['distance_display'] = plot_df['distance'] / 1000
            unit_label = "Distance (kb)"
        else:
            plot_df['distance_display'] = plot_df['distance']
            unit_label = "Distance (bp)"
        
        if plot_type == 'histogram':
            fig, ax = plt.subplots(figsize=(10, 6))
            # Create separate histograms for each distance type
            distance_types = plot_df['distance_type'].unique()
            # Use custom Manus color palette
            colors = (MANUS_PLOT_COLORS * ((len(distance_types) // len(MANUS_PLOT_COLORS)) + 1))[:len(distance_types)]
            
            for i, dist_type in enumerate(distance_types):
                subset = plot_df[plot_df['distance_type'] == dist_type]
                ax.hist(subset['distance_display'], bins=30, alpha=0.8, 
                       label=dist_type, color=colors[i], edgecolor='white', linewidth=0.8)
            
            ax.set_title("Distance Distribution by Type", fontsize=14, fontweight='bold')
            ax.set_xlabel(unit_label, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif plot_type == 'swarm':
            fig, ax = plt.subplots(figsize=(10, 6))
            # For large datasets, sample data to avoid overcrowding
            if len(plot_df) > 2000:
                plot_df_sample = plot_df.sample(n=2000, random_state=42)
            else:
                plot_df_sample = plot_df
            sns.swarmplot(data=plot_df_sample, x='distance_type', y='distance_display', ax=ax, 
                         palette=MANUS_PLOT_COLORS, size=4, alpha=0.8)
            ax.set_title("Distance Distribution by Type", fontsize=14, fontweight='bold')
            ax.set_ylabel(unit_label, fontsize=12)
            ax.set_xlabel("Distance Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif plot_type == 'box':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=plot_df, x='distance_type', y='distance_display', ax=ax, palette=MANUS_PLOT_COLORS)
            ax.set_title("Distance Distribution by Type", fontsize=14, fontweight='bold')
            ax.set_ylabel(unit_label, fontsize=12)
            ax.set_xlabel("Distance Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif plot_type == 'violin':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=plot_df, x='distance_type', y='distance_display', ax=ax, palette=MANUS_PLOT_COLORS)
            ax.set_title("Distance Distribution by Type", fontsize=14, fontweight='bold')
            ax.set_ylabel(unit_label, fontsize=12)
            ax.set_xlabel("Distance Type", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        
        return None

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main Streamlit application"""
    
    # Debug: Add timestamp to identify multiple renders
    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
    
    # Branded header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🧬 CREATE Library  Analysis Platform</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Powered by Manus Bio | Advanced Genomic Design Visualization
        </p>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.6;">
            Debug ID: {timestamp}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads and controls
    st.sidebar.markdown("### 📁 Data Input")
    
    # File upload sections with enhanced styling
    data_file = st.sidebar.file_uploader(
        "Upload CREATE Library  Data",
        type=['csv', 'xlsx'],
        help="Upload your CREATE Library  design file (CSV or Excel format)"
    )
    
    genbank_file = st.sidebar.file_uploader(
        "Upload Reference GenBank File (Optional)",
        type=['gb', 'gbk'],
        help="Upload reference genome for context"
    )
    
    if data_file is None:
        st.info("👆 Please upload a CREATE Library  data file to begin analysis")
        st.markdown("### Expected Data Format")
        st.markdown("""
        Your file (CSV or Excel) should contain columns such as:
        - **genomic_start**: Genomic position
        - **name**: Gene or target name  
        - **CassetteSeq, Spacer**: Sequence data
        - **strand**: Forward/Reverse orientation
        - **libname**: Library identifier
        - **mutation_pattern**: Mutation information
        """)
        return

    # Load and process data with performance optimization
    try:
        # Load data efficiently
        file_data = data_file.getvalue()
        file_extension = data_file.name.split('.')[-1].lower()
        
        # Performance settings in sidebar
        with st.sidebar.expander("⚡ Performance Settings"):
            sample_large_data = st.checkbox(
                "Enable Smart Sampling for Large Datasets", 
                value=True,
                help="Automatically sample large datasets (>10k rows) for faster performance"
            )
            
            if sample_large_data:
                max_rows = st.slider(
                    "Maximum Rows to Process",
                    min_value=1000,
                    max_value=50000,
                    value=15000,
                    step=1000,
                    help="Larger values provide more complete analysis but slower performance"
                )
            else:
                max_rows = None
        
        # Load the data based on file type
        if file_extension == 'csv':
            df = pd.read_csv(data_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(data_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return
        
        original_size = len(df)
        
        # Apply sampling if requested and needed
        if sample_large_data and max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
            is_sampled = True
        else:
            is_sampled = False
        
        st.sidebar.success(f"✅ Loaded {len(df):,} designs from {data_file.name}")
        
        # Show performance information
        show_performance_info(df, is_sampled, original_size)
        
        # Pre-process column information for efficiency
        with st.spinner("🔍 Analyzing data structure..."):
            column_info = preprocess_data_columns(df)
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return
    
    # Parse GenBank if provided
    genome_info = None
    if genbank_file:
        genome_info = parse_genbank_file(genbank_file)
        if genome_info:
            st.sidebar.success(f"✅ Loaded genome ({genome_info['genome_length']:,} bp)")
    
    # Data overview (always visible)
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Designs", len(df))
    with col2:
        unique_genes = df['name'].nunique() if 'name' in df.columns else 'N/A'
        st.metric("Unique Targets", unique_genes)
    with col3:
        # Check for mutation-related columns
        mutation_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['mutation', 'mutant', 'variant'])]
        if mutation_cols:
            # Use the first mutation column found
            mutation_col = mutation_cols[0]
            unique_mutations = df[mutation_col].nunique()
            st.metric("Mutation Types", unique_mutations)
        elif 'libname' in df.columns:
            unique_libs = df['libname'].nunique()
            st.metric("Libraries", unique_libs)
        else:
            st.metric("Categories", 'N/A')
    with col4:
        if genome_info:
            st.metric("Genome Size", f"{genome_info['genome_length']:,} bp")
    
    # Interactive filters in sidebar with performance optimization
    st.sidebar.header("🎛️ Interactive Controls")
    
    # Gene/target filter
    if 'name' in df.columns:
        available_genes = sorted(df['name'].unique())
        # Limit default selection for large datasets to improve performance
        max_default_genes = 10 if len(available_genes) > 50 else 5
        selected_genes = st.sidebar.multiselect(
            "Select Genes/Targets",
            options=available_genes,
            default=available_genes[:max_default_genes] if len(available_genes) > max_default_genes else available_genes,
            help=f"Select from {len(available_genes)} available genes"
        )
    else:
        selected_genes = []
    
    # Apply filters efficiently
    filters = {}
    if selected_genes and 'name' in df.columns:
        filters['name'] = selected_genes
    
    # Use efficient filtering function
    filtered_df = efficient_filter_data(df, filters)
    
    # Show filtering results with performance context
    if len(filtered_df) != len(df):
        st.info(f"📊 Showing {len(filtered_df):,} designs after filtering ({(len(filtered_df)/len(df)*100):.1f}% of loaded data)")
    else:
        st.info(f"📊 Showing all {len(filtered_df):,} loaded designs")
    
    # ================================================================================
    # TABBED INTERFACE FOR MAIN ANALYSIS SECTIONS
    # ================================================================================
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {MANUS_COLORS['primary_teal']} 0%, {MANUS_COLORS['accent_green']} 100%); 
                padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: white; margin: 0; text-align: center;">
            📊 Analysis Dashboard
        </h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
            Select a tab below to explore different analysis views
        </p>
        <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; text-align: center; font-size: 0.9rem;">
            ✅ Tabbed Interface Active - No Content Should Appear Outside Tabs
        </p>
        <p style="color: rgba(255,255,255,0.5); margin: 0.2rem 0 0 0; text-align: center; font-size: 0.7rem;">
            Dashboard ID: {timestamp}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create the main tab interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Circular Genome Map", 
        "🧬 GC Content Analysis", 
        "📏 Distance Analysis",
        "🧬 Mutation Diversity",
        "🔮 Advanced Analytics"
    ])
    
    # ================================================================================
    # TAB 1: CIRCULAR GENOME PLOT
    # ================================================================================
    
    with tab1:
        st.header("🔄 Circular Genome Visualization")
        st.markdown("Professional Circos-style genomic mapping of CRISPR elements")
        
        # Enhanced controls for circular plot
        with st.expander("🎨 Visualization Controls", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Use pre-processed position columns for efficiency
                position_columns = column_info['position_columns']
                if not position_columns:
                    position_columns = column_info['numeric_columns']
                
                position_column = st.selectbox(
                    "📍 Genomic Position",
                    options=position_columns,
                    index=0 if position_columns else 0,
                    help="Select the column containing genomic coordinates",
                    key="circular_position"
                )
            
            with col2:
                # Use pre-processed categorical columns for efficiency
                color_options = []
                priority_cols = ['name', 'gene', 'target', 'libname', 'library', 'strand', 'mutation_pattern', 'type']
                
                # Add priority columns first if they exist
                for col in priority_cols:
                    if col in df.columns:
                        color_options.append(col)
                
                # Add other categorical columns from pre-processed data
                for col in column_info['categorical_columns']:
                    if col not in color_options:
                        color_options.append(col)
                
                color_column = st.selectbox(
                    "🎨 Color Coding",
                    options=color_options,
                    index=0 if color_options else 0,
                    help="Select the column for color-coding data points",
                    key="circular_color"
                )
            
            with col3:
                plot_style = st.selectbox(
                    "🎭 Plot Style",
                    options=["Professional Circos", "Minimal", "Detailed"],
                    index=0,
                    help="Choose the visualization style",
                    key="circular_style"
                )
            
            with col4:
                # Gene label controls
                max_genes = len(df['name'].unique()) if 'name' in df.columns else 20
                num_gene_labels = st.slider(
                    "🏷️ Gene Labels",
                    min_value=0,
                    max_value=min(max_genes, 25),
                    value=min(12, max_genes),
                    step=1,
                    help="Maximum number of gene labels to display (with overlap avoidance)",
                    key="circular_labels"
                )
        
        # Gene name input for GenBank highlighting
        with st.expander("🧬 GenBank Gene Highlighting", expanded=False):
            st.markdown("Enter gene names from your GenBank file to highlight specific genes in the circular plot:")
            gene_names_input = st.text_area(
                "Gene Names (one per line or comma-separated)",
                placeholder="Example:\nrpoB\nkatG\ninhA\n\nOr: rpoB, katG, inhA",
                help="Enter gene names that exist in your GenBank annotation. These will be highlighted on the circular plot.",
                key="gene_names_input"
            )
            
            # Parse the gene names
            selected_genes = []
            if gene_names_input.strip():
                # Handle both line-separated and comma-separated input
                if '\n' in gene_names_input:
                    selected_genes = [name.strip() for name in gene_names_input.strip().split('\n') if name.strip()]
                else:
                    selected_genes = [name.strip() for name in gene_names_input.strip().split(',') if name.strip()]
                
                if selected_genes:
                    st.success(f"✅ Will highlight {len(selected_genes)} genes: {', '.join(selected_genes[:5])}" + 
                              (f" and {len(selected_genes)-5} more..." if len(selected_genes) > 5 else ""))
                    
                    # Check which genes are available in the data
                    if genome_info and 'genes' in genome_info:
                        available_genes = [gene['gene'] for gene in genome_info['genes'] if 'gene' in gene]
                        found_genes = [gene for gene in selected_genes if gene in available_genes]
                        missing_genes = [gene for gene in selected_genes if gene not in available_genes]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if found_genes:
                                st.info(f"📍 **Found in GenBank:** {', '.join(found_genes)}")
                        with col2:
                            if missing_genes:
                                st.warning(f"⚠️ **Not found:** {', '.join(missing_genes)}")
            else:
                selected_genes = []
        
        # Data quality metrics
        if position_column and color_column:
            with st.expander("📊 Data Quality Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    valid_positions = filtered_df[position_column].notna().sum()
                    st.metric("Valid Positions", f"{valid_positions:,}", 
                             f"{(valid_positions/len(filtered_df)*100):.1f}%")
                
                with col2:
                    position_range = filtered_df[position_column].max() - filtered_df[position_column].min()
                    st.metric("Position Span", f"{position_range:,} bp")
                
                with col3:
                    categories = filtered_df[color_column].nunique()
                    st.metric("Color Categories", f"{categories}")
                
                with col4:
                    density = len(filtered_df) / (position_range / 1000) if position_range > 0 else 0
                    st.metric("Element Density", f"{density:.1f}/kb")
        
        if position_column:
            try:
                # Create circular plot
                circular_plotter = CircularGenomePlotter()
                filters = {}
                if selected_genes and 'name' in df.columns:
                    filters['name'] = selected_genes
                
                circular_fig = circular_plotter.create_circular_plot(
                    df, genome_info, position_column, color_column, filters, num_gene_labels
                )
                
                if circular_fig:
                    # Display the plot in a styled container
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    st.plotly_chart(circular_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced plot statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Elements Plotted", f"{len(filtered_df):,}")
                    with col2:
                        if genome_info and 'genome_length' in genome_info:
                            coverage = (len(filtered_df) / genome_info['genome_length']) * 100000
                            st.metric("📏 Coverage", f"{coverage:.1f}/100kb")
                        else:
                            # Show library count if no genome info
                            if 'libname' in filtered_df.columns:
                                lib_count = filtered_df['libname'].nunique()
                                st.metric("📚 Libraries", f"{lib_count}")
                            else:
                                st.metric("📏 Data Range", f"{(filtered_df[position_column].max() - filtered_df[position_column].min()):,} bp")
                    with col3:
                        st.metric("🏷️ Categories", f"{filtered_df[color_column].nunique()}")
                    with col4:
                        avg_spacing = (filtered_df[position_column].max() - filtered_df[position_column].min()) / len(filtered_df)
                        st.metric("📐 Avg Spacing", f"{avg_spacing:,.0f} bp")
                
                # Enhanced action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Refresh Visualization", help="Regenerate the plot with current settings", key="refresh_circular"):
                        st.rerun()
                
                with col2:
                    if st.button("💾 Export Plot Data", help="Download the data used in this visualization", key="export_circular"):
                        csv_buffer = io.StringIO()
                        plot_data = filtered_df[[position_column, color_column] + 
                                              (['name'] if 'name' in filtered_df.columns else [])]
                        plot_data.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"circos_plot_data_{position_column}.csv",
                            mime="text/csv",
                            key="download_circos"
                        )
                
                with col3:
                    if st.button("📋 Copy Settings", help="Copy current visualization settings", key="copy_circular"):
                        settings = {
                            'position_column': position_column,
                            'color_column': color_column,
                            'plot_style': plot_style,
                            'selected_genes': selected_genes
                        }
                        st.success("Settings copied to clipboard!")
                        st.json(settings)
                    
            except Exception as e:
                st.error(f"❌ Error creating circular plot: {str(e)}")
                with st.expander("🔧 Troubleshooting", expanded=True):
                    st.markdown(f"""
                    **Common Issues:**
                    - Ensure the position column contains numeric genomic coordinates
                    - Check that your data has been properly filtered
                    - Verify that position values are reasonable (> 0, < genome size)
                    - Try selecting different position or color columns
                    
                    **Data Info:**
                    - Position column: `{position_column}`
                    - Color column: `{color_column}`
                    - Filtered rows: {len(filtered_df):,}
                    - Valid positions: {filtered_df[position_column].notna().sum() if position_column in filtered_df.columns else 'N/A'}
                    """)
        else:
            st.warning("⚠️ No suitable genomic position columns detected in your data.")
            st.info("💡 Expected column names: genomic_start, position, start, coord, etc.")
    
    # ================================================================================
    # TAB 2: GC CONTENT ANALYSIS
    # ================================================================================
    
    with tab2:
        st.header("🧬 GC Content Analysis")
        
        # Column selection for sequence analysis
        sequence_options = [col for col in filtered_df.columns if any(term in col.lower() 
                               for term in ['seq', 'spacer', 'cassette', 'sequence'])]
        
        # Add any text columns that might contain sequences
        for col in filtered_df.columns:
            if (filtered_df[col].dtype == 'object' and 
                col not in sequence_options and
                any(filtered_df[col].dropna().astype(str).str.match(r'^[ATCGN]+$', na=False))):
                sequence_options.append(col)
        
        sequence_columns = st.multiselect(
            "Select Sequence Columns",
            options=sequence_options,
            default=['CassetteSeq', 'Spacer'] if all(col in sequence_options for col in ['CassetteSeq', 'Spacer']) else sequence_options[:2],
            key="gc_sequence_columns_tab",
            help=f"Found {len(sequence_options)} potential sequence columns in filtered data"
        )
        
        if sequence_columns:
            # Enhanced controls with fraction display and expected values
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.info(f"🧬 Analyzing {len(filtered_df)} designs from {len(sequence_columns)} sequence columns")
            with col2:
                gc_plot_type = st.selectbox(
                    "Plot Type",
                    options=['histogram', 'box', 'violin', 'swarm'],
                    index=0,  # Default to histogram
                    help="Histogram shows frequency distribution, swarm shows individual data points",
                    key="gc_plot_type_tab"
                )
            with col3:
                show_expected = st.checkbox(
                    "Show Expected GC",
                    value=True,
                    help="Display expected GC content from design sequences",
                    key="show_expected_tab"
                )
            with col4:
                count_mode = st.selectbox(
                    "Count Mode",
                    options=["Fraction", "Total Counts"],
                    index=0,
                    help="Display as fraction (normalized) or total counts",
                    key="count_mode_tab"
                )
            
            # Calculate and plot GC content
            try:
                with st.spinner("🔬 Calculating GC content..."):
                    seq_analyzer = SequenceCompositionAnalyzer()
                    gc_data = seq_analyzer.calculate_gc_content_distribution(filtered_df, sequence_columns)
                    
                    # Calculate expected GC content if requested
                    expected_gc = None
                    if show_expected:
                        expected_gc = seq_analyzer.calculate_expected_gc_content(filtered_df, sequence_columns)
                
                total_sequences = sum(len(values) for values in gc_data.values())
                st.success(f"✅ Processed {total_sequences} sequences from {len(gc_data)} columns")
                
                # Display expected GC values if calculated
                if expected_gc:
                    st.subheader("📊 Expected GC Content from Design")
                    exp_cols = st.columns(len(expected_gc))
                    for i, (seq_type, exp_gc) in enumerate(expected_gc.items()):
                        with exp_cols[i]:
                            st.metric(f"{seq_type} Expected GC", f"{exp_gc:.1f}%")
                
                if gc_data and total_sequences > 0:
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    use_fraction = (count_mode == "Fraction")
                    try:
                        gc_fig = seq_analyzer.create_gc_content_plots(gc_data, gc_plot_type, expected_gc, use_fraction)
                        if gc_fig:
                            st.pyplot(gc_fig)
                            plt.close(gc_fig)  # Close the figure to free memory
                        else:
                            st.error("❌ Failed to create GC content plot")
                    except Exception as e:
                        st.error(f"❌ Error creating GC content plot: {str(e)}")
                        st.write(f"Debug info: Plot type: {gc_plot_type}, Data keys: {list(gc_data.keys())}, Total sequences: {total_sequences}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add summary statistics
                    st.subheader("📈 GC Content Summary Statistics")
                    summary_cols = st.columns(len(sequence_columns))
                    for i, col in enumerate(sequence_columns):
                        if col in gc_data and gc_data[col]:
                            with summary_cols[i]:
                                gc_values = gc_data[col]
                                st.metric(
                                    f"{col} Mean GC",
                                    f"{np.mean(gc_values):.1f}%",
                                    f"±{np.std(gc_values):.1f}%"
                                )
                                st.write(f"Range: {np.min(gc_values):.1f}% - {np.max(gc_values):.1f}%")
                else:
                    st.warning("⚠️ No valid sequence data found. Check that selected columns contain DNA sequences.")
                    
            except Exception as e:
                st.error(f"❌ Error in GC content analysis: {str(e)}")
        else:
            st.info("💡 Select sequence columns to analyze GC content distribution.")
    
    # ================================================================================
    # TAB 3: DISTANCE ANALYSIS
    # ================================================================================
    
    with tab3:
        st.header("📏 Edit to Cut Distance Analysis")
        
        # Detect distance columns from filtered data
        distance_columns = []
        primary_distance_col = None
        
        # First, look for the primary "distance" column (case-insensitive)
        for col in filtered_df.columns:
            if col.lower() == 'distance':
                primary_distance_col = col
                distance_columns.append(col)
                break
        
        # Look for other distance-related columns
        distance_patterns = [
            'dist', 'spacing', 'gap', 'length', 'size', 
            'pam_distance', 'target_distance', 'genomic_distance',
            'upstream', 'downstream', 'offset', 'interval'
        ]
        
        for col in filtered_df.columns:
            col_lower = col.lower()
            # Skip if it's already the primary distance column
            if col == primary_distance_col:
                continue
            # Check if column name contains distance-related terms
            if any(pattern in col_lower for pattern in distance_patterns):
                # Verify it contains numeric data
                if filtered_df[col].dtype in ['int64', 'float64']:
                    distance_columns.append(col)
        
        # Also look for any numeric columns that might represent distances
        for col in filtered_df.columns:
            if col not in distance_columns and filtered_df[col].dtype in ['int64', 'float64']:
                # Check if values are in reasonable distance range (1 to 10M)
                numeric_vals = filtered_df[col].dropna()
                if len(numeric_vals) > 0:
                    min_val, max_val = numeric_vals.min(), numeric_vals.max()
                    if 1 <= min_val and max_val <= 10000000:
                        distance_columns.append(col)
        
        # Column selection for distance analysis
        if primary_distance_col:
            # Default to just the primary distance column
            default_selection = [primary_distance_col]
            help_text = f"Primary distance column '{primary_distance_col}' selected by default. Add other columns to create additional groups on the x-axis."
        else:
            # Fallback to first available distance column
            default_selection = distance_columns[:1] if distance_columns else []
            help_text = "Select columns containing distance measurements. Multiple columns will create separate groups on the x-axis."
        
        selected_distance_columns = st.multiselect(
            "Select Distance/Numeric Columns",
            options=distance_columns,
            default=default_selection,
            help=help_text,
            key="distance_columns_tab"
        )
        
        if selected_distance_columns:
            # Add plot type selector for better performance
            col1, col2 = st.columns([3, 1])
            with col1:
                if len(selected_distance_columns) == 1:
                    st.info(f"📊 Analyzing distances from **{selected_distance_columns[0]}** column in {len(filtered_df)} designs.")
                else:
                    st.info(f"📊 Analyzing distances from **{len(selected_distance_columns)} columns**: {', '.join(selected_distance_columns)}.")
            with col2:
                distance_plot_type = st.selectbox(
                    "Plot Type",
                    options=['histogram', 'box', 'violin', 'swarm'],
                    index=0,  # Default to histogram
                    help="Histogram shows frequency distribution, swarm shows individual data points for distance analysis",
                    key="distance_plot_type_tab"
                )
            
            # Calculate and plot distance analysis
            try:
                with st.spinner("📏 Calculating distance distributions..."):
                    seq_analyzer = SequenceCompositionAnalyzer()
                    distance_data = seq_analyzer.calculate_distance_distribution(filtered_df, selected_distance_columns)
                
                total_measurements = sum(len(values) for values in distance_data.values())
                st.success(f"✅ Processed {total_measurements} distance measurements from {len(distance_data)} columns")
                
                if distance_data and total_measurements > 0:
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    try:
                        distance_fig = seq_analyzer.create_distance_plots(distance_data, distance_plot_type)
                        if distance_fig:
                            st.pyplot(distance_fig)
                            plt.close(distance_fig)  # Close the figure to free memory
                        
                        # Add distance summary statistics
                        st.subheader("📈 Distance Summary Statistics")
                        summary_cols = st.columns(len(selected_distance_columns))
                        for i, col in enumerate(selected_distance_columns):
                            if col in distance_data and distance_data[col]:
                                with summary_cols[i]:
                                    dist_values = distance_data[col]
                                    st.metric(
                                        f"{col} Mean",
                                        f"{np.mean(dist_values):,.0f} bp",
                                        f"±{np.std(dist_values):,.0f} bp"
                                    )
                                    st.write(f"Range: {np.min(dist_values):,.0f} - {np.max(dist_values):,.0f} bp")
                                    st.write(f"Median: {np.median(dist_values):,.0f} bp")
                        else:
                            st.error("❌ Failed to create distance plot")
                    except Exception as e:
                        st.error(f"❌ Error creating distance plot: {str(e)}")
                        st.write(f"Debug info: Plot type: {distance_plot_type}, Data keys: {list(distance_data.keys())}, Total measurements: {total_measurements}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No valid distance data found. Check that selected columns contain numeric values.")
                
            except Exception as e:
                st.error(f"❌ Error in distance analysis: {str(e)}")
        else:
            if not distance_columns:
                st.warning("⚠️ No distance-related columns detected in your data.")
                st.markdown("""
                **Expected column names for distance analysis:**
                - `distance` (primary default)
                - `pam_distance`, `target_distance`, `genomic_distance`
                - `spacing`, `gap`, `length`, `size`, `offset`
                - Any numeric columns with reasonable distance values
                """)
            else:
                st.info("💡 Select distance/numeric columns to analyze their distributions.")
    
    # ================================================================================
    # TAB 4: MUTATION DIVERSITY SECTION
    # ================================================================================
    
    with tab4:
        st.header("🧬 Mutation Diversity Analysis")
        
        # Show library context if available
        if 'libname' in filtered_df.columns:
            unique_libs_in_view = filtered_df['libname'].nunique()
            if unique_libs_in_view <= 5:
                lib_names = sorted(filtered_df['libname'].unique())
                st.info(f"📚 **Libraries in view:** {', '.join(lib_names)}")
            else:
                st.info(f"📚 **{unique_libs_in_view} libraries** represented in current view")
        
        # Column selection for mutation analysis
        mutation_columns = st.multiselect(
            "Select Columns for Mutation Analysis",
            options=[col for col in df.columns if any(term in col.lower() 
                    for term in ['mutation', 'target', 'spacer', 'cassette'])],
            default=[col for col in df.columns if 'mutation' in col.lower()][:3],
            key="mutation_columns_tab"
        )
        
        if mutation_columns:
            # Calculate unique mutations
            unique_mutations = identify_unique_mutations(filtered_df, mutation_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Mutation Patterns")
                if not unique_mutations.empty:
                    st.dataframe(unique_mutations.head(10), use_container_width=True)
                
            with col2:
                st.subheader("Diversity Metrics")
                analyzer = MutationDiversityAnalyzer()
                metrics = analyzer.create_diversity_metrics(filtered_df, mutation_columns)
                
                for col, data in metrics.items():
                    st.metric(
                        f"{col} Diversity",
                        f"{data['diversity_ratio']:.2f}",
                        f"{data['unique_mutations']} unique / {data['total_designs']} total"
                    )
            
            # Mutation heatmap
            st.subheader("Mutation Pattern Heatmap")
            try:
                heatmap_fig = analyzer.create_mutation_heatmap(filtered_df, mutation_columns)
                if heatmap_fig:
                    st.pyplot(heatmap_fig)
                    plt.close(heatmap_fig)  # Close the figure to free memory
                else:
                    st.info("No mutation heatmap generated (need at least 2 columns)")
            except Exception as e:
                st.error(f"❌ Error creating mutation heatmap: {str(e)}")
        else:
            st.info("💡 Select mutation-related columns to analyze diversity patterns.")
    
    # ================================================================================
    # TAB 5: ADVANCED ANALYTICS SECTION
    # ================================================================================
    
    with tab5:
        st.header("🔮 Advanced Analytics")
        st.info("""
        **Coming Soon:**
        - 🧮 **Clustering Analysis**: Group similar CRISPR designs
        - 📊 **Dimensionality Reduction**: PCA/t-SNE visualization  
        - 🔍 **Motif Analysis**: Identify sequence patterns
        - 📈 **Efficiency Prediction**: ML-based design scoring
        - 🔗 **Pathway Integration**: Link to metabolic networks
        """)
        
        # Placeholder for future features
        st.markdown("### 🧪 Experimental Features")
        st.warning("Advanced analytics features are under development. Check back in future releases!")
    
    # Data export options (outside tabs, always visible in sidebar)
    st.sidebar.header("💾 Export Options")
    
    # Export format selection
    export_format = st.sidebar.selectbox(
        "Export Format",
        options=['CSV', 'Excel'],
        index=0,
        help="Choose the format for downloading filtered data"
    )
    
    if st.sidebar.button("📊 Export Filtered Data"):
        if export_format == 'CSV':
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name="filtered_crispr_data.csv",
                mime="text/csv"
            )
        else:  # Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='CRISPR_Data', index=False)
            excel_buffer.seek(0)
            
            st.sidebar.download_button(
                label="Download Excel",
                data=excel_buffer.getvalue(),
                file_name="filtered_crispr_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Removed library grouping functionality for performance

if __name__ == "__main__":
    main()
