import streamlit as st
st.set_page_config(layout="wide")

# Enable caching for performance
@st.cache_data
def load_and_process_library_files(library_files_info):
    """Cache library file loading and basic processing"""
    library_dfs = []
    for file_info in library_files_info:
        file_name, file_data = file_info
        if file_name.endswith('.csv'):
            df = pd.read_csv(StringIO(file_data))
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(file_data))
        else:
            continue
        library_dfs.append(df)
    
    if library_dfs:
        return pd.concat(library_dfs, ignore_index=True)
    return None

@st.cache_data
def load_and_process_hit_matrix(hit_matrix_data, hit_matrix_name):
    """Cache hit matrix loading and basic processing"""
    hit_df = pd.read_csv(StringIO(hit_matrix_data), index_col=0)
    return hit_df

@st.cache_data
def compute_gc_content_data(library_df_dict, cassette_id_col, grouping_col, gc_col, selected_groups):
    """Pre-compute all GC content related data"""
    library_df = pd.DataFrame(library_df_dict)
    
    # Filter for selected groups
    filtered_library = library_df[library_df[grouping_col].isin(selected_groups)]
    
    # Get GC data for all cassettes in selected groups
    gc_data = filtered_library[[cassette_id_col, gc_col]].dropna()
    
    return {
        'cassette_gc_map': dict(zip(gc_data[cassette_id_col], gc_data[gc_col])),
        'all_gc_values': gc_data[gc_col].values,
        'cassette_to_group': dict(zip(library_df[cassette_id_col], library_df[grouping_col]))
    }

@st.cache_data
def compute_gc_binning(plot_df_dict, bin_edges, expected_cassettes_by_bin):
    """Cache expensive binning computation"""
    plot_df = pd.DataFrame(plot_df_dict)
    
    # Create bins for observed data
    plot_df['GC_Bin'] = pd.cut(plot_df['GC_Content'], bins=bin_edges, right=False)
    observed_binned = plot_df.groupby('GC_Bin', observed=True).agg({
        'Total_Count': 'sum',
        'Cassette': 'count'
    }).reset_index()
    
    # Create expected dataframe
    expected_binned = pd.DataFrame({
        'GC_Bin': list(expected_cassettes_by_bin.keys()),
        'Expected_Cassettes': list(expected_cassettes_by_bin.values())
    })
    
    return observed_binned.to_dict('records'), expected_binned.to_dict('records')

st.markdown(
    '''
    <style>
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            max-width: 100vw;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

import pandas as pd
import numpy as np
import glob
import os
import plotly.express as px
import scipy.stats as stats
from scipy.stats import skew, nbinom
from io import StringIO, BytesIO

# --- App Title and Description ---
st.title("CREATE Cassette Pool QC Explorer")
st.markdown("""
This app loads cassette hit matrices and cassette library metadata, allowing interactive exploration, grouping by subpool/P2, and cross-amplification analysis.
""")

# --- File Loaders ---
def find_hit_matrices(root_dir):
    return glob.glob(os.path.join(root_dir, '**', 'cassette_hit_matrix.csv'), recursive=True)

def load_library_file(library_file):
    # Accepts a Streamlit UploadedFile object
    if library_file.name.endswith('.csv'):
        return pd.read_csv(library_file)
    elif library_file.name.endswith('.xlsx'):
        return pd.read_excel(library_file)
    else:
        st.error("Unsupported library file format. Please upload a CSV or XLSX file.")
        return None

# --- Sidebar: File Selection ---
st.sidebar.header("Data Selection")
# Allow multiple library files
library_files = st.sidebar.file_uploader("Upload cassette library file(s) (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)
# Allow single hit matrix file (optional)
hit_matrix_file = st.sidebar.file_uploader("Upload hit matrix file (optional, CSV)", type=["csv"])

# --- Load and Merge Library Files ---
if library_files:
    # Prepare file info for caching (since UploadedFile objects can't be cached directly)
    library_files_info = []
    for lib_file in library_files:
        file_data = lib_file.read().decode('utf-8') if lib_file.name.endswith('.csv') else lib_file.read()
        library_files_info.append((lib_file.name, file_data))
    
    # Use cached function
    library_df = load_and_process_library_files(library_files_info)
    
    if library_df is not None:
        st.write("Merged cassette library metadata:", library_df.head())
    else:
        st.warning("No valid library files uploaded.")
        st.stop()
else:
    st.warning("Please upload at least one cassette library file to proceed.")
    st.stop()

# --- Validate Library Columns ---
if 'mutantname' in library_df.columns:
    cassette_id_col = 'mutantname'
elif 'CassetteDescription' in library_df.columns:
    cassette_id_col = 'CassetteDescription'
else:
    st.error("Library file(s) must contain 'mutantname' or 'CassetteDescription' column.")
    st.stop()

# --- Select Grouping Column ---
st.sidebar.header("Grouping Options")
available_columns = [col for col in library_df.columns if col != cassette_id_col]
default_grouping_col = 'P2' if 'P2' in available_columns else available_columns[0] if available_columns else None

if not available_columns:
    st.error("Library file(s) must contain at least one column for grouping besides the cassette identifier.")
    st.stop()

grouping_col = st.sidebar.selectbox(
    "Select column for grouping/subpool analysis",
    available_columns,
    index=available_columns.index(default_grouping_col) if default_grouping_col in available_columns else 0,
    help="Choose which column to use for grouping cassettes (e.g., P2, subpool, etc.)"
)

# --- Performance Options ---
st.sidebar.header("Performance Options")
total_cassettes = len(library_df)
if total_cassettes > 5000:
    st.sidebar.warning(f"Large dataset detected ({total_cassettes:,} cassettes)")
    use_sampling = st.sidebar.checkbox(
        "Enable data sampling for faster performance", 
        value=total_cassettes > 10000,
        help="Sample data for faster visualization. Recommended for >10k cassettes."
    )
    if use_sampling:
        sample_size = st.sidebar.slider(
            "Sample size",
            min_value=1000,
            max_value=min(20000, total_cassettes),
            value=min(5000, total_cassettes),
            step=1000,
            help="Number of cassettes to sample for analysis"
        )
        # Apply sampling
        library_df = library_df.sample(n=sample_size, random_state=42)
        st.sidebar.info(f"Using {sample_size:,} randomly sampled cassettes")
else:
    use_sampling = False

# --- Load Hit Matrix ---
if hit_matrix_file is not None:
    try:
        # Use cached function for hit matrix loading
        hit_matrix_data = hit_matrix_file.read().decode('utf-8')
        all_hits = load_and_process_hit_matrix(hit_matrix_data, hit_matrix_file.name)
        all_hits['__source_file__'] = hit_matrix_file.name
        st.write(f"Loaded hit matrix from {hit_matrix_file.name}")
    except Exception as e:
        st.error(f"Failed to load hit matrix: {e}")
        st.stop()
else:
    st.warning("No hit matrix file uploaded. Please upload a hit matrix CSV file.")
    st.stop()

# --- Merge all_hits with Library Metadata ---
# The index of all_hits is assumed to be the cassette identifier
cassette_ids_in_hits = all_hits.index.unique().tolist()

# Prepare for merge: reset index to get cassette id as a column
all_hits_reset = all_hits.reset_index().rename(columns={all_hits.index.name or 'index': cassette_id_col})

# Merge on cassette_id_col
merged_hits = pd.merge(
    all_hits_reset,
    library_df[[cassette_id_col, grouping_col]],
    on=cassette_id_col,
    how='left',
    indicator=True
)

# Report cassettes not found in library
not_found = merged_hits[merged_hits['_merge'] == 'left_only'][cassette_id_col].unique()
if len(not_found) > 0:
    st.warning(f"{len(not_found)} cassettes in hit matrix not found in library: {not_found[:10]}{'...' if len(not_found) > 10 else ''}")

# Drop merge indicator for downstream use
merged_hits = merged_hits.drop(columns=['_merge'])

st.write("Merged hit matrix with library metadata:", merged_hits.head())

# --- Prepare Data for Heatmap Visualization ---
# Assume count column is present (if not, use first numeric column after cassette_id_col)
count_col = None
for col in merged_hits.columns:
    if col not in [cassette_id_col, 'P2', '__source_file__'] and pd.api.types.is_numeric_dtype(merged_hits[col]):
        count_col = col
        break
if count_col is None:
    st.error("No numeric count column found in hit matrices.")
    st.stop()

# Pivot: rows = samples (source files), columns = cassettes, values = counts
heatmap_df = merged_hits.pivot_table(
    index='__source_file__',
    columns=cassette_id_col,
    values=count_col,
    fill_value=0
)

# Attach grouping info for each cassette (column)
cassette_group_map = library_df.set_index(cassette_id_col)[grouping_col].to_dict()
group_for_cassette = [cassette_group_map.get(cid, 'Unknown') for cid in heatmap_df.columns]

# --- UI Controls for Filtering ---
unique_groups = sorted(set(group_for_cassette))
selected_groups = st.multiselect(f"Filter cassettes by {grouping_col} group", unique_groups, default=unique_groups)

# Filter columns (cassettes) by selected groups
def cassette_in_selected_group(cid):
    return cassette_group_map.get(cid, 'Unknown') in selected_groups
filtered_cassettes = [cid for cid in heatmap_df.columns if cassette_in_selected_group(cid)]
filtered_heatmap_df = heatmap_df[filtered_cassettes]
filtered_group_for_cassette = [cassette_group_map.get(cid, 'Unknown') for cid in filtered_cassettes]

# Optionally filter samples
unique_samples = heatmap_df.index.tolist()
selected_samples = st.multiselect("Filter samples (source files)", unique_samples, default=unique_samples)
filtered_heatmap_df = filtered_heatmap_df.loc[selected_samples]

# Define custom color scale
custom_palette = ['#17b6a7', '#1ccfc1', '#0b3b3e', '#2a7c7c', '#3e8e8e', '#5e9e9e', '#7ebebe', '#a3cfcf', '#cfeaea', '#e5eae3', '#bfae8c']

# Define monotonic navy-to-teal color scale for heatmaps
navy_teal_scale = ['#0b3b3e', '#17b6a7']

# --- Multi-select for grouping columns ---
if grouping_col in library_df.columns:
    all_groups = sorted(library_df[grouping_col].dropna().unique())
    selected_groups_analysis = st.multiselect(f"Select {grouping_col} group(s) for analysis", all_groups, default=all_groups, key="group_multiselect")
    # Get cassettes in selected groups
    cassettes_in_group = set(library_df[library_df[grouping_col].isin(selected_groups_analysis)][cassette_id_col])
else:
    selected_groups_analysis = []
    cassettes_in_group = set(hit_df.index)

# --- Multi-select for samples ---
if hit_matrix_file is not None:
    hit_matrix_file.seek(0)
    hit_df = pd.read_csv(hit_matrix_file, index_col=0)
    sample_names = hit_df.columns.tolist()
    selected_samples = st.multiselect("Select samples for analysis", sample_names, default=sample_names[:1], key="sample_multiselect")
else:
    selected_samples = []

# --- Tabbed Layout for Better Performance ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Hit Count Histogram", "🎯 Coupon Collector", "🔥 Sample Heatmap", "🧬 GC Content (Raw)", "🧬 GC Content (Normalized)"])

with tab1:
    st.header("Per-Sample Cassette Hit Count Histogram")
    if hit_matrix_file is not None and selected_samples:
        sample = selected_samples[0]
        sample_counts = hit_df[sample]
        # Only plot cassettes with >0 reads and in selected groups
        filtered_cassettes = [cass for cass in hit_df.index if cass in cassettes_in_group]
        sample_counts = sample_counts.loc[filtered_cassettes]
        nonzero_counts = sample_counts[sample_counts > 0]
        cassette_detected = nonzero_counts.count()
        total_reads = sample_counts.sum()
        # Determine expected cassettes for this sample's dominant group
        if grouping_col in library_df.columns:
            cassette_to_group = library_df.set_index(cassette_id_col)[grouping_col].to_dict()
            group_counts = {}
            for cassette, count in sample_counts.items():
                group = cassette_to_group.get(cassette, None)
                if group is not None:
                    group_counts[group] = group_counts.get(group, 0) + count
            expected_group = max(group_counts, key=group_counts.get) if group_counts else None
            expected_cassettes = library_df[library_df[grouping_col] == expected_group][cassette_id_col].nunique() if expected_group else 0
            coverage = (cassette_detected / expected_cassettes * 100) if expected_cassettes > 0 else 0
        else:
            expected_group = None
            expected_cassettes = 0
            coverage = 0
        norm_factor = st.slider("Fit normalization (scales fit line)", min_value=0.1, max_value=3.0, value=1.0, step=0.01, key="hist_norm")
        mu_default = float(nonzero_counts.mean()) if cassette_detected > 0 else 1.0
        r_default = max(1.0, float(mu_default**2 / (nonzero_counts.var() - mu_default)) if nonzero_counts.var() > mu_default else 1.0)
        mu = st.slider("NegBin mean (μ)", min_value=0.1, max_value=float(max(30, mu_default*2)), value=float(mu_default), step=0.1, key="hist_mu")
        r = st.slider("NegBin dispersion (r)", min_value=0.1, max_value=100.0, value=float(r_default), step=0.1, key="hist_r")
        p = r / (r + mu)
        max_count = int(max(nonzero_counts.max(), mu * 3))
        x_vals = list(range(1, max_count + 1))
        nbinom_probs = nbinom.pmf(x_vals, r, p)
        expected_per_bin = nbinom_probs * cassette_detected * norm_factor
        nbinom_skewness = (2 - p) / ((r * (1 - p))**0.5) if r > 0 and p < 1 else float('nan')
        label = f"Cassettes detected: {cassette_detected} / {expected_cassettes} ({coverage:.1f}% coverage)<br>Total reads: {total_reads}<br>NegBin μ: {mu:.2f}, r: {r:.2f}<br>Fitted NegBin skewness: {nbinom_skewness:.2f}"
        fig = px.histogram(
            nonzero_counts,
            nbins=30,
            labels={'value': 'Read Count', 'count': 'Number of Cassettes'},
            title=f"Cassette Read Count Distribution for {sample}",
            color_discrete_sequence=['#17b6a7']
        )
        fig.add_scatter(x=x_vals, y=expected_per_bin, mode='lines', name='NegBin fit', line=dict(color='#0b3b3e', width=3, dash='dash'))
        fig.add_annotation(
            text=label,
            xref="paper", yref="paper",
            x=0.95, y=0.95, showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)
        if len(selected_samples) > 1:
            st.info("Multiple samples selected. Showing histogram for the first selected sample only.")
        st.markdown(
            """
            **Interpretation:** The histogram shows the observed distribution of cassette read counts for the selected sample (excluding cassettes with zero reads). The navy dashed line overlays the fitted Negative Binomial distribution, which is commonly used to model overdispersed count data (variance greater than the mean) as seen in sequencing experiments.
            
            - **μ (mean):** The average expected read count per cassette.
            - **r (dispersion/size):** Controls the spread of the distribution. Lower r means more overdispersion (wider spread, heavier tail); higher r approaches a Poisson distribution.
            - **Fit normalization:** Scales the fit line to visually match the histogram.
            - **Fitted NegBin skewness:** Quantifies the expected asymmetry for a random sampling process at this mean and dispersion.
            """
        )

with tab2:
    st.header("Coupon Collector Simulation: Unique Mutations vs. Samplings")
    if hit_matrix_file is not None and selected_samples:
        cassette_names = [cass for cass in hit_df.index if cass in cassettes_in_group]
        # Empirical probabilities: sum read counts across selected samples, add pseudocount 1, normalize
        empirical_counts = hit_df.loc[cassette_names, selected_samples].sum(axis=1) + 1
        empirical_probs = empirical_counts / empirical_counts.sum()
        # Efficiency input
        efficiency = st.number_input(
            "Editing efficiency (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=0.1,
            key="efficiency_input"
        )
        eff_frac = efficiency / 100.0
        # Library size input: use number_input instead of slider
        default_lib_size = len(cassette_names)
        lib_size = st.number_input(
            "Library size (number of unique cassettes)",
            min_value=1,
            max_value=default_lib_size,
            value=default_lib_size,
            step=1,
            key="lib_size_input"
        )
        lib_size = int(lib_size)
        # Ensure both curves use the same set of cassettes
        cassette_names = cassette_names[:lib_size]
        empirical_counts = empirical_counts.loc[cassette_names]
        empirical_probs = empirical_counts / empirical_counts.sum()
        # Simulation
        max_draws = st.slider("Number of samplings (draws)", min_value=1, max_value=lib_size*3, value=lib_size, key="cc_draws")
        n_sim = st.slider("Number of simulation runs", min_value=1, max_value=100, value=20, key="cc_runs")
        import numpy as np
        rng = np.random.default_rng()
        all_curves = []
        for _ in range(n_sim):
            seen = set()
            curve = []
            draws = rng.choice(cassette_names, size=max_draws, p=empirical_probs, replace=True)
            for i, cassette in enumerate(draws, 1):
                seen.add(cassette)
                curve.append(len(seen))
            all_curves.append(curve)
        mean_curve = np.mean(all_curves, axis=0) * eff_frac
        # Theoretical (equiprobable) curve uses the same library size
        equiprob_curve = [lib_size * (1 - (1 - 1/lib_size)**i) * eff_frac for i in range(1, max_draws+1)]
        import plotly.graph_objects as go
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(range(1, max_draws+1)), y=mean_curve, mode='lines', name='Empirical Simulation', line=dict(color='#17b6a7', width=3)))
        fig4.add_trace(go.Scatter(x=list(range(1, max_draws+1)), y=equiprob_curve, mode='lines', name='Equiprobable (Theoretical)', line=dict(color='#0b3b3e', width=3, dash='dash')))
        fig4.update_layout(
            xaxis_title='Number of Samplings',
            yaxis_title='Mean Unique Variants Observed (efficiency-adjusted)',
            title='Coupon Collector Simulation: Unique Variants vs. Samplings',
            legend=dict(x=0.01, y=0.99)
        )
        # Calculate % of designs observed at 1X and 2X screening depth
        ix_1x = lib_size - 1  # 0-based index, so draws == lib_size
        ix_2x = min(2*lib_size - 1, max_draws - 1)  # cap at max_draws-1
        empirical_1x = mean_curve[ix_1x] if ix_1x < len(mean_curve) else mean_curve[-1]
        theoretical_1x = equiprob_curve[ix_1x] if ix_1x < len(equiprob_curve) else equiprob_curve[-1]
        empirical_2x = mean_curve[ix_2x] if ix_2x < len(mean_curve) else mean_curve[-1]
        theoretical_2x = equiprob_curve[ix_2x] if ix_2x < len(equiprob_curve) else equiprob_curve[-1]
        empirical_pct_1x = 100 * empirical_1x / (lib_size * eff_frac) if lib_size > 0 and eff_frac > 0 else 0
        theoretical_pct_1x = 100 * theoretical_1x / (lib_size * eff_frac) if lib_size > 0 and eff_frac > 0 else 0
        empirical_pct_2x = 100 * empirical_2x / (lib_size * eff_frac) if lib_size > 0 and eff_frac > 0 else 0
        theoretical_pct_2x = 100 * theoretical_2x / (lib_size * eff_frac) if lib_size > 0 and eff_frac > 0 else 0
        # Add annotation to plot for 1X
        fig4.add_annotation(
            x=lib_size,
            y=empirical_1x,
            text=f"{empirical_pct_1x:.1f}% at 1X (empirical)",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=60,
            font=dict(size=14, color="#17b6a7"),
            bgcolor="white"
        )
        fig4.add_annotation(
            x=lib_size,
            y=theoretical_1x,
            text=f"{theoretical_pct_1x:.1f}% at 1X (theoretical)",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=-40,
            font=dict(size=14, color="#0b3b3e"),
            bgcolor="white"
        )
        # Add annotation to plot for 2X
        fig4.add_annotation(
            x=min(2*lib_size, max_draws),
            y=empirical_2x,
            text=f"{empirical_pct_2x:.1f}% at 2X (empirical)",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=60,
            font=dict(size=14, color="#17b6a7"),
            bgcolor="white"
        )
        fig4.add_annotation(
            x=min(2*lib_size, max_draws),
            y=theoretical_2x,
            text=f"{theoretical_pct_2x:.1f}% at 2X (theoretical)",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=-40,
            font=dict(size=14, color="#0b3b3e"),
            bgcolor="white"
        )
        st.plotly_chart(fig4, use_container_width=True)
        # Add warning if empirical > theoretical at 1X
        if empirical_1x > theoretical_1x:
            st.warning(
                "The empirical curve is above the theoretical at 1X. This can occur if your empirical pool is highly uneven (a few cassettes are much more abundant), if the library size input is set larger than the number of cassettes with nonzero empirical probability, or due to simulation artifacts. Double-check your library size and cassette selection for accurate comparison."
            )
        st.markdown(
            f"""
            **At 1X screening depth (draws = {lib_size}):**
            - **{empirical_pct_1x:.1f}%** of designs observed (empirical simulation, efficiency-adjusted)
            - **{theoretical_pct_1x:.1f}%** of designs observed (theoretical expectation, efficiency-adjusted)
            
            **At 2X screening depth (draws = {min(2*lib_size, max_draws)}):**
            - **{empirical_pct_2x:.1f}%** of designs observed (empirical simulation, efficiency-adjusted)
            - **{theoretical_pct_2x:.1f}%** of designs observed (theoretical expectation, efficiency-adjusted)
            
            **Editing Efficiency:**
            All values above are scaled by the specified editing efficiency ({efficiency:.1f}%).
            
            **Sampling Source:**
            The plot is currently sampling from the empirical distribution of cassettes in the selected samples and {grouping_col} group(s), with a pseudocount of 1 for zeros. This means the simulation reflects the actual observed abundance of each cassette in your data, not a uniform distribution.
            
            **Interpretation:**
            This simulation models the process of randomly sampling cassettes from your pool, using the empirical distribution of read counts (with a pseudocount of 1 for zeros) from the selected samples and {grouping_col} group(s). The solid line shows the mean number of unique variants observed as more samplings are taken, averaged over multiple simulation runs, and scaled by the editing efficiency. The dashed line is the theoretical expectation if all cassettes were equally likely (equiprobable) for a library of size {lib_size}, also scaled by efficiency. Use this plot to guide how many samplings are needed to discover most or all variants, and to compare your pool's diversity to the ideal.
            
            **Key Considerations:**
            - The empirical simulation is influenced by the actual abundance and dropout of cassettes in your data, including technical and biological biases.
            - The theoretical curve assumes perfect equiprobability and no dropout.
            - The closer your empirical curve is to the theoretical, the more even and complete your pool is.
            - If your empirical curve is much lower, it may indicate bottlenecks, bias, or underrepresentation in your pool or screening process.
            """
        )

with tab3:
    st.header(f"Sample x {grouping_col} Heatmap")
    # hit_df: index = cassette, columns = samples
    # Need: rows = samples, columns = grouping values, values = sum of hits for all cassettes in that group for that sample
    if hit_matrix_file is not None and grouping_col in library_df.columns:
        # Map cassette to grouping column
        cassette_to_group = library_df.set_index(cassette_id_col)[grouping_col].to_dict()
        # Build a new DataFrame: rows = samples, columns = all groups from library
        sample_names = hit_df.columns.tolist()
        groups = sorted(library_df[grouping_col].dropna().unique())
        sample_group_matrix = pd.DataFrame(0, index=sample_names, columns=groups)
        for cassette, row in hit_df.iterrows():
            group = cassette_to_group.get(cassette, None)
            if group in groups:
                for sample in sample_names:
                    sample_group_matrix.at[sample, group] += row[sample]
        # Ensure all group columns are present, even if all zeros
        for group in groups:
            if group not in sample_group_matrix.columns:
                sample_group_matrix[group] = 0
        sample_group_matrix = sample_group_matrix[groups]  # Ensure column order
        fig2 = px.imshow(
            sample_group_matrix,
            labels=dict(x=f"{grouping_col}", y="Sample", color="Total Hits"),
            color_continuous_scale=navy_teal_scale,
            aspect='auto',
            title=f"Sample x {grouping_col} Total Hits Heatmap"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.header("GC Content vs Count Abundance (Raw)")
    if hit_matrix_file is not None and selected_samples:
        with st.spinner("Processing GC content analysis..."):
            # Check for GC content column in library
            gc_col = None
            possible_gc_cols = ['GC_content', 'GC%', 'GC_percent', 'gc_content', 'gc%', 'gc_percent', 'GC']
            for col in possible_gc_cols:
                if col in library_df.columns:
                    gc_col = col
                    break
            
            # If no GC column found, try to calculate from sequence columns
            if gc_col is None:
                sequence_cols = [col for col in library_df.columns if 'seq' in col.lower() or 'sequence' in col.lower()]
                if sequence_cols:
                    seq_col = st.selectbox("Select sequence column for GC calculation", sequence_cols, key="gc_raw_seq_col")
                    if seq_col in library_df.columns:
                        def calculate_gc_content(sequence):
                            if pd.isna(sequence):
                                return np.nan
                            sequence = str(sequence).upper()
                            gc_count = sequence.count('G') + sequence.count('C')
                            total_count = len([base for base in sequence if base in 'ATGC'])
                            return (gc_count / total_count * 100) if total_count > 0 else np.nan
                        
                        library_df['calculated_GC'] = library_df[seq_col].apply(calculate_gc_content)
                        gc_col = 'calculated_GC'
                        st.info(f"GC content calculated from {seq_col} column")
            
            if gc_col is not None:
                # Get total counts per cassette across selected samples
                cassette_total_counts = hit_df[selected_samples].sum(axis=1)
                # Filter for cassettes in selected groups
                filtered_cassettes_gc = [cass for cass in cassette_total_counts.index if cass in cassettes_in_group]
                cassette_total_counts = cassette_total_counts.loc[filtered_cassettes_gc]
                
                # Merge with GC content data - with error handling
                try:
                    gc_data = library_df.set_index(cassette_id_col)[gc_col].to_dict()
                except KeyError:
                    st.error(f"Column '{cassette_id_col}' not found in library data. Available columns: {list(library_df.columns)}")
                    st.stop()
                
                # Create data for plotting - VECTORIZED
                counts_df = cassette_total_counts.to_frame(name='Total_Count').reset_index()
                
                # Ensure the index name matches our expected column name
                if counts_df.columns[0] != cassette_id_col:
                    counts_df = counts_df.rename(columns={counts_df.columns[0]: cassette_id_col})
                    
                counts_df['GC_Content'] = counts_df[cassette_id_col].map(gc_data)
                
                # Filter for valid GC and >0 counts
                valid_mask = (pd.notna(counts_df['GC_Content'])) & (counts_df['Total_Count'] > 0)
                plot_df = counts_df[valid_mask].copy()
                
                if len(plot_df) > 0:
                    plot_df.rename(columns={cassette_id_col: 'Cassette'}, inplace=True)
                    
                    # Create GC content bins
                    bin_size = st.slider("GC bin size (%)", min_value=1.0, max_value=10.0, value=2.5, step=0.5, key="gc_raw_bin_size")
                    
                    # Calculate bin edges
                    min_gc = plot_df['GC_Content'].min()
                    max_gc = plot_df['GC_Content'].max()
                    bin_edges = np.arange(np.floor(min_gc / bin_size) * bin_size, 
                                        np.ceil(max_gc / bin_size) * bin_size + bin_size, 
                                        bin_size)
                    
                    # Create bins for observed data
                    plot_df['GC_Bin'] = pd.cut(plot_df['GC_Content'], bins=bin_edges, right=False)
                    binned_data = plot_df.groupby('GC_Bin', observed=True).agg({
                        'Total_Count': 'sum',
                        'Cassette': 'count'
                    }).reset_index()
                    
                    # Create bin labels for plotting
                    binned_data['GC_Bin_Label'] = binned_data['GC_Bin'].apply(
                        lambda x: f"{x.left:.1f}-{x.right:.1f}%" if pd.notna(x) else "Unknown"
                    )
                    binned_data['GC_Bin_Mid'] = binned_data['GC_Bin'].apply(
                        lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan
                    )
                    
                    # Display options
                    col1, col2 = st.columns(2)
                    with col1:
                        count_mode = st.selectbox(
                            "Count mode",
                            ["Total Counts", "Cassette Count"],
                            key="gc_raw_count_mode",
                            help="Total Counts: Sum of all reads in each GC bin\nCassette Count: Number of unique cassettes in each GC bin"
                        )
                    with col2:
                        log_scale = st.checkbox("Use log scale for y-axis", value=False, key="gc_raw_log_scale")
                    
                    # Set up plot data based on count mode
                    if count_mode == "Total Counts":
                        y_column = 'Total_Count'
                        y_title = 'Total Read Count'
                    else:  # Cassette Count
                        y_column = 'Cassette'
                        y_title = 'Number of Cassettes'
                    
                    if log_scale:
                        binned_data['Plot_Count'] = np.log10(binned_data[y_column] + 1)
                        y_axis_title = f'Log10({y_title} + 1)'
                    else:
                        binned_data['Plot_Count'] = binned_data[y_column]
                        y_axis_title = y_title
                    
                    # Create histogram
                    hover_data_dict = {
                        'GC_Bin_Label': True,
                        'Total_Count': True,
                        'Cassette': ':.0f',
                        'GC_Bin_Mid': False,
                        'Plot_Count': False
                    }
                    
                    fig4 = px.bar(
                        binned_data,
                        x='GC_Bin_Mid',
                        y='Plot_Count',
                        hover_data=hover_data_dict,
                        labels={
                            'GC_Bin_Mid': 'GC Content (%)',
                            'Plot_Count': y_axis_title,
                            'Cassette': 'Cassettes in bin',
                            'Total_Count': 'Total reads in bin'
                        },
                        title=f'Raw GC Content Distribution - {count_mode} (Selected {grouping_col} Groups)',
                        color_discrete_sequence=['#17b6a7']
                    )
                    
                    # Update bar width to match bin size
                    fig4.update_traces(width=bin_size * 0.8)
                    
                    fig4.update_layout(
                        xaxis_title='GC Content (%)',
                        yaxis_title=y_axis_title,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total cassettes", f"{len(plot_df):,}")
                    with col2:
                        st.metric("Mean GC%", f"{plot_df['GC_Content'].mean():.1f}")
                    with col3:
                        st.metric("GC% range", f"{plot_df['GC_Content'].min():.1f} - {plot_df['GC_Content'].max():.1f}")
                        
                    st.markdown(f"""
                    **Interpretation:**
                    This histogram shows the raw distribution of observed counts vs GC content for cassettes in the selected {grouping_col} group(s). 
                    
                    - **X-axis (GC Content):** Percentage of guanine (G) and cytosine (C) bases in the cassette sequence
                    - **Y-axis ({count_mode}):** {'Total read counts' if count_mode == 'Total Counts' else 'Number of unique cassettes'} in each GC bin
                    - **Bin size:** {bin_size}% GC content per bin
                    
                    **What to look for:**
                    - **Even distribution:** Suggests minimal GC bias in sequencing/amplification
                    - **Peaks at specific GC ranges:** May indicate GC-dependent amplification bias
                    - **Low counts at extremes:** Very high (>60%) or very low (<35%) GC content often show reduced efficiency
                    
                    **Technical considerations:**
                    - This shows the raw observed data without normalization to the input library composition
                    - Compare with the normalized view to distinguish between library design effects and technical bias
                    - PCR amplification efficiency typically decreases at GC extremes
                    """)
                else:
                    st.warning("No cassettes with valid GC content and count data found for plotting.")
            else:
                st.warning("No GC content column found in library data. To add GC content analysis, include a column named 'GC_content', 'GC%', or similar, or provide a sequence column for automatic calculation.")

with tab5:
    st.header("GC Content vs Count Abundance (Normalized)")
    if hit_matrix_file is not None and selected_samples:
        with st.spinner("Processing normalized GC content analysis..."):
            # Check for GC content column in library
            gc_col = None
            possible_gc_cols = ['GC_content', 'GC%', 'GC_percent', 'gc_content', 'gc%', 'gc_percent', 'GC']
            for col in possible_gc_cols:
                if col in library_df.columns:
                    gc_col = col
                    break
            
            # If no GC column found, try to calculate from sequence columns
            if gc_col is None:
                sequence_cols = [col for col in library_df.columns if 'seq' in col.lower() or 'sequence' in col.lower()]
                if sequence_cols:
                    seq_col = st.selectbox("Select sequence column for GC calculation", sequence_cols, key="gc_norm_seq_col")
                    if seq_col in library_df.columns:
                        def calculate_gc_content(sequence):
                            if pd.isna(sequence):
                                return np.nan
                            sequence = str(sequence).upper()
                            gc_count = sequence.count('G') + sequence.count('C')
                            total_count = len([base for base in sequence if base in 'ATGC'])
                            return (gc_count / total_count * 100) if total_count > 0 else np.nan
                        
                        library_df['calculated_GC'] = library_df[seq_col].apply(calculate_gc_content)
                        gc_col = 'calculated_GC'
                        st.info(f"GC content calculated from {seq_col} column")
            
            if gc_col is not None:
                # Get total counts per cassette across selected samples
                cassette_total_counts = hit_df[selected_samples].sum(axis=1)
                # Filter for cassettes in selected groups
                filtered_cassettes_gc = [cass for cass in cassette_total_counts.index if cass in cassettes_in_group]
                cassette_total_counts = cassette_total_counts.loc[filtered_cassettes_gc]
                
                # Merge with GC content data - with error handling
                try:
                    gc_data = library_df.set_index(cassette_id_col)[gc_col].to_dict()
                except KeyError:
                    st.error(f"Column '{cassette_id_col}' not found in library data. Available columns: {list(library_df.columns)}")
                    st.stop()
                
                # Create data for plotting - VECTORIZED
                counts_df = cassette_total_counts.to_frame(name='Total_Count').reset_index()
                
                # Ensure the index name matches our expected column name
                if counts_df.columns[0] != cassette_id_col:
                    counts_df = counts_df.rename(columns={counts_df.columns[0]: cassette_id_col})
                    
                counts_df['GC_Content'] = counts_df[cassette_id_col].map(gc_data)
                
                # Filter for valid GC and >0 counts
                valid_mask = (pd.notna(counts_df['GC_Content'])) & (counts_df['Total_Count'] > 0)
                plot_df = counts_df[valid_mask].copy()
                
                if len(plot_df) > 0:
                    plot_df.rename(columns={cassette_id_col: 'Cassette'}, inplace=True)
                    
                    # Create GC content bins
                    bin_size = st.slider("GC bin size (%)", min_value=1.0, max_value=10.0, value=2.5, step=0.5, key="gc_norm_bin_size")
                    
                    # Calculate bin edges
                    min_gc = plot_df['GC_Content'].min()
                    max_gc = plot_df['GC_Content'].max()
                    bin_edges = np.arange(np.floor(min_gc / bin_size) * bin_size, 
                                        np.ceil(max_gc / bin_size) * bin_size + bin_size, 
                                        bin_size)
                    
                    # Create bins for observed data
                    plot_df['GC_Bin'] = pd.cut(plot_df['GC_Content'], bins=bin_edges, right=False)
                    observed_binned = plot_df.groupby('GC_Bin', observed=True).agg({
                        'Total_Count': 'sum',
                        'Cassette': 'count'
                    }).reset_index()
                    
                    # Create bins for ALL cassettes in the library (expected distribution) - VECTORIZED
                    # Get GC content for all cassettes in selected groups
                    all_cassettes_in_group = library_df[library_df[grouping_col].isin(selected_groups_analysis)]
                    
                    # Vectorized approach - no loops
                    valid_gc_mask = pd.notna(all_cassettes_in_group[gc_col])
                    all_gc_df = all_cassettes_in_group[valid_gc_mask][[cassette_id_col, gc_col]].copy()
                    all_gc_df.rename(columns={cassette_id_col: 'Cassette', gc_col: 'GC_Content'}, inplace=True)
                    
                    if len(all_gc_df) > 0:
                        all_gc_df['GC_Bin'] = pd.cut(all_gc_df['GC_Content'], bins=bin_edges, right=False)
                        expected_binned = all_gc_df.groupby('GC_Bin', observed=True).size().reset_index(name='Expected_Cassettes')
                        
                        # Ensure both dataframes have the same categorical bins
                        all_bins = list(set(observed_binned['GC_Bin'].cat.categories) | set(expected_binned['GC_Bin'].cat.categories))
                        observed_binned['GC_Bin'] = observed_binned['GC_Bin'].cat.add_categories([cat for cat in all_bins if cat not in observed_binned['GC_Bin'].cat.categories])
                        expected_binned['GC_Bin'] = expected_binned['GC_Bin'].cat.add_categories([cat for cat in all_bins if cat not in expected_binned['GC_Bin'].cat.categories])
                        
                        # Merge observed and expected
                        binned_data = pd.merge(observed_binned, expected_binned, on='GC_Bin', how='outer')
                        
                        # Fill NaN values appropriately (avoiding categorical issues)
                        binned_data['Total_Count'] = binned_data['Total_Count'].fillna(0).astype(float)
                        binned_data['Cassette'] = binned_data['Cassette'].fillna(0).astype(float)
                        binned_data['Expected_Cassettes'] = binned_data['Expected_Cassettes'].fillna(0).astype(float)
                        
                        # Calculate normalization metrics
                        total_observed_reads = binned_data['Total_Count'].sum()
                        total_expected_cassettes = binned_data['Expected_Cassettes'].sum()
                        
                        # Calculate observed and expected frequencies
                        binned_data['Observed_Freq'] = binned_data['Total_Count'] / total_observed_reads
                        binned_data['Expected_Freq'] = binned_data['Expected_Cassettes'] / total_expected_cassettes
                        
                        # Calculate normalized ratio (observed/expected)
                        binned_data['Normalization_Ratio'] = np.where(
                            binned_data['Expected_Freq'] > 0,
                            binned_data['Observed_Freq'] / binned_data['Expected_Freq'],
                            np.nan
                        )
                        
                        # Calculate reads per cassette
                        binned_data['Reads_Per_Cassette'] = np.where(
                            binned_data['Expected_Cassettes'] > 0,
                            binned_data['Total_Count'] / binned_data['Expected_Cassettes'],
                            0
                        )
                    else:
                        # Fallback if no expected data available
                        binned_data = observed_binned.copy()
                        binned_data['Expected_Cassettes'] = binned_data['Cassette']
                        binned_data['Normalization_Ratio'] = 1.0
                        binned_data['Reads_Per_Cassette'] = binned_data['Total_Count'] / binned_data['Cassette']
                    
                    # Create bin labels for plotting
                    binned_data['GC_Bin_Label'] = binned_data['GC_Bin'].apply(
                        lambda x: f"{x.left:.1f}-{x.right:.1f}%" if pd.notna(x) else "Unknown"
                    )
                    binned_data['GC_Bin_Mid'] = binned_data['GC_Bin'].apply(
                        lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan
                    )
                    
                    # Normalization options
                    col1, col2 = st.columns(2)
                    with col1:
                        normalization_mode = st.selectbox(
                            "Display mode",
                            ["Normalized Ratio", "Reads per Cassette"],
                            key="gc_norm_mode",
                            help="Normalized Ratio: (Observed frequency / Expected frequency)\nReads per Cassette: Average reads per cassette in each bin"
                        )
                    with col2:
                        log_scale = st.checkbox("Use log scale for y-axis", value=False, key="gc_norm_log_scale")
                    
                    # Set up plot data based on normalization mode
                    if normalization_mode == "Normalized Ratio":
                        plot_column = 'Normalization_Ratio' 
                        base_y_title = 'Observed/Expected Ratio'
                    else:  # Reads per Cassette
                        plot_column = 'Reads_Per_Cassette'
                        base_y_title = 'Reads per Cassette'
                    
                    if log_scale and normalization_mode != "Normalized Ratio":
                        binned_data['Plot_Count'] = np.log10(binned_data[plot_column] + 1)
                        y_axis_title = f'Log10({base_y_title} + 1)'
                    elif normalization_mode == "Normalized Ratio":
                        binned_data['Plot_Count'] = binned_data[plot_column]
                        y_axis_title = base_y_title
                        if log_scale:
                            st.info("Log scale not applied to normalized ratios (ratios can be < 1)")
                    else:
                        binned_data['Plot_Count'] = binned_data[plot_column]
                        y_axis_title = base_y_title
                    
                    # Create histogram
                    hover_data_dict = {
                        'GC_Bin_Label': True,
                        'Total_Count': True,
                        'Expected_Cassettes': ':.0f',
                        'Normalization_Ratio': ':.2f',
                        'Reads_Per_Cassette': ':.1f',
                        'GC_Bin_Mid': False,
                        'Plot_Count': False
                    }
                    
                    # Add observed cassettes if different from expected
                    if 'Cassette' in binned_data.columns:
                        hover_data_dict['Cassette'] = ':.0f'
                    
                    fig5 = px.bar(
                        binned_data,
                        x='GC_Bin_Mid',
                        y='Plot_Count',
                        hover_data=hover_data_dict,
                        labels={
                            'GC_Bin_Mid': 'GC Content (%)',
                            'Plot_Count': y_axis_title,
                            'Cassette': 'Observed cassettes',
                            'Expected_Cassettes': 'Expected cassettes',
                            'Total_Count': 'Total reads',
                            'Normalization_Ratio': 'Obs/Exp ratio',
                            'Reads_Per_Cassette': 'Reads/cassette'
                        },
                        title=f'Normalized GC Content Distribution - {normalization_mode} (Selected {grouping_col} Groups)',
                        color_discrete_sequence=['#17b6a7']
                    )
                    
                    # Update bar width to match bin size
                    fig5.update_traces(width=bin_size * 0.8)
                    
                    # Add reference line for normalized ratio
                    if normalization_mode == "Normalized Ratio":
                        fig5.add_hline(
                            y=1.0, 
                            line_dash="dash", 
                            line_color="#0b3b3e",
                            annotation_text="Expected (ratio = 1.0)",
                            annotation_position="top right"
                        )
                    
                    fig5.update_layout(
                        xaxis_title='GC Content (%)',
                        yaxis_title=y_axis_title,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total cassettes", f"{len(plot_df):,}")
                    with col2:
                        if normalization_mode == "Normalized Ratio":
                            mean_ratio = binned_data['Normalization_Ratio'].mean()
                            st.metric("Mean norm. ratio", f"{mean_ratio:.2f}")
                        else:
                            mean_reads = binned_data['Reads_Per_Cassette'].mean()
                            st.metric("Mean reads/cassette", f"{mean_reads:.1f}")
                    with col3:
                        st.metric("GC% range", f"{plot_df['GC_Content'].min():.1f} - {plot_df['GC_Content'].max():.1f}")
                        
                    st.markdown(f"""
                    **Interpretation:**
                    This histogram shows the GC content distribution normalized to the expected distribution in the input library for cassettes in the selected {grouping_col} group(s). 
                    
                    - **X-axis (GC Content):** Percentage of guanine (G) and cytosine (C) bases in the cassette sequence
                    - **Y-axis ({normalization_mode}):** {'Ratio of observed to expected frequency' if normalization_mode == 'Normalized Ratio' else 'Average reads per cassette in each GC bin'}
                    - **Normalization:** Accounts for the GC distribution in the starting library design
                    
                    **What to look for:**
                    - **Ratio = 1.0 (dashed line):** Perfect match to expected library composition
                    - **Ratio > 1.0:** Over-representation (GC bias favoring these ranges)
                    - **Ratio < 1.0:** Under-representation (GC bias against these ranges)
                    - **Flat distribution around 1.0:** Minimal GC bias
                    
                    **Technical considerations:**
                    - This normalization removes the effect of library design GC distribution
                    - Deviations from 1.0 indicate technical bias in amplification, sequencing, or other steps
                    - Compare with the raw view to distinguish design effects from technical bias
                    - Expected values are calculated from all cassettes in the selected {grouping_col} groups
                    """)
                else:
                    st.warning("No cassettes with valid GC content and count data found for plotting.")
            else:
                st.warning("No GC content column found in library data. To add GC content analysis, include a column named 'GC_content', 'GC%', or similar, or provide a sequence column for automatic calculation.") 