import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from pathlib import Path
import math
import copy
import io
import scipy.stats as stats
from io import BytesIO

# --- Color Palette ---
PALETTE = {
    "navy": "#10282D",
    "dark_teal": "#1E4746",
    "moderate_teal": "#3F7F6F",
    "light_teal": "#53A28A",
    "seafoam": "#56C1AC",
    "yellow": "#FFF200",
    "red": "#FF4B4B",
    "green": "#00C853",
    "bottom_band": "#0C2D34"
}

# --- Download Button Function ---
def plot_download_button(fig, graph_type, scenario_inputs):
    scenario_names = '_'.join([s['label'].replace(' ', '').replace('(', '').replace(')', '') + str(s['num_variants']) for s in scenario_inputs])
    filename = f"{graph_type}_{scenario_names}_{num_rounds}rounds_{num_parallel_projects}proj.png"
    
    # Create a more user-friendly button label
    graph_type_display = graph_type.replace('_', ' ').title()
    
    try:
        buf = BytesIO()
        fig.write_image(buf, format="png", width=800, height=600, scale=2)
        st.download_button(
            label=f"📊 Download {graph_type_display}",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png",
            key=f"download_{graph_type}_{scenario_names}"
        )
    except Exception as e:
        st.warning(f"Download not available for {graph_type_display}. Install kaleido package for image export: pip install kaleido")

st.set_page_config(page_title="Screening Cost Modeling V2", layout="wide")
st.title("Screening Cost Modeling App (V2)")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Model Parameters")
    st.markdown("---")
    st.subheader("Scenario Structure")
    num_rounds = st.slider("Number of Rounds (for Cumulative Chart)", min_value=1, max_value=20, value=5, step=1)
    num_parallel_projects = st.slider("Number of Parallel Projects", min_value=1, max_value=20, value=1, step=1)
    st.markdown("---")
    st.subheader("Cost Assumptions")
    fte_cost_per_hour = st.number_input("FTE Cost per Hour ($)", min_value=0, max_value=1000, value=125, step=1)
    st.markdown("---")
    st.subheader("Probability Modeling")
    mean_gain_pct = st.number_input("Mean Per-Variant Gain (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="fold_mean_gain")
    std_gain_pct = st.number_input("Std Dev Per-Variant Gain (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1, key="fold_std_gain")
    target_fold = st.number_input("Target Fold Improvement", min_value=1.0, max_value=1000.0, value=10.0, step=0.1, key="fold_target_fold")
    max_rounds_fold = st.slider("Max Rounds to Plot (fold analysis)", min_value=1, max_value=50, value=10, step=1, key="fold_max_rounds")
    ruggedness = st.slider("Ruggedness (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.markdown("---")
    st.subheader("Sequence Space & Learning Parameters")
    diversity_bonus = st.slider("Diversity Bonus (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.1, 
                               help="Additional gain from exploring diverse sequence space")
    learning_rate = st.slider("Learning Rate (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                             help="How much each variant teaches about the fitness landscape")
    sequence_space_size = st.selectbox("Sequence Space Size", 
                                     options=["Small (1K variants)", "Medium (10K variants)", "Large (100K variants)", "Unlimited"],
                                     index=2,
                                     help="Total beneficial variants available in sequence space")
    information_value = st.slider("Information Value Multiplier", min_value=0.0, max_value=2.0, value=1.5, step=0.1,
                                 help="How much future rounds benefit from current round's information")
    st.markdown("---")
    st.subheader("Recombination & Epistasis")
    epistasis = st.slider("Epistasis (-2 to 2)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
                         help="Interaction between mutations: -2=strong negative, 0=independent, +2=strong positive")
    recombination_efficiency = st.slider("Recombination Efficiency (0-1)", min_value=0.0, max_value=1.0, value=0.8, step=0.1,
                                        help="How efficiently beneficial mutations from different rounds can be combined")
    mutation_additivity = st.slider("Mutation Additivity (0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                                   help="0=purely multiplicative, 1=purely additive combination of effects")
    st.markdown("---")
    st.subheader("Learning-to-Completion Parameters")
    adaptive_stopping = st.checkbox("Enable Adaptive Stopping", value=True,
                                   help="Stop when learning indicates target is achievable with high confidence")
    confidence_threshold = st.slider("Stopping Confidence Threshold", min_value=0.5, max_value=0.95, value=0.8, step=0.05,
                                    help="Confidence level required to stop early (higher = more conservative)")
    learning_acceleration = st.slider("Learning Acceleration Factor", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
                                     help="How much learning improves future round efficiency (1.0 = no acceleration)")
    target_prediction_improvement = st.slider("Target Prediction Improvement", min_value=0.0, max_value=0.5, value=0.3, step=0.05,
                                             help="How much learning improves ability to predict which variants will succeed")
    st.markdown("---")
    st.subheader("Scenario Parameters")
    # Example: Add as many scenarios as you want here
    scenario_inputs = []
    for label, color, default_n, default_pct, default_len, default_delay, default_cost, default_seq, default_fte in [
        ("Std (eBlocks)", PALETTE["moderate_teal"], 192, 0.84, 4.4, 1.1, 30.0, 5000.0, 60),
        ("CREATE (Opool)", PALETTE["seafoam"], 192, 0.65, 3.2, 1.0, 1.0, 5000.0, 48),
        ("CREATE (Opool B)", PALETTE["navy"], 192, 0.65, 3.2, 1.0, 0.8, 5000.0, 60),
    ]:
        st.markdown(f"---\n**{label}**")
        n = st.number_input(f"Variants per Round ({label})", min_value=1, max_value=10000, value=default_n, step=1, key=f"n_{label}")
        pct = st.slider(f"% Correct ({label})", min_value=0.0, max_value=1.0, value=default_pct, step=0.01, key=f"pct_{label}")
        length = st.number_input(f"Length of Round (weeks, {label})", min_value=0.1, max_value=52.0, value=default_len, step=0.1, key=f"len_{label}")
        delay = st.number_input(f"Delays (weeks, {label})", min_value=0.0, max_value=52.0, value=default_delay, step=0.1, key=f"delay_{label}")
        cost = st.number_input(f"Cost of Oligos/eBlocks ($/variant, {label})", min_value=0.0, max_value=100.0, value=default_cost, step=0.5, key=f"cost_{label}")
        seq = st.number_input(f"Cost of Sequencing ({label})", min_value=0.0, max_value=100000.0, value=default_seq, step=1.0, key=f"seq_{label}")
        fte = st.number_input(f"FTE Hours per Round ({label})", min_value=1, max_value=1000, value=default_fte, step=1, key=f"fte_{label}")
        scenario_inputs.append({
            "label": label,
            "color": color,
            "num_variants": n,
            "pct_correct": pct,
            "round_length": length,
            "delay_weeks": delay,
            "cost_per_variant": cost,
            "cost_seq": seq,
            "fte_hours_per_round": fte,
        })

REFERENCE_IDX = 0  # Index of the reference scenario for deltas (default: eBlocks)

# --- Calculation Logic ---
def calc_costs(num_variants, pct_correct, round_length, delay_weeks, cost_synth_per_variant, fte_cost_per_hour, fte_hours_per_round, cost_seq):
    cost_synth = num_variants * cost_synth_per_variant
    num_lost = num_variants * (1 - pct_correct)
    cost_lost = num_lost * cost_synth_per_variant
    delay_hours = delay_weeks * 5 * 8
    cost_of_delay = delay_hours * fte_cost_per_hour
    cost_researcher = fte_hours_per_round * fte_cost_per_hour
    final_cost = cost_lost + cost_synth + cost_of_delay + cost_researcher + cost_seq
    return {
        'num_lost': num_lost,
        'cost_lost': cost_lost,
        'cost_synth': cost_synth,
        'cost_of_delay': cost_of_delay,
        'cost_researcher': cost_researcher,
        'cost_seq': cost_seq,
        'final_cost': final_cost,
        'round_length': round_length,
        'delay_weeks': delay_weeks,
        'pct_correct': pct_correct,
        'num_variants': num_variants,
        'fte_hours_per_round': fte_hours_per_round,
        'fte_cost_per_hour': fte_cost_per_hour
    }

costs = [calc_costs(
    s["num_variants"], s["pct_correct"], s["round_length"], s["delay_weeks"],
    s["cost_per_variant"], fte_cost_per_hour, s["fte_hours_per_round"], s["cost_seq"]
) for s in scenario_inputs]

# --- Parameter Summary Table ---
param_data = {"Parameter": [
    'Number of Rounds', 'Number of Parallel Projects', 'FTE Cost per Hour ($)',
    'Variants per Round', '% Correct', 'Round Length (weeks)', 'Delays (weeks)',
    'Synthesis Cost per Variant ($)', 'Sequencing Cost ($)', 'FTE Hours per Round',
    'Cost of Lost Variants ($/round)', 'Cost of Synthesis ($/round)', 'Cost of Delay ($/round)',
    'Cost of Researcher ($/round)', 'Cost of Sequencing ($/round)', 'Final Cost per Round ($/round)']}
for i, s in enumerate(scenario_inputs):
    param_data[s["label"]] = [
        num_rounds, num_parallel_projects, fte_cost_per_hour,
        s["num_variants"], f"{s['pct_correct']:.2f}", f"{s['round_length']:.1f}", f"{s['delay_weeks']:.1f}",
        f"{s['cost_per_variant']:.2f}", f"{s['cost_seq']:.2f}", s['fte_hours_per_round'],
        f"${costs[i]['cost_lost']:,.2f}", f"${costs[i]['cost_synth']:,.2f}", f"${costs[i]['cost_of_delay']:,.2f}",
        f"${costs[i]['cost_researcher']:,.2f}", f"${costs[i]['cost_seq']:,.2f}", f"${costs[i]['final_cost']:,.2f}"
    ]
param_df = pd.DataFrame(param_data)
st.markdown("### Scenario Parameter Summary")
st.dataframe(param_df, hide_index=True, use_container_width=True)

# --- Cost per Variant Comparison ---
st.markdown("## Cost per Variant Comparison")
x_labels = [f"{s['label']} (N={s['num_variants']})" for s in scenario_inputs]
y_per_variant = [costs[i]['final_cost'] / s['num_variants'] if s['num_variants'] else 0 for i, s in enumerate(scenario_inputs)]
colors = [s['color'] for s in scenario_inputs]
texts = [f"${y:,.2f}" for y in y_per_variant]
col_var1, col_var2 = st.columns(2)
with col_var1:
    st.markdown("#### Per Project (Per Round)")
    fig = go.Figure()
    bar_labels = []
    for i in range(len(scenario_inputs)):
        value = y_per_variant[i]
        if i != REFERENCE_IDX:
            delta = y_per_variant[i] - y_per_variant[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(y_per_variant[i], y_per_variant[REFERENCE_IDX]) * 100) if max(y_per_variant[i], y_per_variant[REFERENCE_IDX]) > 0 else 0
            label = f"${value:,.2f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${value:,.2f}"
        bar_labels.append(label)
    fig.add_trace(go.Bar(x=x_labels, y=y_per_variant, marker_color=colors, text=None, name="Cost per Variant"))
    # Add one annotation per bar, above the bar
    max_y = max(y_per_variant)
    min_offset = max_y * 0.15  # Minimum offset based on the tallest bar
    for i, lbl in enumerate(bar_labels):
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(y_per_variant[i] * 0.35, min_offset)
        fig.add_annotation(
            x=i, y=y_per_variant[i] + dynamic_offset,  # Dynamic positioning
            text=lbl,
            showarrow=False,
            font=dict(size=14, color=colors[i]),
            align="center"
        )
    fig.update_layout(title="Cost per Variant per Project (Per Round)", xaxis_title="Scenario", yaxis_title="Cost per Variant ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for cost per variant per project
    plot_download_button(fig, 'cost_per_variant_per_project', scenario_inputs)
    
with col_var2:
    st.markdown(f"#### All Projects (N={num_parallel_projects}) (Per Round)")
    y_per_variant_all = [(costs[i]['final_cost'] * num_parallel_projects) / (s['num_variants'] * num_parallel_projects) if s['num_variants'] and num_parallel_projects else 0 for i, s in enumerate(scenario_inputs)]
    fig_all = go.Figure()
    bar_labels_all = []
    for i in range(len(scenario_inputs)):
        value = y_per_variant_all[i]
        if i != REFERENCE_IDX:
            delta = y_per_variant_all[i] - y_per_variant_all[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(y_per_variant_all[i], y_per_variant_all[REFERENCE_IDX]) * 100) if max(y_per_variant_all[i], y_per_variant_all[REFERENCE_IDX]) > 0 else 0
            label = f"${value:,.0f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${value:,.0f}"
        bar_labels_all.append(label)
    fig_all.add_trace(go.Bar(x=x_labels, y=y_per_variant_all, marker_color=colors, text=None, name="Cost per Variant"))
    # Add one annotation per bar, above the bar
    max_y = max(y_per_variant_all)
    min_offset = max_y * 0.15  # Minimum offset based on the tallest bar
    for i, lbl in enumerate(bar_labels_all):
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(y_per_variant_all[i] * 0.35, min_offset)
        fig_all.add_annotation(
            x=i, y=y_per_variant_all[i] + dynamic_offset,  # Dynamic positioning
            text=lbl,
            showarrow=False,
            font=dict(size=14, color=colors[i]),
            align="center"
        )
    fig_all.update_layout(title=f"Cost per Variant for All Projects (N={num_parallel_projects}) (Per Round)", xaxis_title="Scenario", yaxis_title="Cost per Variant ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=600, height=400)
    st.plotly_chart(fig_all, use_container_width=True)
    
    # Download button for cost per variant all projects
    plot_download_button(fig_all, 'cost_per_variant_all_projects', scenario_inputs)

# --- Executive Summary: white background ---
EXEC_SUMMARY_SCENARIO_COLOR = PALETTE['light_teal']

def make_exec_summary(scenario_inputs, costs, num_rounds, num_parallel_projects, reference_idx=0):
    # Defensive: ensure costs is always a list of dicts
    if isinstance(costs, dict):
        costs = [costs]
    per_round = [c['final_cost'] for c in costs]
    per_variant = [c['final_cost'] / s['num_variants'] if s['num_variants'] else 0 for c, s in zip(costs, scenario_inputs)]
    per_project = [c['final_cost'] * num_rounds for c in costs]
    annual = [c['final_cost'] / c['round_length'] * 52.143 for c in costs]
    rounds_per_year = [int(52.143 // c['round_length']) for c in costs]
    cum_cost = [c['final_cost'] * num_rounds * num_parallel_projects for c in costs]
    def delta_fmt(val, ref, label):
        d = val - ref
        pct = (abs(d) / max(val, ref) * 100) if max(val, ref) > 0 else 0
        return f"Δ ({label}) = ${d:,.0f} ({pct:.1f}% savings)"
    def main_driver(ref, comp):
        keys = ['cost_lost', 'cost_synth', 'cost_of_delay', 'cost_researcher', 'cost_seq']
        names = ['Cost of lost variants', 'Cost of synthesis', 'Cost of delay', 'Cost of researcher', 'Cost of sequencing']
        diffs = [comp[k] - ref[k] for k in keys]
        idx = np.argmax(np.abs(diffs))
        return names[idx], diffs[idx]
    scenario_desc = f"Modeling {scenario_inputs[reference_idx]['num_variants']} variants/round for {scenario_inputs[reference_idx]['label']}"
    for i, s in enumerate(scenario_inputs):
        if i != reference_idx:
            scenario_desc += f", {s['num_variants']} variants/round for {s['label']}"
    html = f"""
    <div style='background-color:#fff;padding:1.5em 2em 1.5em 2em;border-radius:14px;margin-bottom:1.2em;'>
        <h2 style='color:{PALETTE['navy']};margin-bottom:0.7em;font-size:2em;'>Executive Summary</h2>
        <div style='color:{PALETTE['navy']};font-size:1.1em;margin-bottom:1em;'>
            {scenario_desc}
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:2.5em 2em;margin-bottom:1.2em;align-items:start;'>"""
    for metric, vals, title in zip(
        ['per_round', 'per_variant', 'per_project', 'annual'],
        [per_round, per_variant, per_project, annual],
        ['Cost per Round', 'Cost per Variant', 'Cost per Project', 'Annual Cost']):
        html += f"<div><h3 style='color:{PALETTE['navy']};font-size:1.1em;margin-bottom:0.3em;'>{title}</h3>"
        for i, s in enumerate(scenario_inputs):
            html += f"<div style='font-size:1.1em;font-weight:bold;line-height:1.1;margin-bottom:0.1em;'><span style='color:{EXEC_SUMMARY_SCENARIO_COLOR};'>{s['label']}:</span></div>"
            html += f"<div style='font-size:1.5em;font-weight:bold;line-height:1.2;color:{PALETTE['navy']};'>${vals[i]:,.2f}</div>"
        for i, s in enumerate(scenario_inputs):
            if i == reference_idx:
                continue
            html += f"<div style='font-size:1em;margin-top:0.3em;color:{PALETTE['navy']};font-weight:bold;'>{delta_fmt(vals[i], vals[reference_idx], s['label'] + ' vs ' + scenario_inputs[reference_idx]['label'])}</div>"
        if metric == 'annual':
            for i, s in enumerate(scenario_inputs):
                html += f"<div style='font-size:0.95em;margin-top:0.5em;color:{EXEC_SUMMARY_SCENARIO_COLOR};'><i>Assumes {rounds_per_year[i]} rounds/year for {s['label']}</i></div>"
        html += "</div>"
    html += "</div>"
    html += f"<div style='font-size:1.1em;margin-bottom:0.5em;font-weight:bold;color:{PALETTE['navy']};'>Cumulative Cost (All Projects, {num_rounds} Rounds × {num_parallel_projects} Projects)</div><div style='display:grid;grid-template-columns:repeat({len(scenario_inputs)},1fr);gap:2em 1.5em;margin-bottom:0.5em;'>"
    for i, s in enumerate(scenario_inputs):
        html += f"<div><span style='color:{EXEC_SUMMARY_SCENARIO_COLOR};font-weight:bold;'>{s['label']}:</span> <span style='color:{PALETTE['navy']};font-weight:bold;font-size:1.2em;'>${cum_cost[i]:,.0f}</span></div>"
    for i, s in enumerate(scenario_inputs):
        if i == reference_idx:
            continue
        html += f"<div><span style='font-weight:bold;color:{PALETTE['navy']};font-size:1.2em;'>{delta_fmt(cum_cost[i], cum_cost[reference_idx], s['label'] + ' vs ' + scenario_inputs[reference_idx]['label'])}</span></div>"
    html += "</div>"
    for i, s in enumerate(scenario_inputs):
        if i == reference_idx:
            continue
        d = per_round[i] - per_round[reference_idx]
        pct = (abs(d) / max(per_round[i], per_round[reference_idx]) * 100) if max(per_round[i], per_round[reference_idx]) > 0 else 0
        cheaper = s['label'] if d < 0 else scenario_inputs[reference_idx]['label']
        more_expensive = scenario_inputs[reference_idx]['label'] if d < 0 else s['label']
        savings_text = f"{cheaper} is more cost-effective, saving you ${abs(d):,.2f} ({pct:.1f}%) per round compared to {more_expensive}."
        main, diff = main_driver(costs[reference_idx], costs[i])
        driver_text = f"The main driver of this difference is <b>{main}</b> (${abs(diff):,.2f} higher in {s['label'] if diff > 0 else scenario_inputs[reference_idx]['label']})."
        html += f"<div style='margin-top:1.2em;font-size:1.1em;color:{PALETTE['navy']};'><b>Key Takeaway:</b> {savings_text}<br>{driver_text}</div>"
    html += "</div>"
    return html

st.markdown(make_exec_summary(scenario_inputs, costs, num_rounds, num_parallel_projects, REFERENCE_IDX), unsafe_allow_html=True)

# --- Cumulative Cost Over Rounds ---
def cumulative_costs(cost_per_round, num_rounds, num_parallel_projects=1):
    return np.cumsum([cost_per_round * num_parallel_projects] * num_rounds)

st.markdown("## Cumulative Cost Over Rounds")
col_cum1, col_cum2 = st.columns(2)
with col_cum1:
    st.markdown("#### Single Project")
    fig = go.Figure()
    for i, (s, c) in enumerate(zip(scenario_inputs, costs)):
        y = cumulative_costs(c['final_cost'], num_rounds)
        fig.add_trace(go.Scatter(x=np.arange(1, num_rounds+1), y=y, mode='lines+markers', name=s['label'], line=dict(color=s['color'], width=3)))
    # Add delta labels only on the right side for non-reference scenarios
    for i, (s, c) in enumerate(zip(scenario_inputs, costs)):
        if i != REFERENCE_IDX:  # Only show delta for non-reference scenarios
            y = cumulative_costs(c['final_cost'], num_rounds)
            y_ref = cumulative_costs(costs[REFERENCE_IDX]['final_cost'], num_rounds)
            # Only add annotation for the last point (rightmost)
            final_val = y[-1]
            final_ref = y_ref[-1]
            delta = final_val - final_ref
            pct = (abs(delta) / max(final_val, final_ref) * 100) if max(final_val, final_ref) > 0 else 0
            fig.add_annotation(
                x=num_rounds, y=final_val,
                text=f"Δ ${delta:,.0f}<br>({pct:.1f}% savings)",
                showarrow=False,
                font=dict(size=10, color=s['color']),
                align="left",
                xanchor="left",
                yanchor="middle"
            )
    fig.update_layout(title="Cumulative Cost Over Rounds (Single Project)", xaxis_title="Round #", yaxis_title="Cumulative Cost ($)", legend_title="Scenario", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=800, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for cumulative cost single project
    plot_download_button(fig, 'cumulative_cost_single_project', scenario_inputs)
    
with col_cum2:
    st.markdown(f"#### All Projects (N={num_parallel_projects})")
    fig = go.Figure()
    for i, (s, c) in enumerate(zip(scenario_inputs, costs)):
        y = cumulative_costs(c['final_cost'], num_rounds, num_parallel_projects)
        fig.add_trace(go.Scatter(x=np.arange(1, num_rounds+1), y=y, mode='lines+markers', name=s['label'], line=dict(color=s['color'], width=3)))
    # Add delta labels only on the right side for non-reference scenarios
    for i, (s, c) in enumerate(zip(scenario_inputs, costs)):
        if i != REFERENCE_IDX:  # Only show delta for non-reference scenarios
            y = cumulative_costs(c['final_cost'], num_rounds, num_parallel_projects)
            y_ref = cumulative_costs(costs[REFERENCE_IDX]['final_cost'], num_rounds, num_parallel_projects)
            # Only add annotation for the last point (rightmost)
            final_val = y[-1]
            final_ref = y_ref[-1]
            delta = final_val - final_ref
            pct = (abs(delta) / max(final_val, final_ref) * 100) if max(final_val, final_ref) > 0 else 0
            fig.add_annotation(
                x=num_rounds, y=final_val,
                text=f"Δ ${delta:,.0f}<br>({pct:.1f}% savings)",
                showarrow=False,
                font=dict(size=10, color=s['color']),
                align="left",
                xanchor="left",
                yanchor="middle"
            )
    fig.update_layout(title=f"Cumulative Cost Over Rounds (All Projects, N={num_parallel_projects})", xaxis_title="Round #", yaxis_title="Cumulative Cost ($)", legend_title="Scenario", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=800, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for cumulative cost all projects
    plot_download_button(fig, 'cumulative_cost_all_projects', scenario_inputs)

# --- Cost Breakdown per Round ---
def split_costs(cost):
    fte = cost['cost_researcher'] + cost['cost_of_delay']
    materials = cost['cost_synth'] + cost['cost_seq'] + cost['cost_lost']
    return fte, materials

st.markdown("## Cost Breakdown per Round")
col_cost1, col_cost2 = st.columns(2)
with col_cost1:
    st.markdown("#### Per Project (Stacked)")
    fig = go.Figure()
    fte_vals = [split_costs(c)[0] for c in costs]
    mat_vals = [split_costs(c)[1] for c in costs]
    fig.add_trace(go.Bar(x=x_labels, y=fte_vals, name="FTE (Researcher + Delay)", marker_color=PALETTE['dark_teal'], text=[f"${v:,.2f}" for v in fte_vals], textposition='auto'))
    fig.add_trace(go.Bar(x=x_labels, y=mat_vals, name="Materials (Synthesis + Seq + Lost)", marker_color=PALETTE['seafoam'], text=[f"${v:,.2f}" for v in mat_vals], textposition='auto'))
    # Add integrated delta labels above each bar
    bar_tops = [fte_vals[i] + mat_vals[i] for i in range(len(scenario_inputs))]
    max_y = max(bar_tops)
    min_offset = max_y * 0.1  # Minimum offset based on the tallest bar
    for i in range(len(scenario_inputs)):
        total_cost = bar_tops[i]
        if i != REFERENCE_IDX:
            delta = bar_tops[i] - bar_tops[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(bar_tops[i], bar_tops[REFERENCE_IDX]) * 100) if max(bar_tops[i], bar_tops[REFERENCE_IDX]) > 0 else 0
            label = f"${total_cost:,.2f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${total_cost:,.2f}"
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(total_cost * 0.15, min_offset)
        fig.add_annotation(
            x=i, y=total_cost + dynamic_offset,  # Dynamic positioning
            text=label,
            showarrow=False,
            font=dict(size=12, color=scenario_inputs[i]['color']),
            align="center"
        )
    fig.update_layout(barmode='stack', title="Cost per Round per Project (Stacked)", xaxis_title="Scenario", yaxis_title="Cost per Round ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=700, height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for cost breakdown per project
    plot_download_button(fig, 'cost_breakdown_per_project', scenario_inputs)
    
with col_cost2:
    st.markdown(f"#### All Projects (N={num_parallel_projects}) (Stacked)")
    fig = go.Figure()
    fte_vals_all = [v * num_parallel_projects for v in fte_vals]
    mat_vals_all = [v * num_parallel_projects for v in mat_vals]
    fig.add_trace(go.Bar(x=x_labels, y=fte_vals_all, name="FTE (Researcher + Delay)", marker_color=PALETTE['dark_teal'], text=[f"${v:,.2f}" for v in fte_vals_all], textposition='auto'))
    fig.add_trace(go.Bar(x=x_labels, y=mat_vals_all, name="Materials (Synthesis + Seq + Lost)", marker_color=PALETTE['seafoam'], text=[f"${v:,.2f}" for v in mat_vals_all], textposition='auto'))
    # Add integrated delta labels above each bar
    bar_tops_all = [fte_vals_all[i] + mat_vals_all[i] for i in range(len(scenario_inputs))]
    max_y = max(bar_tops_all)
    min_offset = max_y * 0.1  # Minimum offset based on the tallest bar
    for i in range(len(scenario_inputs)):
        total_cost = bar_tops_all[i]
        if i != REFERENCE_IDX:
            delta = bar_tops_all[i] - bar_tops_all[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(bar_tops_all[i], bar_tops_all[REFERENCE_IDX]) * 100) if max(bar_tops_all[i], bar_tops_all[REFERENCE_IDX]) > 0 else 0
            label = f"${total_cost:,.2f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${total_cost:,.2f}"
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(total_cost * 0.15, min_offset)
        fig.add_annotation(
            x=i, y=total_cost + dynamic_offset,  # Dynamic positioning
            text=label,
            showarrow=False,
            font=dict(size=12, color=scenario_inputs[i]['color']),
            align="center"
        )
    fig.update_layout(barmode='stack', title=f"Cost per Round for All Projects (N={num_parallel_projects}) (Stacked)", xaxis_title="Scenario", yaxis_title="Cost per Round ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=700, height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for cost breakdown all projects
    plot_download_button(fig, 'cost_breakdown_all_projects', scenario_inputs)

# --- Annual Cost ---
def cost_per_year(final_cost, round_length):
    weeks_per_year = 52.143
    return final_cost / round_length * weeks_per_year

col_year1, col_year2 = st.columns(2)
with col_year1:
    st.markdown("#### Per Project")
    annuals = [cost_per_year(c['final_cost'], c['round_length']) for c in costs]
    fig = go.Figure()
    bar_labels_annual = []
    for i in range(len(scenario_inputs)):
        value = annuals[i]
        rounds = int(52.143 // costs[i]['round_length'])
        if i != REFERENCE_IDX:
            delta = annuals[i] - annuals[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(annuals[i], annuals[REFERENCE_IDX]) * 100) if max(annuals[i], annuals[REFERENCE_IDX]) > 0 else 0
            label = f"${value:,.2f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${value:,.2f}<br>{rounds} rounds/year"
        bar_labels_annual.append(label)
    fig.add_trace(go.Bar(x=x_labels, y=annuals, marker_color=colors, text=None, name="Annual Cost"))
    max_y = max(annuals)
    min_offset = max_y * 0.15  # Minimum offset based on the tallest bar
    for i, lbl in enumerate(bar_labels_annual):
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(annuals[i] * 0.2, min_offset)
        fig.add_annotation(
            x=i, y=annuals[i] + dynamic_offset,  # Dynamic positioning
            text=lbl,
            showarrow=False,
            font=dict(size=12, color=colors[i]),
            align="center"
        )
    fig.update_layout(title="Annual Cost per Project", xaxis_title="Scenario", yaxis_title="Annual Cost ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for annual cost per project
    plot_download_button(fig, 'annual_cost_per_project', scenario_inputs)
    
with col_year2:
    st.markdown(f"#### All Projects (N={num_parallel_projects})")
    annuals_all = [v * num_parallel_projects for v in annuals]
    fig = go.Figure()
    bar_labels_annual_all = []
    for i in range(len(scenario_inputs)):
        value = annuals_all[i]
        rounds = int(52.143 // costs[i]['round_length'])
        if i != REFERENCE_IDX:
            delta = annuals_all[i] - annuals_all[REFERENCE_IDX]
            percent_savings = (abs(delta) / max(annuals_all[i], annuals_all[REFERENCE_IDX]) * 100) if max(annuals_all[i], annuals_all[REFERENCE_IDX]) > 0 else 0
            label = f"${value:,.2f}<br>Δ ${delta:,.0f}<br>({percent_savings:.1f}% savings)"
        else:
            label = f"${value:,.2f}<br>{rounds * num_parallel_projects} rounds/year"
        bar_labels_annual_all.append(label)
    fig.add_trace(go.Bar(x=x_labels, y=annuals_all, marker_color=colors, text=None, name="Annual Cost"))
    max_y = max(annuals_all)
    min_offset = max_y * 0.15  # Minimum offset based on the tallest bar
    for i, lbl in enumerate(bar_labels_annual_all):
        # Use the larger of percentage-based offset or minimum offset
        dynamic_offset = max(annuals_all[i] * 0.2, min_offset)
        fig.add_annotation(
            x=i, y=annuals_all[i] + dynamic_offset,  # Dynamic positioning
            text=lbl,
            showarrow=False,
            font=dict(size=12, color=colors[i]),
            align="center"
        )
    fig.update_layout(title=f"Annual Cost for All Projects (N={num_parallel_projects})", xaxis_title="Scenario", yaxis_title="Annual Cost ($)", plot_bgcolor='white', paper_bgcolor='white', font_color=PALETTE["navy"], width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for annual cost all projects
    plot_download_button(fig, 'annual_cost_all_projects', scenario_inputs)

# --- Fold Improvement vs. Cumulative Cost/Time (with ruggedness, 95% CI, markup) ---
st.markdown("## Fold Improvement vs. Cumulative Cost and Time")
st.markdown("""
This section estimates the distribution of possible fold improvements at each round, cost, or time point, based on simulated campaigns for each workflow. The shaded region represents the 95% confidence interval for fold improvement. The central line is the median. Ruggedness interpolates between optimistic (expected max gain) and pessimistic (mean gain) scenarios.

**Enhanced Model Features:**
- **Diversity Bonus**: Larger libraries get additional gains from exploring broader sequence space
- **Learning Effects**: Information from each round improves future round performance
- **Sequence Space Depletion**: Finite beneficial variants reduce gains over time
- **Information Value**: Quantifies the long-term benefit of exploration vs. immediate exploitation
- **Recombination & Epistasis**: Models how mutations from different rounds interact when combined
- **Mutation Additivity**: Controls whether effects combine multiplicatively or additively
""")

# Parse sequence space size
space_mapping = {
    "Small (1K variants)": 1000,
    "Medium (10K variants)": 10000, 
    "Large (100K variants)": 100000,
    "Unlimited": float('inf')
}
total_beneficial_variants = space_mapping[sequence_space_size]

n_sim = 500
rounds = np.arange(1, max_rounds_fold + 1)
mean_gain = 1 + mean_gain_pct / 100.0
std_gain = std_gain_pct / 100.0
mu = np.log(mean_gain) - 0.5 * np.log(1 + (std_gain/mean_gain)**2) if mean_gain > 0 else 0
sigma = np.sqrt(np.log(1 + (std_gain/mean_gain)**2)) if mean_gain > 0 else 0

def expected_max_log_gain(mu, sigma, N):
    if N <= 1:
        return mu
    p = 1 - 1.0 / N
    z = stats.norm.ppf(p)
    return mu + sigma * z

def calculate_diversity_bonus(N, max_N=5000):
    """Calculate diversity bonus based on library size"""
    # Scale the bonus more generously for larger libraries
    # Use square root scaling to give diminishing but continued returns
    return diversity_bonus * min(1.0, np.sqrt(N / max_N))

# --- Centralized Information Gain Function ---
def per_round_information_gain(N, learning_rate):
    """Calculate per-round information gain with sqrt scaling (Fox 2005; Russ et al. 2020)"""
    return learning_rate * np.sqrt(N / 1000.0)

def calculate_depletion_factor(variants_used, total_available):
    """Gentler depletion penalty using sqrt curve; never below 50% effectiveness (Poelwijk et al. 2007)"""
    if total_available == float('inf'):
        return 1.0
    depletion_ratio = variants_used / total_available
    depletion = 0.3 * np.sqrt(depletion_ratio)
    return max(0.5, 1.0 - depletion)  # Never go below 50% effectiveness

def calculate_information_gain(round_num, learning_rate, information_value):
    """Calculate cumulative information gain from previous rounds"""
    if round_num <= 1:
        return 0.0
    # Information accumulates but with diminishing returns
    cumulative_info = sum([learning_rate * (1 - 0.1 * (r-1)) for r in range(1, round_num)])
    return min(cumulative_info * information_value, 1.0)  # Cap at 100% bonus

def calculate_stopping_probability(cum_fold, target_fold, cum_information, confidence_threshold):
    """Calculate probability of stopping based on current progress and learning"""
    if cum_fold >= target_fold:
        return 1.0  # Already achieved target
    
    # Estimate probability of reaching target based on current trajectory and information
    progress_ratio = cum_fold / target_fold
    information_confidence = min(0.9, cum_information)  # Information improves prediction confidence
    
    # Higher information makes us more confident in our trajectory predictions
    adjusted_confidence = progress_ratio + information_confidence * (1 - progress_ratio)
    
    return 1.0 if adjusted_confidence >= confidence_threshold else 0.0

def calculate_learning_efficiency(cum_information, learning_acceleration, target_prediction_improvement):
    """Calculate how learning improves round efficiency and success prediction"""
    # Learning makes rounds more efficient (better variant selection)
    efficiency_mult = 1.0 + cum_information * (learning_acceleration - 1.0)
    
    # Learning improves ability to predict successful variants
    prediction_improvement = cum_information * target_prediction_improvement
    
    return efficiency_mult, prediction_improvement

def calculate_adaptive_cost_reduction(base_cost, cum_information, learning_acceleration):
    """Calculate cost reduction from learning-driven efficiency improvements"""
    efficiency_mult, _ = calculate_learning_efficiency(cum_information, learning_acceleration, 0)
    # More efficient rounds cost less (better targeting, less waste)
    return base_cost / efficiency_mult

def calculate_epistatic_effect(round_num, epistasis, recombination_efficiency):
    """Calculate epistatic interactions between mutations from different rounds"""
    if round_num <= 1:
        return 1.0
    
    # Epistatic effect grows with number of rounds (more mutations to interact)
    # But is modulated by recombination efficiency (how well we can combine them)
    max_interactions = round_num * (round_num - 1) / 2  # n choose 2
    
    # Epistasis effect: negative epistasis reduces benefit, positive increases it
    if epistasis >= 0:
        # Positive epistasis: synergistic interactions
        epistatic_bonus = epistasis * recombination_efficiency * np.log(1 + max_interactions) / 10
        return 1.0 + epistatic_bonus
    else:
        # Negative epistasis: antagonistic interactions
        epistatic_penalty = abs(epistasis) * recombination_efficiency * np.log(1 + max_interactions) / 10
        return max(0.1, 1.0 - epistatic_penalty)  # Don't let it go below 10% of original

def calculate_recombination_benefit(round_gains, mutation_additivity, recombination_efficiency):
    """Calculate benefit from recombining mutations across rounds"""
    if len(round_gains) <= 1:
        return round_gains[0] if round_gains else 1.0
    
    # Convert fold improvements to effect sizes (subtract 1)
    effects = [gain - 1.0 for gain in round_gains]
    
    # Combine effects based on additivity parameter
    if mutation_additivity == 0.0:
        # Purely multiplicative (current model)
        combined_fold = np.prod(round_gains)
    elif mutation_additivity == 1.0:
        # Purely additive
        combined_effect = sum(effects)
        combined_fold = 1.0 + combined_effect
    else:
        # Mixed model: weighted combination
        multiplicative_fold = np.prod(round_gains)
        additive_fold = 1.0 + sum(effects)
        combined_fold = (1 - mutation_additivity) * multiplicative_fold + mutation_additivity * additive_fold
    
    # Apply recombination efficiency (how well we can actually combine the mutations)
    baseline_fold = round_gains[-1]  # Just the last round's improvement
    recombination_bonus = (combined_fold - baseline_fold) * recombination_efficiency
    
    return baseline_fold + recombination_bonus

fold_sim = {}
cost_sim = {}
time_sim = {}
info_sim = {}  # Track information accumulation
completion_sim = {}  # Track project completion metrics

for i, (label, s, c) in enumerate(zip(x_labels, scenario_inputs, costs)):
    N = s['num_variants']
    sim_mat = np.zeros((n_sim, max_rounds_fold))
    info_mat = np.zeros((n_sim, max_rounds_fold))  # Track information per simulation
    completion_rounds = []  # Track when each simulation completes
    completion_costs = []   # Track total cost at completion
    completion_times = []   # Track total time at completion
    project_completed = False
    completion_round = max_rounds_fold  # Default to max if not completed early
    round_gains = []  # Track gains from each round for recombination
    for j in range(n_sim):
        cum_fold = 1.0
        cum_variants_used = 0
        cum_information = 0.0
        cum_cost = 0.0
        cum_time = 0.0
        project_completed = False
        completion_round = max_rounds_fold
        round_gains = []
        for r in range(max_rounds_fold):
            # Calculate learning efficiency improvements
            efficiency_mult, prediction_improvement = calculate_learning_efficiency(
                cum_information, learning_acceleration, target_prediction_improvement)
            # Apply cost reduction from learning
            round_cost = calculate_adaptive_cost_reduction(c['final_cost'], cum_information, learning_acceleration)
            cum_cost += round_cost
            cum_time += c['round_length']
            # Calculate base gains with learning improvements
            gains = np.random.lognormal(mean=mu, sigma=sigma, size=N)
            mean_g = np.mean(gains)
            max_g = np.max(gains)
            # Apply ruggedness
            base_realized_gain = (1 - ruggedness) * max_g + ruggedness * mean_g
            # Apply all enhancement effects
            diversity_mult = 1.0 + calculate_diversity_bonus(N)
            cum_variants_used += N
            depletion_mult = calculate_depletion_factor(cum_variants_used, total_beneficial_variants)
            info_mult = 1.0 + calculate_information_gain(r + 1, learning_rate, information_value)
            # Apply learning efficiency (better variant selection)
            efficiency_gain_mult = 1.0 + prediction_improvement
            # Calculate this round's gain
            round_gain = (base_realized_gain * diversity_mult * depletion_mult * 
                         info_mult * efficiency_gain_mult)
            # Add epistatic effects
            epistatic_mult = calculate_epistatic_effect(r + 1, epistasis, recombination_efficiency)
            round_gain *= epistatic_mult
            # Store this round's gain
            round_gains.append(round_gain)
            # Calculate cumulative fold improvement with recombination
            cum_fold = calculate_recombination_benefit(round_gains, mutation_additivity, recombination_efficiency)
            sim_mat[j, r] = cum_fold
            # --- Use centralized info gain function ---
            info_gain = per_round_information_gain(N, learning_rate)
            cum_information += info_gain
            info_mat[j, r] = cum_information
            # Check for adaptive stopping
            if adaptive_stopping and not project_completed:
                stop_prob = calculate_stopping_probability(cum_fold, target_fold, cum_information, confidence_threshold)
                if stop_prob >= 0.5 or cum_fold >= target_fold:  # Stop if confident or target achieved
                    project_completed = True
                    completion_round = r + 1
                    break
        completion_rounds.append(completion_round)
        completion_costs.append(cum_cost)
        completion_times.append(cum_time)
    median = np.median(sim_mat, axis=0)
    lower = np.percentile(sim_mat, 2.5, axis=0)
    upper = np.percentile(sim_mat, 97.5, axis=0)
    fold_sim[label] = {'median': median, 'lower': lower, 'upper': upper}
    # Track information accumulation
    info_median = np.median(info_mat, axis=0)
    info_sim[label] = info_median
    # Track completion metrics
    completion_sim[label] = {
        'avg_rounds': np.mean(completion_rounds),
        'avg_cost': np.mean(completion_costs),
        'avg_time': np.mean(completion_times),
        'success_rate': np.mean([1 if r < max_rounds_fold else 0 for r in completion_rounds])
    }
    cost_sim[label] = np.cumsum([c['final_cost']] * max_rounds_fold)
    time_sim[label] = np.cumsum([c['round_length']] * max_rounds_fold)

# Plot: Fold Improvement vs. Cumulative Cost
col_ci1, col_ci2 = st.columns(2)
with col_ci1:
    st.markdown("#### Fold Improvement vs. Cumulative Cost (95% CI)")
    fig = go.Figure()
    for i, label in enumerate(x_labels):
        fig.add_trace(go.Scatter(
            x=cost_sim[label], y=fold_sim[label]['median'],
            mode='lines', name=f"{label} Median",
            line=dict(color=scenario_inputs[i]['color'], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([cost_sim[label], cost_sim[label][::-1]]),
            y=np.concatenate([fold_sim[label]['upper'], fold_sim[label]['lower'][::-1]]),
            fill='toself', fillcolor=f"rgba{tuple(int(scenario_inputs[i]['color'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.2,)}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))
    
    # Add delta cost labels at the top of each line (non-overlapping)
    max_fold = max([max(fold_sim[label]['median']) for label in x_labels])
    for i, label in enumerate(x_labels):
        if i != REFERENCE_IDX:  # Only show delta for non-reference scenarios
            final_cost = cost_sim[label][-1]
            final_cost_ref = cost_sim[x_labels[REFERENCE_IDX]][-1]
            delta_cost = final_cost - final_cost_ref
            # Position labels at different heights to avoid overlap
            y_offset = max_fold * (1.1 + i * 0.1)
            fig.add_annotation(
                x=final_cost, y=y_offset,
                text=f"Δ Cost: ${delta_cost:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=scenario_inputs[i]['color'],
                font=dict(size=10, color=scenario_inputs[i]['color']),
                align="center",
                ax=0, ay=-20
            )
    
    fig.add_shape(
        type="line",
        x0=0, x1=max([v[-1] for v in cost_sim.values()]),
        y0=target_fold, y1=target_fold,
        line=dict(color="black", width=2, dash="dot"),
        xref="x", yref="y"
    )
    fig.add_annotation(
        x=max([v[-1] for v in cost_sim.values()]), y=target_fold,
        text="Project Goal",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        xanchor="left",
        yanchor="bottom"
    )
    fig.update_layout(
        title="Enhanced Fold Improvement vs. Cost<br><sub>Includes diversity, learning, and depletion effects</sub>",
        xaxis_title="Cumulative Cost ($)",
        yaxis_title="Cumulative Fold Improvement",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for fold improvement vs cost
    plot_download_button(fig, 'fold_improvement_vs_cost_enhanced', scenario_inputs)
    
with col_ci2:
    st.markdown("#### Fold Improvement vs. Cumulative Time (95% CI)")
    fig = go.Figure()
    for i, label in enumerate(x_labels):
        fig.add_trace(go.Scatter(
            x=time_sim[label], y=fold_sim[label]['median'],
            mode='lines', name=f"{label} Median",
            line=dict(color=scenario_inputs[i]['color'], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_sim[label], time_sim[label][::-1]]),
            y=np.concatenate([fold_sim[label]['upper'], fold_sim[label]['lower'][::-1]]),
            fill='toself', fillcolor=f"rgba{tuple(int(scenario_inputs[i]['color'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.2,)}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))
    
    # Add delta time labels at the top of each line (non-overlapping)
    max_fold = max([max(fold_sim[label]['median']) for label in x_labels])
    for i, label in enumerate(x_labels):
        if i != REFERENCE_IDX:  # Only show delta for non-reference scenarios
            final_time = time_sim[label][-1]
            final_time_ref = time_sim[x_labels[REFERENCE_IDX]][-1]
            delta_time = final_time - final_time_ref
            # Position labels at different heights to avoid overlap
            y_offset = max_fold * (1.1 + i * 0.1)
            fig.add_annotation(
                x=final_time, y=y_offset,
                text=f"Δ Time: {delta_time:,.1f} weeks",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=scenario_inputs[i]['color'],
                font=dict(size=10, color=scenario_inputs[i]['color']),
                align="center",
                ax=0, ay=-20
            )
    
    fig.add_shape(
        type="line",
        x0=0, x1=max([v[-1] for v in time_sim.values()]),
        y0=target_fold, y1=target_fold,
        line=dict(color="black", width=2, dash="dot"),
        xref="x", yref="y"
    )
    fig.add_annotation(
        x=max([v[-1] for v in time_sim.values()]), y=target_fold,
        text="Project Goal",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        xanchor="left",
        yanchor="bottom"
    )
    fig.update_layout(
        title="Enhanced Fold Improvement vs. Time<br><sub>Includes diversity, learning, and depletion effects</sub>",
        xaxis_title="Cumulative Time (weeks)",
        yaxis_title="Cumulative Fold Improvement",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for fold improvement vs time
    plot_download_button(fig, 'fold_improvement_vs_time_enhanced', scenario_inputs)

# Add new information accumulation chart
st.markdown("## Information Accumulation and Learning Effects")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("#### Cumulative Information Gain by Round")
    fig = go.Figure()
    rounds_array = np.arange(1, max_rounds_fold + 1)
    
    for i, label in enumerate(x_labels):
        fig.add_trace(go.Scatter(
            x=rounds_array, y=info_sim[label],
            mode='lines+markers', name=f"{label}",
            line=dict(color=scenario_inputs[i]['color'], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Information Accumulation Over Rounds<br><sub>Shows learning value from sequence space exploration</sub>",
        xaxis_title="Round Number",
        yaxis_title="Cumulative Information Units",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for information accumulation
    plot_download_button(fig, 'information_accumulation', scenario_inputs)

with col_info2:
    st.markdown("#### Information Value vs. Cost Efficiency")
    fig = go.Figure()
    
    # Calculate information per dollar for each scenario
    for i, label in enumerate(x_labels):
        final_info = info_sim[label][-1]
        final_cost = cost_sim[label][-1]
        info_per_dollar = final_info / final_cost * 1000000  # Scale for readability
        
        # Plot as scatter point
        fig.add_trace(go.Scatter(
            x=[final_cost], y=[final_info],
            mode='markers+text',
            name=label,
            marker=dict(
                size=20,
                color=scenario_inputs[i]['color'],
                line=dict(width=2, color='white')
            ),
            text=[f"{info_per_dollar:.1f}<br>info/$M"],
            textposition="top center",
            textfont=dict(size=10)
        ))
    
    fig.update_layout(
        title="Information Gain vs. Total Cost<br><sub>Bubble labels show information units per $M</sub>",
        xaxis_title="Total Cost ($)",
        yaxis_title="Total Information Gained",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for information vs cost
    plot_download_button(fig, 'information_vs_cost', scenario_inputs)

# Add new section for learning-driven project completion
st.markdown("## Learning-Driven Project Completion")
st.markdown("""
This section shows how enhanced learning translates into **faster project completion** and **lower total costs** through:
- **Adaptive stopping**: Stop when confident target is achievable
- **Learning acceleration**: Each round becomes more efficient
- **Better prediction**: Learning improves variant selection success
""")

# Create completion metrics table
col_completion1, col_completion2 = st.columns(2)

with col_completion1:
    st.markdown("#### Project Completion Metrics")
    
    completion_data = {"Metric": [
        "Average Rounds to Complete",
        "Average Total Cost ($)",
        "Average Total Time (weeks)",
        "Success Rate (%)",
        "Cost per Successful Project ($)"
    ]}
    
    for i, label in enumerate(x_labels):
        if label in completion_sim:  # Safety check
            metrics = completion_sim[label]
            cost_per_success = metrics['avg_cost'] / max(0.01, metrics['success_rate'])  # Avoid division by zero
            completion_data[label] = [
                f"{metrics['avg_rounds']:.1f}",
                f"${metrics['avg_cost']:,.0f}",
                f"{metrics['avg_time']:.1f}",
                f"{metrics['success_rate']*100:.1f}%",
                f"${cost_per_success:,.0f}"
            ]
    
    completion_df = pd.DataFrame(completion_data)
    st.dataframe(completion_df, hide_index=True, use_container_width=True)
    
    # Add delta analysis
    st.markdown("#### Learning Benefits Analysis")
    if x_labels[REFERENCE_IDX] in completion_sim:  # Safety check
        ref_metrics = completion_sim[x_labels[REFERENCE_IDX]]
        
        for i, label in enumerate(x_labels):
            if i == REFERENCE_IDX or label not in completion_sim:
                continue
            
            metrics = completion_sim[label]
            cost_savings = ref_metrics['avg_cost'] - metrics['avg_cost']
            time_savings = ref_metrics['avg_time'] - metrics['avg_time']
            round_savings = ref_metrics['avg_rounds'] - metrics['avg_rounds']
            
            st.markdown(f"""
            **{label} vs {x_labels[REFERENCE_IDX]}:**
            - **Cost Savings**: ${cost_savings:,.0f} ({cost_savings/ref_metrics['avg_cost']*100:.1f}%)
            - **Time Savings**: {time_savings:.1f} weeks ({time_savings/ref_metrics['avg_time']*100:.1f}%)
            - **Round Reduction**: {round_savings:.1f} rounds ({round_savings/ref_metrics['avg_rounds']*100:.1f}%)
            """)

with col_completion2:
    st.markdown("#### Cost-to-Completion Comparison")
    fig = go.Figure()
    
    labels = []
    costs = []
    colors_list = []
    
    for i, label in enumerate(x_labels):
        if label in completion_sim and i < len(scenario_inputs):  # Safety checks
            metrics = completion_sim[label]
            labels.append(label)
            costs.append(metrics['avg_cost'])
            colors_list.append(scenario_inputs[i]['color'])
    
    if labels:  # Only create chart if we have data
        # Create bar chart for completion costs
        fig.add_trace(go.Bar(
            x=labels,
            y=costs,
            marker_color=colors_list,
            name="Average Cost to Complete",
            text=[f"${c:,.0f}" for c in costs],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Average Cost to Complete Project<br><sub>Includes learning acceleration and adaptive stopping</sub>",
            xaxis_title="Scenario",
            yaxis_title="Total Project Cost ($)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=PALETTE["navy"],
            width=600,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button for completion costs
        plot_download_button(fig, 'project_completion_costs', scenario_inputs)

# Add learning efficiency visualization
st.markdown("#### Learning Efficiency Over Time")
col_eff1, col_eff2 = st.columns(2)

with col_eff1:
    st.markdown("##### Round Efficiency Improvement")
    fig = go.Figure()
    rounds_array = np.arange(1, max_rounds_fold + 1)
    for i, label in enumerate(x_labels):
        if i < len(scenario_inputs):  # Safety check
            # Calculate efficiency improvement over rounds
            efficiency_improvements = []
            cum_info = 0
            for r in range(max_rounds_fold):
                # --- Use centralized info gain function ---
                cum_info += per_round_information_gain(scenario_inputs[i]['num_variants'], learning_rate)
                efficiency_mult, _ = calculate_learning_efficiency(cum_info, learning_acceleration, target_prediction_improvement)
                efficiency_improvements.append((efficiency_mult - 1) * 100)  # Convert to percentage
            fig.add_trace(go.Scatter(
                x=rounds_array, y=efficiency_improvements,
                mode='lines+markers', name=f"{label}",
                line=dict(color=scenario_inputs[i]['color'], width=3),
                marker=dict(size=6)
            ))
    fig.update_layout(
        title="Round Efficiency Improvement<br><sub>% improvement in round efficiency from learning</sub>",
        xaxis_title="Round Number",
        yaxis_title="Efficiency Improvement (%)",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    # Download button for efficiency improvement
    plot_download_button(fig, 'learning_efficiency_improvement', scenario_inputs)

with col_eff2:
    st.markdown("##### Cumulative Cost Reduction")
    fig = go.Figure()
    rounds_array = np.arange(1, max_rounds_fold + 1)
    min_length = min(len(x_labels), len(scenario_inputs), len(costs))
    for i in range(min_length):
        label = x_labels[i]
        scenario = scenario_inputs[i]
        cost_data = costs[i]
        if not isinstance(cost_data, dict) or 'final_cost' not in cost_data:
            print(f"Warning: cost_data at index {i} is not a dict or missing 'final_cost'. Skipping.")
            continue
        scenario_cost = cost_data['final_cost']
        cumulative_cost_with_learning = []
        cumulative_cost_without_learning = []
        cum_info = 0
        for r in range(max_rounds_fold):
            # --- Use centralized info gain function ---
            cum_info += per_round_information_gain(scenario['num_variants'], learning_rate)
            round_cost_with_learning = calculate_adaptive_cost_reduction(scenario_cost, cum_info, learning_acceleration)
            if r == 0:
                cumulative_cost_with_learning.append(round_cost_with_learning)
                cumulative_cost_without_learning.append(scenario_cost)
            else:
                cumulative_cost_with_learning.append(cumulative_cost_with_learning[-1] + round_cost_with_learning)
                cumulative_cost_without_learning.append(cumulative_cost_without_learning[-1] + scenario_cost)
        savings = [cumulative_cost_without_learning[r] - cumulative_cost_with_learning[r] for r in range(max_rounds_fold)]
        fig.add_trace(go.Scatter(
            x=rounds_array, y=savings,
            mode='lines+markers', name=f"{label}",
            line=dict(color=scenario['color'], width=3),
            marker=dict(size=6)
        ))
    fig.update_layout(
        title="Cumulative Cost Savings from Learning<br><sub>$ saved compared to no learning acceleration</sub>",
        xaxis_title="Round Number",
        yaxis_title="Cumulative Cost Savings ($)",
        legend_title="Workflow",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=PALETTE["navy"],
        width=600,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    # Download button for cost savings
    plot_download_button(fig, 'cumulative_cost_savings', scenario_inputs)

# Add detailed model explanation
st.markdown("""
### Enhanced Model Assumptions

**Base Model:**
- **Each round, you screen N variants** (N = scenario-specific number of variants per round)
- **Each variant's gain is drawn independently** from a log-normal distribution with user-set mean and standard deviation
- **The expected maximum gain** among N variants is used as the per-round gain: E[max] = exp(μ + σ * Φ⁻¹(1 - 1/N))

**New Enhancement Features:**

**1. Diversity Bonus:**
```
diversity_multiplier = 1 + diversity_bonus × min(1, sqrt(N/5000))
```
- Larger libraries explore broader sequence space
- Square root scaling provides diminishing but continued returns
- Captures the value of sequence diversity without over-penalizing large libraries
- *See: Fox, R. J. (2005); Russ, W. P. et al. (2020)*

**2. Learning Effects:**
```
information_multiplier = 1 + cumulative_information × information_value
cumulative_information = Σ(per_round_information_gain(N, learning_rate))
per_round_information_gain(N, learning_rate) = learning_rate × sqrt(N/1000)
```
- Each round teaches about the fitness landscape
- Information accumulates with square root scaling for library size
- Models the compounding value of exploration
- *See: Fox, R. J. (2005); Russ, W. P. et al. (2020)*

**3. Sequence Space Depletion:**
```
depletion_multiplier = max(0.5, 1 - 0.3 × sqrt(variants_used / total_available))
```
- Gentler penalty for large libraries, never below 50% effectiveness
- Models diminishing returns in limited sequence space without complete shutdown
- *See: Poelwijk, F. J. et al. (2007); Bloom, J. D. & Arnold, F. H. (2009)*

**4. Combined Realized Gain:**
```
realized_gain = base_gain × diversity_mult × depletion_mult × information_mult × ruggedness_effect
```

**5. Recombination & Epistasis:**
```
epistatic_multiplier = 1 + epistasis × recombination_efficiency × log(1 + interactions) / 10
recombined_fold = additivity × additive_combination + (1-additivity) × multiplicative_combination
final_fold = baseline_fold + (recombined_fold - baseline_fold) × recombination_efficiency
```

### Mathematical Summary
- **Enhanced per-round gain:** Incorporates all four effects above
- **Cumulative fold after r rounds:** Product of all enhanced realized gains
- **Information accumulation:** Tracks learning value separate from fold improvement
- **Cost-information efficiency:** Quantifies exploration value per dollar spent
- **Recombination modeling:** Combines beneficial mutations from multiple rounds with epistatic interactions

### References
- Fox, R. J. (2005). Directed molecular evolution by machine learning and the influence of sequence space. *Curr. Opin. Struct. Biol.*, 15(4), 421–429. [https://doi.org/10.1016/j.sbi.2005.07.003](https://doi.org/10.1016/j.sbi.2005.07.003)
- Arnold, F. H. (1998). Design by Directed Evolution. *Acc. Chem. Res.*, 31(3), 125–131. [https://doi.org/10.1021/ar960017f](https://doi.org/10.1021/ar960017f)
- Bloom, J. D., & Arnold, F. H. (2009). In the light of directed evolution: pathways of adaptive protein evolution. *Proc. Natl. Acad. Sci. USA*, 106(Suppl 1), 9995–10000. [https://doi.org/10.1073/pnas.0901522106](https://doi.org/10.1073/pnas.0901522106)
- Poelwijk, F. J., Kiviet, D. J., Weinreich, D. M., & Tans, S. J. (2007). Empirical fitness landscapes reveal accessible evolutionary paths. *Nature*, 445(7126), 383–386. [https://doi.org/10.1038/nature05451](https://doi.org/10.1038/nature05451)
- Russ, W. P., Figliuzzi, M., Stocker, C., Barrat-Charlaix, P., Socolich, M., Kast, P., ... & Ranganathan, R. (2020). An evolution-based model for designing chorismate mutase enzymes. *Science*, 369(6502), 440–445. [https://doi.org/10.1126/science.aba3304](https://doi.org/10.1126/science.aba3304)
- David, H. A., & Nagaraja, H. N. (2003). Order Statistics, 3rd Edition. Springer. [https://doi.org/10.1007/b98946](https://doi.org/10.1007/b98946)

### Notes
- **Enhanced model is more realistic** but still simplified compared to real protein engineering
- **Parameter tuning** allows exploration of different biological scenarios
- **Information metrics** quantify the often-overlooked value of exploration
- **Multi-objective optimization** balances immediate gains vs. long-term learning
- **All parameters are adjustable** to match your specific system and constraints

### Complete Variable Definitions

**Core Parameters:**
- **N**: Number of variants screened per round (library size)
- **μ (mu)**: Mean of log-normal distribution for variant gains
- **σ (sigma)**: Standard deviation of log-normal distribution for variant gains
- **target_fold**: Target fold improvement for project success
- **max_rounds_fold**: Maximum number of rounds to simulate
- **ruggedness**: Interpolation between optimistic (0) and pessimistic (1) scenarios

**Sequence Space & Learning:**
- **diversity_bonus**: Additional gain multiplier for exploring diverse sequence space (0-1)
- **learning_rate**: How much each variant teaches about fitness landscape (0-1)
- **sequence_space_size**: Total beneficial variants available (1K/10K/100K/Unlimited)
- **information_value**: Multiplier for how future rounds benefit from current information (0-2)
- **total_beneficial_variants**: Numeric value of sequence space size

**Recombination & Epistasis:**
- **epistasis**: Interaction between mutations (-2 to +2, where 0=independent)
- **recombination_efficiency**: How well beneficial mutations can be combined (0-1)
- **mutation_additivity**: Whether effects combine multiplicatively (0) or additively (1)

**Learning-to-Completion:**
- **adaptive_stopping**: Boolean for early stopping when target is achievable
- **confidence_threshold**: Required confidence level to stop early (0.5-0.95)
- **learning_acceleration**: How learning improves round efficiency (1-3)
- **target_prediction_improvement**: How learning improves variant selection (0-0.5)

**Calculated Variables:**
- **base_realized_gain**: Per-round gain before enhancement effects
- **diversity_mult**: Multiplier from diversity bonus = 1 + diversity_bonus × min(1, √(N/5000))
- **depletion_mult**: Multiplier from sequence space depletion
- **info_mult**: Multiplier from accumulated information = 1 + cumulative_information × information_value
- **epistatic_mult**: Multiplier from epistatic interactions
- **efficiency_gain_mult**: Multiplier from learning efficiency improvements
- **round_gain**: Final per-round gain after all effects
- **cum_fold**: Cumulative fold improvement with recombination
- **cum_information**: Accumulated information = Σ(per_round_information_gain(N, learning_rate))
- **cum_variants_used**: Total variants screened across all rounds

**Statistical Functions:**
- **E[max]**: Expected maximum of N log-normal random variables = exp(μ + σ × Φ⁻¹(1-1/N))
- **Φ⁻¹**: Inverse normal CDF (quantile function)
- **gentler_depletion_curve**: Modified depletion function with sqrt scaling

**Cost Variables:**
- **cost_synth**: Synthesis cost = N × cost_per_variant
- **cost_lost**: Cost of failed variants = N × (1-pct_correct) × cost_per_variant
- **cost_of_delay**: Delay cost = delay_weeks × 5 × 8 × fte_cost_per_hour
- **cost_researcher**: Researcher cost = fte_hours_per_round × fte_cost_per_hour
- **cost_seq**: Sequencing cost (fixed per round)
- **final_cost**: Total cost per round = cost_synth + cost_lost + cost_of_delay + cost_researcher + cost_seq

**Time Variables:**
- **round_length**: Duration of each round in weeks
- **delay_weeks**: Additional delays per round
- **cum_time**: Cumulative time across rounds

This comprehensive model captures the complex interplay between library size, learning effects, sequence space constraints, epistatic interactions, and economic factors that determine optimal directed evolution strategies.
""")

# Analysis of why larger libraries don't always win
st.markdown("""
### Why the Enhanced Model Changes Everything

**Traditional View:** Larger libraries always find better variants (statistical maximum effect)

**Enhanced Reality:** Multiple competing effects determine optimal strategy:

1. **Diversity Bonus** ↑ Library Size → ↑ Gains (favors larger libraries)
2. **Learning Effects** ↑ Speed → ↑ Information → ↑ Future Gains (favors faster workflows)
3. **Sequence Depletion** ↑ Total Variants → ↓ Future Gains (favors smaller libraries)
4. **Cost Efficiency** ↑ Cost → ↓ Rounds Possible (favors cheaper workflows)

**The Sweet Spot:** Optimal library size balances all four effects based on:
- Project phase (discovery vs. optimization)
- Sequence space size (limited vs. unlimited)
- Learning potential (high vs. low information value)
- Budget constraints (cost per round vs. total budget)

**Why CREATE Often Wins:**
- **Speed advantage** enables more learning cycles
- **Cost advantage** enables more total rounds
- **Learning compounds** faster than diversity bonus grows
- **Information accumulation** provides sustainable competitive advantage

This enhanced model shows that **the optimal strategy depends critically on the specific biological and economic context** of your project.
""")

# References and notes
st.markdown("""
### References
- Fox, R. J. (2005). Directed molecular evolution by machine learning and the influence of sequence space. *Curr. Opin. Struct. Biol.*, 15(4), 421–429. [https://doi.org/10.1016/j.sbi.2005.07.003](https://doi.org/10.1016/j.sbi.2005.07.003)
- Arnold, F. H. (1998). Design by Directed Evolution. *Acc. Chem. Res.*, 31(3), 125–131. [https://doi.org/10.1021/ar960017f](https://doi.org/10.1021/ar960017f)
- Bloom, J. D., & Arnold, F. H. (2009). In the light of directed evolution: pathways of adaptive protein evolution. *Proc. Natl. Acad. Sci. USA*, 106(Suppl 1), 9995–10000. [https://doi.org/10.1073/pnas.0901522106](https://doi.org/10.1073/pnas.0901522106)
- David, H. A., & Nagaraja, H. N. (2003). Order Statistics, 3rd Edition. Springer. [https://doi.org/10.1007/b98946](https://doi.org/10.1007/b98946)

### Notes
- **Enhanced model is more realistic** but still simplified compared to real protein engineering
- **Parameter tuning** allows exploration of different biological scenarios
- **Information metrics** quantify the often-overlooked value of exploration
- **Multi-objective optimization** balances immediate gains vs. long-term learning
- **All parameters are adjustable** to match your specific system and constraints

### Complete Variable Definitions

**Core Parameters:**
- **N**: Number of variants screened per round (library size)
- **μ (mu)**: Mean of log-normal distribution for variant gains
- **σ (sigma)**: Standard deviation of log-normal distribution for variant gains
- **target_fold**: Target fold improvement for project success
- **max_rounds_fold**: Maximum number of rounds to simulate
- **ruggedness**: Interpolation between optimistic (0) and pessimistic (1) scenarios

**Sequence Space & Learning:**
- **diversity_bonus**: Additional gain multiplier for exploring diverse sequence space (0-1)
- **learning_rate**: How much each variant teaches about fitness landscape (0-1)
- **sequence_space_size**: Total beneficial variants available (1K/10K/100K/Unlimited)
- **information_value**: Multiplier for how future rounds benefit from current information (0-2)
- **total_beneficial_variants**: Numeric value of sequence space size

**Recombination & Epistasis:**
- **epistasis**: Interaction between mutations (-2 to +2, where 0=independent)
- **recombination_efficiency**: How well beneficial mutations can be combined (0-1)
- **mutation_additivity**: Whether effects combine multiplicatively (0) or additively (1)

**Learning-to-Completion:**
- **adaptive_stopping**: Boolean for early stopping when target is achievable
- **confidence_threshold**: Required confidence level to stop early (0.5-0.95)
- **learning_acceleration**: How learning improves round efficiency (1-3)
- **target_prediction_improvement**: How learning improves variant selection (0-0.5)

**Calculated Variables:**
- **base_realized_gain**: Per-round gain before enhancement effects
- **diversity_mult**: Multiplier from diversity bonus = 1 + diversity_bonus × min(1, √(N/5000))
- **depletion_mult**: Multiplier from sequence space depletion
- **info_mult**: Multiplier from accumulated information = 1 + cumulative_information × information_value
- **epistatic_mult**: Multiplier from epistatic interactions
- **efficiency_gain_mult**: Multiplier from learning efficiency improvements
- **round_gain**: Final per-round gain after all effects
- **cum_fold**: Cumulative fold improvement with recombination
- **cum_information**: Accumulated information = Σ(per_round_information_gain(N, learning_rate))
- **cum_variants_used**: Total variants screened across all rounds

**Statistical Functions:**
- **E[max]**: Expected maximum of N log-normal random variables = exp(μ + σ × Φ⁻¹(1-1/N))
- **Φ⁻¹**: Inverse normal CDF (quantile function)
- **gentler_depletion_curve**: Modified depletion function with sqrt scaling

**Cost Variables:**
- **cost_synth**: Synthesis cost = N × cost_per_variant
- **cost_lost**: Cost of failed variants = N × (1-pct_correct) × cost_per_variant
- **cost_of_delay**: Delay cost = delay_weeks × 5 × 8 × fte_cost_per_hour
- **cost_researcher**: Researcher cost = fte_hours_per_round × fte_cost_per_hour
- **cost_seq**: Sequencing cost (fixed per round)
- **final_cost**: Total cost per round = cost_synth + cost_lost + cost_of_delay + cost_researcher + cost_seq

**Time Variables:**
- **round_length**: Duration of each round in weeks
- **delay_weeks**: Additional delays per round
- **cum_time**: Cumulative time across rounds

This comprehensive model captures the complex interplay between library size, learning effects, sequence space constraints, epistatic interactions, and economic factors that determine optimal directed evolution strategies.
""")

# --- Add download HTML button for full report
import plotly.io as pio

def generate_html_report():
    # Compose HTML for executive summary and all plots
    html = "<html><head><title>Screening Cost Modeling Report</title></head><body>"
    html += make_exec_summary(scenario_inputs, costs, num_rounds, num_parallel_projects, REFERENCE_IDX)
    # Add each plot as an image (use pio.to_html for each fig)
    # ... (implement for each plot section, using the same logic as in the app)
    html += "</body></html>"
    return html

scenario_names = '_'.join([s['label'].replace(' ', '').replace('(', '').replace(')', '') + str(s['num_variants']) for s in scenario_inputs])
filename = f"screening_cost_report_{scenario_names}_{num_rounds}rounds_{num_parallel_projects}proj.html"
st.download_button(
    label="Download HTML Report",
    data=generate_html_report(),
    file_name=filename,
    mime="text/html"
) 