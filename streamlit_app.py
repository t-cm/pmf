import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # Added missing import

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='PMF Dashboard',
    page_icon=':earth_americas:',
    layout='wide'  # Added for better visualization of plots
)

# -----------------------------------------------------------------------------
# Custom function for net return calculation
# -----------------------------------------------------------------------------
def custom_function(x, a, b, c):
    """
    Calculate net return percentage after borrowing cost.
    
    Parameters:
    - x: Current price point
    - a: Initial price (3000)
    - b: Target price
    - c: Parameter value
    
    Returns:
    - Net return percentage after 5% borrowing cost
    """
    if a == b:
        return np.nan
    if x <= b:
        numerator = ((a * c / (b - a)) * (x - a) + a * (1 - c))
    else:
        numerator = ((a * c / (b - a)) * (x - a) - a * c)
    denominator = (a * (1 + (a * c / (b - a))))
    return 100 * (numerator / denominator) - 5  # Returns net percentage after 5% borrowing cost

# Fixed x range validation not needed anymore

# -----------------------------------------------------------------------------
# 1. Define parameters for the problem: strikes, prices, underlying spot.
# -----------------------------------------------------------------------------
# k_values = [4000, 5000, 6000, 7000, 80000, 10000]      # 7 binary strikes
# c_values = [0.75, 0.50, 0.35, 0.28, 0.25, 0.15]  # Example prices for each binary
# a = 3250  # Current price of the underlying asset (spot)

k_values = [4500, 5000, 6000, 7000, 8000]      # 7 binary strikes
c_values = [0.26, 0.15, 0.08, 0.06, 0.05]  # Example prices for each binary
a = 3250  # Current price of the underlying asset (spot)

# Validate binary prices are in [0,1]
if not all(0 <= c <= 1 for c in c_values):
    st.error("All binary prices must be between 0 and 1")

# -----------------------------------------------------------------------------
# 2. Define payoff functions with input validation
# -----------------------------------------------------------------------------

def binary_payoff(w, c, x, k):
    """
    Returns payoff of one binary position (long/short) at terminal price x.
    Parameters:
    - w ∈ [-1, 1]: +1 means fully long, -1 means fully short
    - c = price of the binary in [0, 1]
    - k = strike price of the binary
    - x = underlying terminal price
    """
    # Input validation
    if not -1 <= w <= 1:
        raise ValueError("Weight w must be between -1 and 1")
    if not 0 <= c <= 1:
        raise ValueError("Binary price c must be between 0 and 1")
    
    return w * (-c if x < k else (1 - c)) if w >= 0 else abs(w) * (c if x < k else (c - 1))

def underlying_payoff(w_asset, a, x):
    """
    Calculate the terminal payoff for a position in the underlying asset.
    Parameters:
    - w_asset ∈ [0,1]: fraction of capital invested in the underlying
    - a: current price of the underlying
    - x: terminal price of the underlying
    """
    # Input validation
    if not -1 <= w_asset <= 1:
        raise ValueError("Asset weight must be between 0 and 1")
    if a <= 0:
        raise ValueError("Current price must be positive")
    if x < 0:
        raise ValueError("Terminal price cannot be negative")
        
    return w_asset * (x / a - 1)

# -----------------------------------------------------------------------------
# 3. Page layout and text
# -----------------------------------------------------------------------------

st.title("📈 Binary Options + Underlying Payoff Dashboard")
st.markdown("""
This dashboard lets you:
1. Explore the *individual payoff* of going long or short a binary option (or a long position in the underlying).
2. Combine multiple binary positions **plus** the underlying into a single portfolio and view the **total payoff**.
""")

# -----------------------------------------------------------------------------
# 4. Fixed x range and number of points
# -----------------------------------------------------------------------------
x_min = 0
x_max = k_values[-1]+1000
n_points = 200
x_values = np.linspace(x_min, x_max, n_points)

# -----------------------------------------------------------------------------
# 5. Tabs: Individual Positions vs. Portfolio
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Individual Positions", "Portfolio", "All Positions"])

# -----------------------------------------------------------------------------
# Tab 1: Individual Payoffs
# -----------------------------------------------------------------------------
with tab1:
    st.header("Individual Payoffs")

    choice = st.selectbox(
        "Choose an instrument to visualize",
        ["Underlying"] + [f"Binary_K{k_values[i]}" for i in range(len(k_values))]
    )

    try:
        # If Underlying selected
        if choice == "Underlying":
            st.markdown("Underlying payoff formula: `w_asset × (x/a - 1)`")
            w_und = st.slider("Underlying weight (w_asset)", 0.0, 1.0, 1.0, step=0.1)

            payoff_vals = [underlying_payoff(w_und, a, x) for x in x_values]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=payoff_vals,
                mode='lines',
                name=f"Underlying (w={w_und:.2f})"
            ))
            
        else:
            # Binary selected
            i = k_values.index(int(choice.split('K')[1]))  # Get index based on strike price
            k_i = k_values[i]
            c_i = c_values[i]

            st.markdown(f"""
            Binary Option K{k_i}:
            - Strike = {k_i}
            - Price (c) = {c_i}
            """)
            
            w_bin = st.slider("Binary weight (long>0, short<0)", -1.0, 1.0, 1.0, step=0.1)

            payoff_vals = [binary_payoff(w_bin, c_i, x, k_i) for x in x_values]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=payoff_vals,
                mode='lines',
                name=f"Binary_{'LONG' if w_bin > 0 else 'SHORT'}_{k_i}"
            ))

        # Common plot settings
        fig.add_shape(
            type="line", x0=x_min, x1=x_max, y0=0, y1=0,
            line=dict(color="gray", dash="dash")
        )
        fig.update_layout(
            xaxis_title="Terminal Price (x)",
            yaxis_title="Payoff",
            title=f"Individual Payoff: {choice}",
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# -----------------------------------------------------------------------------
# Tab 3: Net Return Curves
# -----------------------------------------------------------------------------
with tab3:
    st.header("Net Return Curves")
    
    st.markdown("""
    This visualization shows net return curves after a 5% borrowing cost for different parameter combinations.
    Each curve represents a different (b, c) pair, where:
    - a = 3000 (fixed initial price)
    - b = target price
    - c = parameter value
    """)
    
    # Constants
    a = 3250
    b_c_values = {
        4000: 0.25,
        5000: 0.50,
        6000: 0.65,
        7000: 0.72,
        8000: 0.75,
        10000: 0.86
    }
    
    # Create toggles for different curves
    st.subheader("Select curves to display:")
    cols = st.columns(3)
    active_curves = {}
    for i, (b, c) in enumerate(b_c_values.items()):
        col_idx = i % 3
        with cols[col_idx]:
            active_curves[b] = st.toggle(f"b={b}, c={c:.2f}", value=True)
    
    # Calculate values at x=1000 for ranking
    values_at_1000 = {}
    for b, c in b_c_values.items():
        value = custom_function(1000, a, b, c)
        values_at_1000[(b, c)] = value
    
    # Sort by value at x=1000
    sorted_by_1000 = sorted(values_at_1000.items(), key=lambda x: x[1], reverse=True)
    
    # Display ranked values
    with st.expander("Show ranked net returns at x=1000"):
        st.markdown("Net returns at x=1000 (after 5% borrowing cost):")
        for (b, c), value in sorted_by_1000:
            st.markdown(f"- b={b}, c={c:.2f}: {value:.2f}%")
    
    # Create the figure
    fig = go.Figure()
    
    # Color scale for the curves
    colors = px.colors.qualitative.Set3
    
    # Add traces in sorted order
    for i, ((b, c), _) in enumerate(sorted_by_1000):
        if active_curves[b]:
            x = np.linspace(0, b * 1.5, 200)  # Extended range to 1.5 times b
            y = np.array([custom_function(xi, a, b, c) for xi in x])
            fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                name=f'b={b}, c={c:.2f}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Net Return Curves for a={a} (After 5% Borrowing Cost)',
        xaxis_title='Price (x)',
        yaxis_title='Net Return (%)',
        legend_title='Parameters (b, c)',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # Add a horizontal line at y=0 to show break-even point
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add vertical lines at each b value
    for b, c in b_c_values.items():
        if active_curves[b]:
            fig.add_vline(x=b, line_dash="dot", line_color="gray", opacity=0.3)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("📖 How to read this chart"):
        st.markdown("""
        This chart shows net return curves after a 5% borrowing cost:
        
        - Each curve represents a different combination of parameters (b, c)
        - The horizontal dashed line at y=0 represents the break-even point
        - Vertical dotted lines show the b values (target prices)
        - Use the toggles above to show/hide different curves
        - Hover over the curves to see exact values
        
        The curves are ranked based on their return at x=1000, which you can view 
        in the expandable section above the chart.
        """)

# -----------------------------------------------------------------------------
# Tab 2: Portfolio Payoff
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Tab 2: Portfolio Payoff
# -----------------------------------------------------------------------------
with tab2:
    st.header("Portfolio Payoff")

    st.markdown("""
    Assign weights to create a portfolio where the sum of absolute weights equals 1:
    - The **underlying** (`w_asset ∈ [0,1]`)
    - Each **binary** (`w_i ∈ [-1,1]`)
    
    Current weight sum: **{:.2f}** (must equal 1.0)
    """)

    try:
        col1, col2 = st.columns(2)
        
        with col1:
            w_asset = st.slider(
                "Underlying weight (w_asset)",
                -1.0,
                1.0,
                0.0,
                step=0.01,
                key="underlying_weight"
            )
            remaining_weight = 1.0 - abs(w_asset)
            
            # Display remaining weight
            st.markdown(f"Remaining weight to allocate: **{remaining_weight:.2f}**")

        # Binary weights with strike prices shown
        w_bins = []
        
        # Split binaries between columns
        with col1:
            for i in range(len(k_values)//2):
                current_remaining = 1.0 - (abs(w_asset) + sum(abs(w) for w in w_bins))
                w_i = st.slider(
                    f"Binary_K{k_values[i]} weight",
                    -1.0,  # Allow full range
                    1.0,   # Allow full range
                    0.0,
                    step=0.01,
                    key=f"binary_{i}"
                )
                w_bins.append(w_i)
                st.markdown(f"Remaining weight: **{max(0, current_remaining):.2f}**")
                
        with col2:
            for i in range(len(k_values)//2, len(k_values)):
                current_remaining = 1.0 - (abs(w_asset) + sum(abs(w) for w in w_bins))
                w_i = st.slider(
                    f"Binary_K{k_values[i]} weight",
                    -1.0,  # Allow full range
                    1.0,   # Allow full range
                    0.0,
                    step=0.01,
                    key=f"binary_{i}"
                )
                w_bins.append(w_i)
                st.markdown(f"Remaining weight: **{max(0, current_remaining):.2f}**")

        # Calculate total absolute weight
        total_weight = abs(w_asset) + sum(abs(w) for w in w_bins)
        
        # Show warning if weights don't sum to 1
        # if abs(total_weight - 1.0) > 0.01:  # Using 0.01 tolerance due to floating point
        #     st.warning(f"Total absolute weight must equal 1.0 (currently {total_weight:.2f})")
        #     if total_weight > 1.0:
        #         st.error("Please reduce some weights to make the total absolute sum equal to 1.0")
        #         st.stop()
        
        # Calculate portfolio payoff with error handling
        payoff_portfolio = []
        for x in x_values:
            total = underlying_payoff(w_asset, a, x)
            for k_i, c_i, w_i in zip(k_values, c_values, w_bins):
                total += binary_payoff(w_i, c_i, x, k_i)
            payoff_portfolio.append(total)

        # Plot
        fig = go.Figure()
        
        # Add individual position traces with low opacity
        if w_asset != 0:
            y_underlying = [underlying_payoff(w_asset, a, x) for x in x_values]
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_underlying,
                mode='lines',
                name=f"Underlying (w={w_asset:.2f})",
                line=dict(dash='dash', width=1),
                opacity=0.3
            ))
            
        for k_i, c_i, w_i in zip(k_values, c_values, w_bins):
            if w_i != 0:
                y_binary = [binary_payoff(w_i, c_i, x, k_i) for x in x_values]
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_binary,
                    mode='lines',
                    name=f"Binary_K{k_i} (w={w_i:.2f})",
                    line=dict(dash='dash', width=1),
                    opacity=0.3
                ))
        
        # Add total portfolio payoff with full opacity
        fig.add_trace(go.Scatter(
            x=x_values,
            y=payoff_portfolio,
            mode='lines',
            name="Portfolio total",
            line=dict(color='black', width=2)
        ))
        
        fig.add_shape(
            type="line", x0=x_min, x1=x_max, y0=0, y1=0,
            line=dict(color="gray", dash="dash")
        )
        
        fig.update_layout(
            xaxis_title="Terminal Price (x)",
            yaxis_title="Total Payoff",
            title="Portfolio Payoff",
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(
                tickformat=',d',
                dtick=1000
            ),
            yaxis=dict(
                tickformat='.2f',
                dtick=0.2
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display weight allocation table
        st.markdown("### Weight Allocation Summary")
        weights_data = {
            'Position': ['Underlying'] + [f'Binary_K{k}' for k in k_values],
            'Weight': [w_asset] + w_bins,
            'Abs Weight': [abs(w_asset)] + [abs(w) for w in w_bins]
        }
        weights_df = pd.DataFrame(weights_data)
        weights_df.loc['Total'] = ['', sum(weights_df['Weight']), sum(weights_df['Abs Weight'])]
        st.dataframe(weights_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")