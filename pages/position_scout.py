"""
Position Scout page.
Reads player_versatility.csv, lets the user pick a player,
and shows distribution plots of that player's versatility value
compared to all other players and to position groups.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import fitz

from utils.page_components import add_common_page_elements

# ── Twelve colour palette (mirrors classes/visual.py) ────────────────────
DARK_GREEN = "rgba(0,44,28,1)"
MEDIUM_GREEN = "rgba(0,56,33,1)"
BRIGHT_GREEN = "rgba(0,169,56,{a})"
WHITE = "rgba(255,255,255,{a})"
BRIGHT_YELLOW = "rgba(255,204,0,{a})"
FONT_TITLE = "Gilroy-Medium"
FONT_BODY = "Gilroy-Light"


def _base_layout(fig, height=500):
    """Apply the shared Twelve dark-green layout to a figure."""
    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(l=60, r=60, b=70, t=75, pad=16),
        paper_bgcolor=DARK_GREEN,
        plot_bgcolor=DARK_GREEN,
        legend=dict(
            orientation="h",
            font=dict(color=WHITE.format(a=1), family=FONT_BODY, size=11),
            itemclick=False,
            itemdoubleclick=False,
            x=0.5, xanchor="center",
            y=-0.2, yanchor="bottom",
            valign="middle",
        ),
        xaxis=dict(
            tickfont=dict(color=WHITE.format(a=0.5), family=FONT_BODY, size=12),
        ),
        yaxis=dict(
            tickfont=dict(color=WHITE.format(a=0.5), family=FONT_BODY, size=12),
        ),
    )


def _add_title(fig, title, subtitle):
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:15px'>{title}</span><br>{subtitle}",
            font=dict(family=FONT_TITLE, color=WHITE.format(a=1), size=12),
            x=0.05, xanchor="left", y=0.93, yanchor="top",
        ),
    )


# ── page chrome ──────────────────────────────────────────────────────────
sidebar_container = add_common_page_elements()

st.divider()

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_csv("data/position_maps/player_versatility.csv")

KPI_COLUMNS = [
    "Versatility",
    "Vertical Center of Gravity",
    "Lateral Center of Gravity",
    "Vertical Range",
    "Lateral Range",
]

KPI_AXIS_LABELS = {
    "Versatility": ("← More specialised", "More versatile →"),
    "Vertical Center of Gravity": ("← Deeper / defensive", "Higher / attacking →"),
    "Lateral Center of Gravity": ("← More left", "More right →"),
    "Vertical Range": ("← Less depth covered", "More depth covered →"),
    "Lateral Range": ("← Less width covered", "More width covered →"),
}

KPI_DESCRIPTIONS = {
    "Versatility": r"""
**Versatility** measures how spread out a player's positioning is across the pitch.
It is the **Shannon entropy** of the 25-zone position profile.

The pitch is divided into a 5×5 grid (5 rows from defence to attack, 5 columns from left to right).
Each zone $i$ has a value $p_i$ — the fraction of time the player spent there.

$$H = -\sum_{i=1}^{25} p_i \, \ln(p_i)$$

| z-score | Interpretation | Example |
|---|---|---|
| **z ≈ 0** | Average positional spread | A centre-back who occasionally steps into midfield |
| **z > 0** | More versatile — occupied many zones | A box-to-box midfielder who drops deep to collect the ball and also pushes into the final third |
| **z < 0** | More specialised — concentrated in fewer zones | A target striker who stays almost exclusively in the centre-forward zone |
""",
    "Vertical Center of Gravity": r"""
**Vertical Center of Gravity** is the average depth at which a player positions themselves on the pitch.

Each of the 5 rows in the grid has a row number (1 = deepest/defensive, 5 = highest/attacking).
The metric is the weighted mean of these row numbers, where the weight is $p_i$ (fraction of time in that zone):

$$\text{Vertical COG} = \sum_{i=1}^{25} p_i \cdot \text{row}_i$$

| z-score | Interpretation | Example |
|---|---|---|
| **z > 0** | Positioned higher / more attacking than peers in the same role | A left-back who pushes up to the wing like an auxiliary winger |
| **z < 0** | Positioned deeper / more defensive than peers in the same role | A defensive midfielder who sits right in front of the centre-backs instead of advancing |
""",
    "Lateral Center of Gravity": r"""
**Lateral Center of Gravity** is the average left-right position of a player on the pitch.

Each of the 5 columns in the grid has a column number (1 = far left, 5 = far right).
The metric is the weighted mean of these column numbers, where the weight is $p_i$ (fraction of time in that zone):

$$\text{Lateral COG} = \sum_{i=1}^{25} p_i \cdot \text{col}_i$$

| z-score | Interpretation | Example |
|---|---|---|
| **z > 0** | Biased more to the right than peers in the same role | A centre-mid who consistently drifts to the right half-space |
| **z < 0** | Biased more to the left than peers in the same role | A striker who favours the left channel rather than staying central |
""",
    "Vertical Range": r"""
**Vertical Range** measures how much ground a player covers along the depth axis (defence ↔ attack).

It is the weighted standard deviation of the row numbers, where the weight is $p_i$ (fraction of time in that zone)
and $\text{Vertical COG}$ is the player's average row position:

$$\text{Vertical Range} = \sqrt{\sum_{i=1}^{25} p_i \,(\text{row}_i - \text{Vertical COG})^2}$$

| z-score | Interpretation | Example |
|---|---|---|
| **z > 0** | Covers more depth than peers | A wing-back who tracks back to their own box and also arrives in the opposition box |
| **z < 0** | Covers less depth than peers | A centre-back who holds a strict line and rarely ventures beyond the halfway mark |
""",
    "Lateral Range": r"""
**Lateral Range** measures how much ground a player covers along the width axis (left ↔ right).

It is the weighted standard deviation of the column numbers, where the weight is $p_i$ (fraction of time in that zone)
and $\text{Lateral COG}$ is the player's average column position:

$$\text{Lateral Range} = \sqrt{\sum_{i=1}^{25} p_i \,(\text{col}_i - \text{Lateral COG})^2}$$

| z-score | Interpretation | Example |
|---|---|---|
| **z > 0** | Covers more width than peers | A winger who drifts inside to the half-space and also hugs the touchline |
| **z < 0** | Covers less width than peers | A central midfielder who stays narrow through the middle of the pitch |
""",
}

KPI_DEFINITIONS_MARKDOWN = "\n\n---\n\n".join(
    [f"### {kpi}\n\n{KPI_DESCRIPTIONS[kpi].strip()}" for kpi in KPI_COLUMNS]
)

# ── sidebar filter ───────────────────────────────────────────────────────
with sidebar_container:
    position_type = st.radio(
        "Position type",
        options=["SkillCorner", "Ours"],
        index=0,
        horizontal=True,
    )
    pos_col = "skillcorner_position" if position_type == "SkillCorner" else "position"

    team_names = sorted(df["team_name"].unique())
    selected_team = st.selectbox(
        "Team",
        options=team_names,
        index=None,
        placeholder="Type to search...",
    )

    if selected_team is not None:
        player_names = sorted(df[df["team_name"] == selected_team]["player_name"].unique())
    else:
        player_names = sorted(df["player_name"].unique())

    selected_player = st.selectbox(
        "Player",
        options=player_names,
        index=None,
        placeholder="Type to search...",
    )

if selected_player is None:
    st.info("Select a player from the sidebar to get started.")
    st.stop()

# ── z-score each KPI within position group ───────────────────────────────
for kpi in KPI_COLUMNS:
    group_mean = df.groupby(pos_col)[kpi].transform("mean")
    group_std = df.groupby(pos_col)[kpi].transform("std")
    df[f"{kpi}_z"] = (df[kpi] - group_mean) / group_std

# All rows for the selected player (one per match)
player_df = df[df["player_name"] == selected_player]
player_positions = player_df[pos_col].value_counts().index.tolist()
player_team = player_df.iloc[0]["team_name"]

# Only keep positions the selected player has appeared in
positions = player_positions
df_filtered = df[df[pos_col].isin(positions)]

# ── KPI definitions in one clean dropdown ────────────────────────────────
with st.expander("KPI definitions", expanded=False):
    st.markdown(KPI_DEFINITIONS_MARKDOWN)

# ── One distribution plot per position (all KPIs in each figure) ─────────
for pos in positions:
    fig_pos = go.Figure()
    df_pos = df[df[pos_col] == pos]

    # Aggregate to one dot per player (mean z-score within this position)
    df_agg = df_pos.groupby("player_name", as_index=False)[[f"{k}_z" for k in KPI_COLUMNS]].mean()

    # Other players by KPI row
    showlegend_group = True
    for i, kpi in enumerate(KPI_COLUMNS):
        z_col = f"{kpi}_z"
        fig_pos.add_trace(
            go.Scatter(
                x=df_agg[z_col].tolist(),
                y=[i] * len(df_agg),
                mode="markers",
                marker=dict(
                    color=BRIGHT_GREEN.format(a=0.2),
                    size=10,
                    line_width=1.5,
                    line_color=BRIGHT_GREEN.format(a=1),
                ),
                text=df_agg["player_name"],
                hovertemplate="%{text}<br>" + kpi + " z: %{x:.2f}<extra></extra>",
                name="Other players  ",
                showlegend=showlegend_group,
            )
        )
        showlegend_group = False

    # Selected player marker per KPI
    player_pos_df = player_df[player_df[pos_col] == pos]
    player_agg = player_pos_df[[f"{k}_z" for k in KPI_COLUMNS]].mean()
    showlegend_player = True
    for i, kpi in enumerate(KPI_COLUMNS):
        z_col = f"{kpi}_z"
        fig_pos.add_trace(
            go.Scatter(
                x=[player_agg[z_col]],
                y=[i],
                mode="markers",
                marker=dict(
                    color=WHITE.format(a=0.5),
                    size=10,
                    symbol="square",
                    line_width=1.5,
                    line_color=WHITE.format(a=1),
                ),
                text=[selected_player],
                hovertemplate="%{text}<br>" + kpi + " z: %{x:.2f}<extra></extra>",
                name=selected_player,
                showlegend=showlegend_player,
            )
        )
        showlegend_player = False

    _base_layout(fig_pos, height=max(360, len(KPI_COLUMNS) * 70))

    n_matches = len(player_pos_df)
    suffix = "match" if n_matches == 1 else "matches"
    _add_title(
        fig_pos,
        f"{selected_player} at {pos} – KPI distribution (z-scores)",
        f"{player_team} · {n_matches} {suffix} in this position",
    )

    all_z = df_pos[[f"{k}_z" for k in KPI_COLUMNS]].stack().dropna()
    x_min = all_z.min() - 0.5
    x_max = all_z.max() + 0.5
    fig_pos.update_xaxes(
        range=[x_min, x_max],
        fixedrange=True,
        title=dict(
            text="Lower relative to position peers     ·     z-score     ·     Higher relative to position peers",
            font=dict(color=WHITE.format(a=0.6), family=FONT_BODY, size=11),
        ),
    )
    fig_pos.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(KPI_COLUMNS))),
        ticktext=KPI_COLUMNS,
        fixedrange=True,
        gridcolor=MEDIUM_GREEN,
        zerolinecolor=MEDIUM_GREEN,
    )
    fig_pos.add_shape(
        type="line", x0=0, y0=-0.5, x1=0, y1=len(KPI_COLUMNS) - 0.5,
        line=dict(color="gray", width=1, dash="dot"),
    )

    st.plotly_chart(fig_pos, config={"displayModeBar": False}, use_container_width=True)

st.divider()

# ── KPI correlation matrix ──────────────────────────────────────────────
st.subheader("KPI correlation matrix")
corr = df[KPI_COLUMNS].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=KPI_COLUMNS,
    y=KPI_COLUMNS,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=12, color=WHITE.format(a=1)),
    colorscale=[[0, DARK_GREEN], [0.5, MEDIUM_GREEN], [1, BRIGHT_GREEN.format(a=1)]],
    zmin=-1, zmax=1,
    hovertemplate="%{x} vs %{y}<br>r = %{z:.2f}<extra></extra>",
    showscale=False,
))
_base_layout(fig_corr, height=450)
fig_corr.update_layout(
    margin=dict(l=160, r=60, b=120, t=50, pad=16),
    xaxis=dict(tickangle=-45),
)

st.plotly_chart(fig_corr, config={"displayModeBar": False}, use_container_width=True)

st.divider()

# ── Position maps from cases.pdf ────────────────────────────────────────
st.subheader(f"Position maps – {selected_player}")

# The CSV row index corresponds 1:1 to the PDF page index
player_sorted = player_df.sort_values("match_date")
player_indices = player_sorted.index.tolist()

doc = fitz.open("data/position_maps/cases.pdf")
cols = st.columns(4)
for col_idx, idx in enumerate(player_indices):
    page = doc[idx]
    # Crop to the 5x5 grid only (skip title at top, subtitle at bottom)
    clip = fitz.Rect(0, 28, page.rect.width, 310)
    pix = page.get_pixmap(dpi=100, clip=clip)
    img_bytes = pix.tobytes("png")
    match_date = player_sorted.loc[idx, "match_date"]
    match_pos = player_sorted.loc[idx, pos_col]
    col = cols[col_idx % 4]
    col.image(img_bytes)
    col.markdown(
        f"<p style='text-align:center; font-size:18px; color:gray;'>{match_pos} – {match_date}</p>",
        unsafe_allow_html=True,
    )
doc.close()
