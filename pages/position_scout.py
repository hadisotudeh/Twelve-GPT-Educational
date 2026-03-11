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

st.expander("How is versatility computed?", expanded=False).markdown(r"""
**Versatility** measures how spread out a player's positioning is across the pitch
during a match. It is derived from the **Shannon entropy** of the position_profile,
a vector of 25 values that represent the fraction of time a player spent in each inferred tactical position
:

$$
H = -\sum_{i=1}^{25} p_i \, \ln(p_i)
$$

where $p_i$ is the proportion of time spent in position $i$.

The raw entropy $H$ is then **z-scored** across all player-match observations to
produce the versatility value shown in the plots:

$$
\text{versatility} = \frac{H - \mu_H}{\sigma_H}
$$

| Value | Interpretation |
|---|---|
| **versatility ≈ 0** | Average positional spread |
| **versatility > 0** | More versatile than average (occupied many zones) |
| **versatility < 0** | More specialised than average (concentrated in fewer zones) |

Because the dataset contains one row per player per match, the same player can
show different versatility values depending on the tactical setup of each game.
""")

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_csv("data/position_maps/player_versatility.csv")

# ── sidebar filter ───────────────────────────────────────────────────────
with sidebar_container:
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

# All rows for the selected player (one per match)
player_df = df[df["player_name"] == selected_player]
player_positions = sorted(player_df["skillcorner_position"].unique())
player_team = player_df.iloc[0]["team_name"]

# Average versatility across matches for the selected player
player_versatility = player_df["versatility"].mean()

# Only keep positions the selected player has appeared in
positions = player_positions
df_filtered = df[df["skillcorner_position"].isin(positions)]

# ── Distribution by position group (dot-strip per position) ─────────────
fig_pos = go.Figure()

# All players per position row
showlegend_group = True
for i, pos in enumerate(positions):
    pos_df = df_filtered[df_filtered["skillcorner_position"] == pos]
    fig_pos.add_trace(
        go.Scatter(
            x=pos_df["versatility"].tolist(),
            y=[i] * len(pos_df),
            mode="markers",
            marker=dict(
                color=BRIGHT_GREEN.format(a=0.2),
                size=10,
                line_width=1.5,
                line_color=BRIGHT_GREEN.format(a=1),
            ),
            text=pos_df["player_name"],
            hovertemplate="%{text}<br>Versatility: %{x:.2f}<extra></extra>",
            name="Other players  ",
            showlegend=showlegend_group,
        )
    )
    showlegend_group = False

# Selected player markers – one per position they played
showlegend_player = True
for i, pos in enumerate(positions):
    player_pos_df = player_df[player_df["skillcorner_position"] == pos]
    if player_pos_df.empty:
        continue
    fig_pos.add_trace(
        go.Scatter(
            x=player_pos_df["versatility"].tolist(),
            y=[i] * len(player_pos_df),
            mode="markers",
            marker=dict(
                color=WHITE.format(a=0.5),
                size=10,
                symbol="square",
                line_width=1.5,
                line_color=WHITE.format(a=1),
            ),
            text=[selected_player] * len(player_pos_df),
            hovertemplate="%{text}<br>Versatility: %{x:.2f}<extra></extra>",
            name=selected_player,
            showlegend=showlegend_player,
        )
    )
    showlegend_player = False

_base_layout(fig_pos, height=max(300, len(positions) * 55))
_add_title(fig_pos, f"Versatility by position – {selected_player}",
           f"{player_team} · Positions played: {', '.join(positions)}")

fig_pos.update_xaxes(
    range=[df_filtered["versatility"].min() - 0.5,
           df_filtered["versatility"].max() + 0.5],
    fixedrange=True,
)
fig_pos.update_yaxes(
    tickmode="array",
    tickvals=list(range(len(positions))),
    ticktext=positions,
    fixedrange=True,
    gridcolor=MEDIUM_GREEN,
    zerolinecolor=MEDIUM_GREEN,
)
fig_pos.add_shape(
    type="line", x0=0, y0=-0.5, x1=0, y1=len(positions) - 0.5,
    line=dict(color="gray", width=1, dash="dot"),
)

st.plotly_chart(fig_pos, config={"displayModeBar": False}, use_container_width=True)

st.divider()

# ── Position maps from cases.pdf ────────────────────────────────────────
st.subheader(f"Position maps – {selected_player}")

# The CSV row index corresponds 1:1 to the PDF page index
player_indices = player_df.index.tolist()

doc = fitz.open("data/position_maps/cases.pdf")
for idx in player_indices:
    page = doc[idx]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    match_date = player_df.loc[idx, "match_date"]
    match_pos = player_df.loc[idx, "skillcorner_position"]
    st.image(img_bytes, caption=f"{selected_player} – {match_pos} – {match_date}")
doc.close()
