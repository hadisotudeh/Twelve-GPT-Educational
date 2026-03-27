"""
Position Scout page.
Analyzes player versatility across positions using modular classes.
"""

import streamlit as st
import fitz

from utils.page_components import add_common_page_elements
from classes.data_source import PositionVersatilityStats
from classes.visual import PositionVersatilityVisual
from classes.description import PositionVersatilityDescription


def draw_position_maps(player_df, pos_col, selected_player):
    """Display position maps from PDF for the selected player."""
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


def main():
    # ── page chrome ──────────────────────────────────────────────────────
    sidebar_container = add_common_page_elements()
    st.divider()

    # ── load data ────────────────────────────────────────────────────────
    stats = PositionVersatilityStats()
    visual = PositionVersatilityVisual()

    # ── sidebar filter ───────────────────────────────────────────────────
    with sidebar_container:
        player_names = sorted(stats.df["player_name"].unique())

        selected_player = st.selectbox(
            "Player",
            options=player_names,
            index=None,
            placeholder="Type to search...",
        )

    if selected_player is None:
        st.info("Select a player from the sidebar to get started.")
        st.stop()

    # ── get player data ──────────────────────────────────────────────────
    player_df, positions, player_team = stats.get_player_data(selected_player)

    if player_df is None:
        st.error(f"Player {selected_player} not found.")
        st.stop()

    # ── Get main (most frequent) position ────────────────────────────────
    main_pos = positions[0]  # positions are sorted by frequency (value_counts)
    main_positions = [main_pos]

    # ── KPI definitions in one clean dropdown ────────────────────────────
    with st.expander("KPI definitions", expanded=False):
        st.markdown(visual.get_kpi_definitions())

    # ── Distribution plot for main position only ────────────────────────
    st.subheader("Positional Distribution Analysis")
    df_pos = stats.get_position_data(main_pos)
    player_pos_df = player_df[player_df[stats.pos_col] == main_pos]

    fig = visual.create_position_kpi_plot(
        main_pos, selected_player, df_pos, player_pos_df, player_team
    )

    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")

    st.divider()

    # ── Generate natural language description for main position only ────
    st.subheader("Wordalisation")
    
    # Generate description
    description = PositionVersatilityDescription(
        player_df=player_df,
        positions=main_positions,
        player_name=selected_player,
        player_team=player_team,
        stats=stats,
    )
    
    # Stream the GPT summary
    summary = description.stream_gpt(stream=True)
    st.write(summary)
    
    st.divider()

    # ── Position maps from cases.pdf ─────────────────────────────────────
    st.subheader(f"Position maps – {selected_player}")
    draw_position_maps(player_df, stats.pos_col, selected_player)


if __name__ == "__main__":
    main()
