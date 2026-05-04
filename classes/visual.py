import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s


class Visual:
    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")

    def __init__(self, pdf=False, plot_type="scout"):
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.0
        self.fig = go.Figure()
        self._setup_styles()
        self.plot_type = plot_type

        if plot_type == "scout":
            self.annotation_text = (
                "<span style=''>{metric_name}: {data:.2f} per 90</span>"
            )
        else:
            # self.annotation_text = "<span style=''>{metric_name}: {data:.0f}/66</span>"  # TODO: this text will not automatically update!
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            width="stretch",
        )

    def _setup_styles(self):
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            width="stretch",
        )

    def close(self):
        pass


class DistributionPlot(Visual):
    def __init__(self, columns, labels=None, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

        # Add a vertical line at x=0
        self.fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=len(self.columns),
            line=dict(color="gray", width=1, dash="dot"),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):

        for i, col in enumerate(self.columns):
            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots].tolist(), 
                    y=list(np.ones(len(df_plot[col + plots])) * i),
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.dark_green, opacity=0.2),
                        "size": 10,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(self.bright_green),
                    },
                    hovertemplate="%{text}<br>" + hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover].tolist(),
                    showlegend=False,
                )
            )
            

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=self.annotation_text.format(
                    metric_name=metric_name,
                    data=(
                        ser_plot[col]
                        # if self.plot_type == "scout"
                        # else ser_plot[col + hover]
                    ),
                ),
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )


    def add_player(self, player: Union[Player, Country], n_group, metrics):

        # # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    # def add_players(self, players: PlayerStats, metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_group_data(
    #         df_plot=players.df,
    #         plots="_Z",
    #         names=players.df["player_name"],
    #         hover="_Ranks",
    #         hover_string="Rank: %{customdata}/" + str(len(players.df)),
    #         legend=f"Other players  ",  # space at end is important
    #     )

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


class DistributionPlotPersonality(Visual):
    def __init__(self, columns, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=["Worse", "Average", "Better"],
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=round(df_plot[col + hover]),
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[round(ser_plot[col + hover])],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_person(self, person: Person, n_group, metrics):
        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_data_point(
            ser_plot=person.ser_metrics,
            plots="_Z",
            name=person.name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    def add_persons(self, persons: PersonStat, metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_group_data(
            df_plot=persons.df,
            plots="_Z",
            names=persons.df["name"],
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(len(persons.df)),
            legend=f"Other persons  ",
        )

    def add_title_from_person(self, person: Person):
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)


"""class ViolinPlot(Visual):
    def violin(data, point_data):
        # Create a figure object
        fig = go.Figure()

        # Labels for the columnshover
        labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

        # Loop through each label to add a violin plot trace
        for label in labels:
            fig.add_trace(go.Violin(
                x=df_plot[label],  # Use x for the data
                name=label,      # Label each violin plot correctly
                box_visible=True,
                meanline_visible=True,
                line_color='black',  # Color of the violin outline
                fillcolor='rgba(0,100,200,0.3)',  # Color of the violin fill
                opacity=0.6,
                orientation='h'  # Set orientation to horizontal
            )
        )
        for label, value in point_data.items():
            fig.add_trace(
                go.Scatter(x=[value], y=[label], mode='markers', marker=dict(color='red', size=8, symbol='cross'), name=f'{label} Candidate Point'))

        # Update layout for better visualization
        fig.update_layout(
            title='Distribution of Personality Traits',
            xaxis_title='Score',  
            yaxis_title='Trait',
            xaxis=dict(range=[0, 40]),
            violinmode='overlay', 
            showlegend=True)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


    def radarPlot(Visual):
        # Data import
        data_r = data_p.to_list()  
        labels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        df = pd.DataFrame({'data': data_r,'label': labels})
    
        # Create the radar plot
        fig = px.line_polar(df, r='data', theta='label', line_close=True, markers=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 40])),showlegend=True, title= 'Candidate profile')
        fig.update_traces(fill='toself', marker=dict(size=5))
        # Display the plot in Streamlit
        st.plotly_chart(fig)"""


class PositionVersatilityVisual(Visual):
    """
    Handles all visualizations for position versatility analysis.
    """

    # ── Twelve colour palette (mirrors classes/visual.py) ────────────────
    DARK_GREEN = "rgba(0,44,28,1)"
    MEDIUM_GREEN = "rgba(0,56,33,1)"
    BRIGHT_GREEN = "rgba(0,169,56,{a})"
    WHITE = "rgba(255,255,255,{a})"
    BRIGHT_YELLOW = "rgba(255,204,0,{a})"
    FONT_TITLE = "Gilroy-Medium"
    FONT_BODY = "Gilroy-Light"

    def __init__(self):
        super().__init__()
        self.kpi_columns = [
            "in_possession_versatility",
            "in_possession_vertical_center_of_gravity",
            "in_possession_lateral_center_of_gravity",
            "out_of_possession_versatility",
            "out_of_possession_vertical_center_of_gravity",
            "out_of_possession_lateral_center_of_gravity",
        ]
        self.kpi_label_map = {
            "in_possession_versatility": "In-possession versatility",
            "in_possession_vertical_center_of_gravity": "In-possession vertical COG",
            "in_possession_lateral_center_of_gravity": "In-possession lateral COG",
            "out_of_possession_versatility": "Out-of-possession versatility",
            "out_of_possession_vertical_center_of_gravity": "Out-of-possession vertical COG",
            "out_of_possession_lateral_center_of_gravity": "Out-of-possession lateral COG",
        }

    def _base_layout(self, fig, height=500):
        """Apply the shared Twelve dark-green layout to a figure."""
        fig.update_layout(
            autosize=True,
            height=height,
            margin=dict(l=60, r=170, b=70, t=90, pad=16),
            paper_bgcolor=self.DARK_GREEN,
            plot_bgcolor=self.DARK_GREEN,
            legend=dict(
                orientation="v",
                font=dict(color=self.WHITE.format(a=1), family=self.FONT_BODY, size=11),
                itemclick=False,
                itemdoubleclick=False,
                x=1.02,
                xanchor="left",
                y=1,
                yanchor="top",
                valign="middle",
            ),
            xaxis=dict(
                tickfont=dict(color=self.WHITE.format(a=0.5), family=self.FONT_BODY, size=12),
            ),
            yaxis=dict(
                tickfont=dict(color=self.WHITE.format(a=0.5), family=self.FONT_BODY, size=12),
            ),
        )

    def _add_title(self, fig, title, subtitle):
        """Add title and subtitle to figure."""
        fig.update_layout(
            title=dict(
                text=f"<span style='font-size:15px'>{title}</span><br>{subtitle}",
                font=dict(family=self.FONT_TITLE, color=self.WHITE.format(a=1), size=12),
                x=0.05,
                xanchor="left",
                y=0.93,
                yanchor="top",
            ),
        )

    def _position_group_label(self, position):
        position_labels = {
            "AM": "attacking midfielders",
            "CB": "centre backs",
            "CF": "centre forwards",
            "DM": "defensive midfielders",
            "LB": "left backs",
            "LCB": "left centre backs",
            "LDM": "left defensive midfielders",
            "LF": "left forwards",
            "LM": "left midfielders",
            "LW": "left wingers",
            "LWB": "left wing backs",
            "RB": "right backs",
            "RCB": "right centre backs",
            "RDM": "right defensive midfielders",
            "RF": "right forwards",
            "RM": "right midfielders",
            "RW": "right wingers",
            "RWB": "right wing backs",
        }
        return position_labels.get(position, f"{position} players")

    def create_position_kpi_plot(
        self,
        position: str,
        player_name: str,
        df_pos: pd.DataFrame,
        player_pos_df: pd.DataFrame,
        player_team: str,
        comparison_player_name: str = None,
        comparison_player_pos_df: pd.DataFrame = None,
        comparison_player_team: str = None,
        comparison_players: list = None,
    ):
        """
        Create KPI distribution plot for a position comparing selected player to peers.
        """
        self.fig = go.Figure()

        z_columns = [f"{k}_z" for k in self.kpi_columns]
        plot_columns = z_columns + ["average_kpi_z"]
        plot_labels = [self.kpi_label_map[kpi] for kpi in self.kpi_columns] + [
            "Average KPI z-score"
        ]

        # Aggregate to one dot per player (mean z-score within this position)
        df_agg = df_pos.groupby("player_name", as_index=False)[z_columns].mean()
        df_agg["average_kpi_z"] = df_agg[z_columns].mean(axis=1)

        # Other players by KPI row
        other_players_label = f"Other {self._position_group_label(position)}  "
        showlegend_group = True
        for i, z_col in enumerate(plot_columns):
            kpi_label = plot_labels[i]
            value_label = "score" if z_col == "average_kpi_z" else "z"
            self.fig.add_trace(
                go.Scatter(
                    x=df_agg[z_col].tolist(),
                    y=[i] * len(df_agg),
                    mode="markers",
                    marker=dict(
                        color=self.BRIGHT_GREEN.format(a=0.2),
                        size=10,
                        line_width=1.5,
                        line_color=self.BRIGHT_GREEN.format(a=1),
                    ),
                    text=df_agg["player_name"],
                    hovertemplate="%{text}<br>" + kpi_label + f" {value_label}: " + "%{x:.2f}<extra></extra>",
                    name=other_players_label,
                    showlegend=showlegend_group,
                )
            )
            showlegend_group = False

        # Selected player marker per KPI
        player_agg = player_pos_df[z_columns].mean()
        player_agg["average_kpi_z"] = player_agg[z_columns].mean()
        showlegend_player = True
        for i, z_col in enumerate(plot_columns):
            kpi_label = plot_labels[i]
            value_label = "score" if z_col == "average_kpi_z" else "z"
            self.fig.add_trace(
                go.Scatter(
                    x=[player_agg[z_col]],
                    y=[i],
                    mode="markers",
                    marker=dict(
                        color=self.WHITE.format(a=0.5),
                        size=10,
                        symbol="square",
                        line_width=1.5,
                        line_color=self.WHITE.format(a=1),
                    ),
                    text=[player_name],
                    hovertemplate="%{text}<br>" + kpi_label + f" {value_label}: " + "%{x:.2f}<extra></extra>",
                    name=player_name,
                    showlegend=showlegend_player,
                )
            )
            showlegend_player = False

        if comparison_players is None:
            comparison_players = []
            if (
                comparison_player_name
                and comparison_player_pos_df is not None
                and not comparison_player_pos_df.empty
            ):
                comparison_players.append(
                    {
                        "name": comparison_player_name,
                        "team": comparison_player_team,
                        "df": comparison_player_pos_df,
                    }
                )

        comparison_colors = [
            self.BRIGHT_YELLOW,
            "rgba(0,149,255,{a})",
            "rgba(255,75,0,{a})",
            "rgba(255,255,255,{a})",
            self.BRIGHT_GREEN,
        ]
        comparison_symbols = ["diamond", "circle", "triangle-up", "x", "star"]

        # Optional comparison player markers per KPI
        for comparison_index, comparison_player in enumerate(comparison_players):
            comparison_df = comparison_player["df"]
            if comparison_df is None or comparison_df.empty:
                continue

            comparison_name = comparison_player["name"]
            comparison_agg = comparison_df[z_columns].mean()
            comparison_agg["average_kpi_z"] = comparison_agg[z_columns].mean()
            comparison_color = comparison_colors[
                comparison_index % len(comparison_colors)
            ]
            comparison_symbol = comparison_symbols[
                comparison_index % len(comparison_symbols)
            ]
            showlegend_comparison = True
            for i, z_col in enumerate(plot_columns):
                kpi_label = plot_labels[i]
                value_label = "score" if z_col == "average_kpi_z" else "z"
                self.fig.add_trace(
                    go.Scatter(
                        x=[comparison_agg[z_col]],
                        y=[i],
                        mode="markers",
                        marker=dict(
                            color=comparison_color.format(a=0.65),
                            size=11,
                            symbol=comparison_symbol,
                            line_width=1.5,
                            line_color=comparison_color.format(a=1),
                        ),
                        text=[comparison_name],
                        hovertemplate="%{text}<br>" + kpi_label + f" {value_label}: " + "%{x:.2f}<extra></extra>",
                        name=comparison_name,
                        showlegend=showlegend_comparison,
                    )
                )
                showlegend_comparison = False

        self._base_layout(self.fig, height=max(420, len(plot_columns) * 70))

        n_matches = len(player_pos_df)
        suffix = "match" if n_matches == 1 else "matches"
        if comparison_players:
            comparison_names = ", ".join(
                comparison_player["name"] for comparison_player in comparison_players
            )
            title = f"{player_name} vs comparisons at {position}"
            subtitle = f"{player_team} · {n_matches} {suffix} | {comparison_names}"
        else:
            title = f"{player_name} at {position} – KPI distribution (z-scores)"
            subtitle = f"{player_team} · {n_matches} {suffix} in this position"

        self._add_title(self.fig, title, subtitle)

        self.fig.update_xaxes(
            autorange=True,
            fixedrange=False,
            title=dict(
                text="Lower relative to position peers     ·     z-score     ·     Higher relative to position peers",
                font=dict(color=self.WHITE.format(a=0.6), family=self.FONT_BODY, size=11),
            ),
        )
        self.fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(plot_columns))),
            ticktext=plot_labels,
            fixedrange=True,
            gridcolor=self.MEDIUM_GREEN,
            zerolinecolor=self.MEDIUM_GREEN,
        )
        self.fig.add_shape(
            type="line",
            x0=0,
            y0=-0.5,
            x1=0,
            y1=len(plot_columns) - 0.5,
            line=dict(color="gray", width=1, dash="dot"),
        )

        # Add horizontal separator lines between specific KPI rows
        # Separator between out-of-possession versatility and in-possession lateral COG (y = 2.5)
        # This sits between index 2 (in_possession_lateral_center_of_gravity) and index 3 (out_of_possession_versatility)
        if len(plot_columns) >= 4:
            self.fig.add_shape(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=2.5,
                y1=2.5,
                line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
            )

        # Separator between out-of-possession lateral COG and Average KPI (y = 5.5)
        # This sits between index 5 (out_of_possession_lateral_center_of_gravity) and index 6 (average_kpi_z)
        if len(plot_columns) >= 7:
            self.fig.add_shape(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=5.5,
                y1=5.5,
                line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
            )
        return self

    def get_kpi_definitions(self) -> str:
        """Get markdown string with all KPI definitions."""
        kpi_descriptions = {
            "in_possession_versatility": r"""
**In-possession versatility** measures how broadly a player occupies different zones while their team has the ball.

| Range | Interpretation |
|---|---|
| **z > 0.5** | Appears in a broad range of tactical positions in possession |
| **z < -0.5** | Focused on specific tactical positions in possession |
| **-0.5 ≤ z ≤ 0.5** | Shows typical in-possession variety for this role |
""",
            "in_possession_vertical_center_of_gravity": r"""
**In-possession vertical COG** is the average attacking depth of a player when their team has the ball.

| Range | Interpretation |
|---|---|
| **z > 0.5** | Appears in higher tactical positions when in possession |
| **z < -0.5** | Operates deeper in possession |
| **-0.5 ≤ z ≤ 0.5** | Holds typical in-possession depth |
""",
            "in_possession_lateral_center_of_gravity": r"""
**In-possession lateral COG** is the average left-right attacking lane when their team has the ball.

| Range | Interpretation |
|---|---|
| **z > 1** | Leans strongly to the right in possession |
| **0.3 < z ≤ 1** | Favors the right lane in possession |
| **-0.3 ≤ z ≤ 0.3** | Laterally balanced in possession |
| **-1 ≤ z < -0.3** | Favors the left lane in possession |
| **z < -1** | Leans strongly to the left in possession |
""",
            "out_of_possession_versatility": r"""
**Out-of-possession versatility** measures how broadly a player covers zones while defending.

| Range | Interpretation |
|---|---|
| **z > 0.5** | Covers many tactical positions when out of possession |
| **z < -0.5** | Focused on specific tactical positions when out of possession |
| **-0.5 ≤ z ≤ 0.5** | Shows typical out-of-possession coverage |
""",
            "out_of_possession_vertical_center_of_gravity": r"""
**Out-of-possession vertical COG** is the average defensive depth when the team does not have the ball.

| Range | Interpretation |
|---|---|
| **z > 0.5** | Defends relatively high |
| **z < -0.5** | Defends from deeper positions |
| **-0.5 ≤ z ≤ 0.5** | Keeps typical defensive depth |
""",
            "out_of_possession_lateral_center_of_gravity": r"""
**Out-of-possession lateral COG** is the average left-right defensive lane while out of possession.

| Range | Interpretation |
|---|---|
| **z > 1** | Defends predominantly on the right side |
| **0.3 < z ≤ 1** | Defends slightly right of center |
| **-0.3 ≤ z ≤ 0.3** | Defends with central lateral balance |
| **-1 ≤ z < -0.3** | Defends slightly left of center |
| **z < -1** | Defends predominantly on the left side |
""",
        }

        return "\n\n---\n\n".join(
            [
                f"### {self.kpi_label_map[kpi]}\n\n{kpi_descriptions[kpi].strip()}"
                for kpi in self.kpi_columns
            ]
        )
