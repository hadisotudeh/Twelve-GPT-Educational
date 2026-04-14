from abc import ABC, abstractmethod
from typing import List, Union, Dict

import pandas as pd

from openai import OpenAI


import utils.sentences as sentences
from utils.gemini import convert_messages_format

from classes.data_point import Player, Country, Person
from classes.data_source import PersonStat

import json

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import (
        GPT_BASE,
        GPT_KEY,
        GPT_CHAT_MODEL,
        GPT_SUPPORTS_REASONING,
        GPT_AVAILABLE_REASONING_EFFORTS,
        GPT_SUPPORTS_TEMPERATURE,
    )

import streamlit as st



class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def get_messages_from_excel(
        self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  
            print(e)
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
        return messages

    def stream_gpt(self, temperature=1, reasoning_effort=None, stream=False):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """

        st.expander("Chat transcript", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_0.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        stream=True,
                    )

                def streamed_chunks():
                    for event in response_stream:
                        if event.type == "response.output_text.delta":
                            yield event.delta

                answer = streamed_chunks()
            else:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                    )

                answer = response.output_text

        return answer


class PlayerDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football football data journalist. "
                    "You provide to the point, engaging, and accurate explanations about players using data. "
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's tactical positional behavior. "
            f"The first sentence should use describe players full name and tactical position (no accronym) and number of matches. "
            "The second sentence should describe the player's in-possesoin KPIs. "
            "The third sentence should describe the player's out-of-possesoin KPIs. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

        # read data/wvs/intermediate_data/relevant_questions.json
        with open("data/wvs/intermediate_data/relevant_questions.json", "r") as f:
            self.relevant_questions = json.load(f)

        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using social factors derived from the World Value Survey. "
                    "You use the information given to you to answer questions about how countries score in various social factors that attempt to measure the social values held by the population of a country."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        description = f"Here is a statistical description of the societal values of {self.country.name.capitalize()}."

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in self.country.relevant_metrics:

            description += f"\n\nAccording to the WVS, {self.country.name.capitalize()} was found to "
            description += sentences.describe_level(
                self.country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same wave. "

            if metric in self.country.drill_down_metrics:

                if self.country.ser_metrics[metric + "_Z"] > 0:
                    index = 1
                else:
                    index = 0

                question, value = self.country.drill_down_metrics[metric]
                question, value = question[index], value[index]
                description += "In response to the question '"
                description += self.relevant_questions[metric][question][0]
                description += "', on average participants "
                description += self.relevant_questions[metric][question][1]
                description += " '"
                description += self.relevant_questions[metric][question][2][str(value)]
                description += "' "
                description += self.relevant_questions[metric][question][3]
                description += ". "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 2 short paragraph summary of the social values held by population of the country. "
            f"The first paragraph should focus on any factors or values for which the country is above or bellow average. If the country is neither above nor below average in any values, mention that. "
            f"The remaining paragraph should mention any specific values or factors that are neither high nor low compared to the average. "
        )
        return [{"role": "user", "content": prompt}]


class PersonDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward_bigfive.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward_bigfive.xlsx"]

    def __init__(self, person: Person):
        self.person = person
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a recruiter. "
                    "You provide succinct and to the point explanations about a candidate using data.  "
                    "You use the information given to you from the data and answers"
                    "to earlier user/assistant pairs to give summaries of candidates."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the candidate as a candidate or a person?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the candidate as a person. "
                    "When I say candidate, I mean person. "
                    "But I always talk about the candidate, as a person."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a candidate for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def categorie_description(self, value):
        if value <= -2:
            return "The candidate is extremely "
        elif -2 < value <= -1:
            return "The candidate is very "
        elif -1 < value <= -0.5:
            return "The candidate is quite "
        elif -0.5 < value <= 0.5:
            return "The candidate is relatively "
        elif 0.5 < value <= 1:
            return "The candidate is quite "
        elif 1 < value <= 2:
            return "The candidate is very "
        else:
            return "The candidate is extremely "

    def all_max_indices(self, row):
        max_value = row.max()
        return list(row[row == max_value].index)

    def all_min_indices(self, row):
        min_value = row.min()
        return list(row[row == min_value].index)

    def get_description(self, person):
        # here we need the dataset to check the min and max score of the person

        person_metrics = person.ser_metrics
        person_stat = PersonStat()
        questions = person_stat.get_questions()

        name = person.name
        extraversion = person_metrics["extraversion_Z"]
        neuroticism = person_metrics["neuroticism_Z"]
        agreeableness = person_metrics["agreeableness_Z"]
        conscientiousness = person_metrics["conscientiousness_Z"]
        openness = person_metrics["openness_Z"]

        text = []

        # extraversion
        cat_0 = "solitary and reserved. "
        cat_1 = "outgoing and energetic. "

        if extraversion > 0:
            text_t = (self.categorie_description(extraversion) 
            + cat_1
            + "The candidate tends to be more social. "
                     )
            if extraversion > 1:
                index_max = person_metrics[0:10].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (self.categorie_description(extraversion) 
            + cat_0
            + "The candidate tends to be less social. "
                     )
            if extraversion < -1:
                index_min = person_metrics[0:10].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # neuroticism
        cat_0 = "resilient and confident. "
        cat_1 = "sensitive and nervous. "

        if neuroticism > 0:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_1
                + "The candidate tends to feel more negative emotions and anxiety. "
            )
            if neuroticism > 1:
                index_max = person_metrics[10:20].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_0
                + "The candidate tends to feel less negative emotions and anxiety. "
            )
            if neuroticism < -1:
                index_min = person_metrics[10:20].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # agreeableness
        cat_0 = "critical and rational. "
        cat_1 = "friendly and compassionate. "

        if agreeableness > 0:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_1
                + "The candidate tends to be more cooperative, polite, kind and friendly. "
            )
            if agreeableness > 1:
                index_max = person_metrics[20:30].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_0
                + "The candidate tends to be less cooperative, polite, kind and friendly. "
            )
            if agreeableness < -1:
                index_min = person_metrics[20:30].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # conscientiousness
        cat_0 = "extravagant and careless. "
        cat_1 = "efficient and organized. "

        if conscientiousness > 0:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_1
                + "The candidate tends to be more careful or diligent. "
            )
            if conscientiousness > 1:
                index_max = person_metrics[30:40].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_0
                + "The candidate tends to be less careful or diligent. "
            )
            if conscientiousness < -1:
                index_min = person_metrics[30:40].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # openness
        cat_0 = "consistent and cautious. "
        cat_1 = "inventive and curious. "

        if openness > 0:
            text_t = (
                self.categorie_description(openness)
                + cat_1
                + "The candidate tends to be more open to new ideas and experiences. "
            )
            if openness > 1:
                index_max = person_metrics[40:50].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(openness)
                + cat_0
                + "The candidate tends to be less open to new ideas and experiences. "
            )
            if openness < -1:
                index_min = person_metrics[40:50].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        text = "".join(text)
        text = text.replace(",", "")
        return text

    def synthesize_text(self):
        person = self.person
        metrics = self.person.ser_metrics
        description = self.get_description(person)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the person's personality, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the person. "
            "The second sentence should describe the person's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the person is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the person compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class PositionVersatilityDescription(Description):
    """
    Generates natural language descriptions of player position versatility KPIs.
    Includes position-specific analysis with match counts and role-specific performance.
    """

    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/PositionVersatility.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/PositionVersatility.xlsx"]

    def __init__(self, player_df, positions, player_name, player_team, stats):
        self.player_df = player_df
        self.positions = positions
        self.player_name = player_name
        self.player_team = player_team
        self.stats = stats
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Introduction messages for position versatility analysis.
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football position analyst. "
                    "You provide clear, data-driven explanations about players inferred positional behavior. "
                    "You note the number of matches a player operates in a position. "
                    "You use in-possession and out-of-possession versatility, lateral center of gravity, and vertical center of gravity metrics, which are in z-scores to explain how a player compares to peers in that KPI. "
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about positional versatility KPIs?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):
        """
        Convert KPI z-scores to natural language description, broken down by position.
        Generates flowing narrative text rather than bullet points.
        """
        kpi_columns = self.stats.kpi_columns
        pos_col = self.stats.pos_col

        description = f"Here is a statistical description of {self.player_name}, who played for {self.player_team}. "
        description += f"\n\n"

        # Analyze each position separately
        for position in self.positions:
            # Get data for this specific position
            position_player_df = self.player_df[self.player_df[pos_col] == position]
            n_matches = len(position_player_df)
            match_suffix = "match" if n_matches == 1 else "matches"
            
            # Get z-scores for this position
            position_z_data = position_player_df[
                [f"{k}_z" for k in kpi_columns]
            ].mean()

            # Build position-specific narrative description
            position_narratives = []

            # In-possession versatility
            z_in_vers = position_z_data["in_possession_versatility_z"]
            if z_in_vers > 0.5:
                position_narratives.append("appeared in a broad range of tactical positions in possession")
            elif z_in_vers < -0.5:
                position_narratives.append("focusd on specific tactical positions in possession")
            else:
                position_narratives.append("showed typical in-possession variety for this role")

            # Out-of-possession versatility
            z_out_vers = position_z_data["out_of_possession_versatility_z"]
            if z_out_vers > 0.5:
                position_narratives.append("covered many tactical positions when out of possession")
            elif z_out_vers < -0.5:
                position_narratives.append("focusd on specific tactical positions when out of possession")
            else:
                position_narratives.append("showed typical out-of-possession coverage")

            # In-possession vertical positioning
            z_in_vcog = position_z_data["in_possession_vertical_center_of_gravity_z"]
            if z_in_vcog > 0.5:
                position_narratives.append("appeared in higher tactical positions when in possession")
            elif z_in_vcog < -0.5:
                position_narratives.append("operated deeper in possession")
            else:
                position_narratives.append("held typical in-possession depth")

            # Out-of-possession vertical positioning
            z_out_vcog = position_z_data["out_of_possession_vertical_center_of_gravity_z"]
            if z_out_vcog > 0.5:
                position_narratives.append("defended relatively high")
            elif z_out_vcog < -0.5:
                position_narratives.append("defended from deeper positions")
            else:
                position_narratives.append("kept typical defensive depth")

            # In-possession lateral positioning
            z_in_lcog = position_z_data["in_possession_lateral_center_of_gravity_z"]
            if z_in_lcog > 1:
                position_narratives.append("leaned strongly to the right in possession")
            elif z_in_lcog > 0.3:
                position_narratives.append("favored the right lane in possession")
            elif z_in_lcog < -1:
                position_narratives.append("leaned strongly to the left in possession")
            elif z_in_lcog < -0.3:
                position_narratives.append("favored the left lane in possession")
            else:
                position_narratives.append("was laterally balanced in possession")

            # Out-of-possession lateral positioning
            z_out_lcog = position_z_data["out_of_possession_lateral_center_of_gravity_z"]
            if z_out_lcog > 1:
                position_narratives.append("defended predominantly on the right side")
            elif z_out_lcog > 0.3:
                position_narratives.append("defended slightly right of center")
            elif z_out_lcog < -1:
                position_narratives.append("defended predominantly on the left side")
            elif z_out_lcog < -0.3:
                position_narratives.append("defended slightly left of center")
            else:
                position_narratives.append("defended with central lateral balance")

            # Combine narratives into cohesive text
            position_text = f"**{position}** ({n_matches} {match_suffix}): In this role, {self.player_name} " + ", ".join(position_narratives).lower() + ". "
            description += position_text + "\n"

        return description

    def get_prompt_messages(self):
        """
        Task instructions for the LLM, emphasizing position-specific analysis.
        """
        prompt = (
            f"Write a brief football positional analysis of the selected player for fans in exactly 5 sentences maximum. "
            f"Use the data-driven style of a football data journalist — accessible, insightful, and entertaining for general audiences. "
            f"For players, just use their surname (E. Haaland to Haaland) when writing the text. "
            f"For a position, explain how {self.player_name}'s positional behavior stack up against peers in engaging language. "
            f"Avoid jargon and hellosiation and only rely on this data (KPIs) only."
            f"For the position, instead of using accronyms, use the full name of the position. For example, instead of saying 'CM' say 'central midfielder'. "
            f"Keep it simple langauge understandable for fans who are not native English speakers and avoid chit-chat."
        )
        return [{"role": "user", "content": prompt}]
