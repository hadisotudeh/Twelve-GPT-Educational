import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import numpy as np
import json

from settings import USE_GEMINI, USE_LM_STUDIO

if USE_GEMINI:
    from settings import GEMINI_API_KEY, GEMINI_CHAT_MODEL
elif USE_LM_STUDIO:
    from settings import LM_STUDIO_API_KEY, LM_STUDIO_CHAT_MODEL, LM_STUDIO_API_BASE
else:
    from settings import (
        GPT_BASE,
        GPT_KEY,
        GPT_CHAT_MODEL,
        GPT_SUPPORTS_REASONING,
        GPT_AVAILABLE_REASONING_EFFORTS,
        GPT_SUPPORTS_TEMPERATURE,
    )

from classes.description import (
    PlayerDescription,
    CountryDescription,
    PersonDescription,
)
from classes.embeddings import PlayerEmbeddings, CountryEmbeddings, PersonEmbeddings

from classes.visual import Visual, DistributionPlot, DistributionPlotPersonality

import utils.sentences as sentences
from utils.gemini import convert_messages_format


class Chat:
    function_names = []

    def __init__(self, chat_state_hash, state="empty"):

        if (
            "chat_state_hash" not in st.session_state
            or chat_state_hash != st.session_state.chat_state_hash
        ):
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state
        if isinstance(self, PlayerChat):
            self.name = self.player.name
        elif isinstance(self, PersonChat):
            self.name = self.person.name
        else:
            pass

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state

    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible=True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    # def get_input(self):
    #     """
    #     Get input from streamlit."""

    #     if x := st.chat_input(
    #         placeholder=f"What else would you like to know about {self.player.name}?"
    #     ):
    #         if len(x) > 500:
    #             st.error(
    #                 f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
    #             )

    #         self.handle_input(x)

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages.
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevant information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})

        messages.append(
            {
                "role": "user",
                "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```",
            }
        )

        # Remove all items in messages where content is not a string
        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        # Show the messages in an expander
        st.expander("Chat transcript", expanded=False).write(messages)

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_1.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        elif USE_LM_STUDIO:
            client = OpenAI(api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_API_BASE)
            if stream:
                # Collect chunks eagerly so the generator over the list is
                # near-instantaneous — preventing Streamlit re-runs from
                # hitting the same generator while it is still executing.
                chunks = [
                    chunk.choices[0].delta.content
                    for chunk in client.chat.completions.create(
                        model=LM_STUDIO_CHAT_MODEL,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                    )
                    if chunk.choices and chunk.choices[0].delta.content
                ]

                def streamed_chunks():
                    yield from chunks

                answer = streamed_chunks()
            else:
                response = client.chat.completions.create(
                    model=LM_STUDIO_CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                answer = response.choices[0].message.content
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
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
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                    )

                answer = response.output_text
        message = {"role": "assistant", "content": answer}

        # Add the returned value to the messages.
        self.messages_to_display.append(message)

    def display_content(self, content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try:
                content.show()
            except:
                try:
                    st.write(content.get_string())
                except:
                    raise ValueError(
                        f"Message content of type {type(content)} not supported."
                    )

    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns.
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        # st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message_block = st.chat_message(name=key, avatar=avatar)
            with message_block:
                for message in group:
                    content = message["content"]
                    if isinstance(content, GeneratorType):
                        final_text = st.write_stream(content)
                        message["content"] = final_text
                    else:
                        self.display_content(content)

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state


class PlayerChat(Chat):
    tools = [
        {
            "type": "function",
            "name": "get_player_summary",
            "description": "Returns a data-driven statistical summary of the selected player.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "search_football_knowledge",
            "description": "Searches a knowledge base for information relevant to a question about data analytics in football, especially about forwards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    ]

    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

    def _get_player_summary(self):
        return PlayerDescription(self.player).synthesize_text()

    def _search_knowledge(self, query):
        results = self.embeddings.search(query, top_n=5)
        return "\n".join(results["assistant"].to_list())

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.player.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        if USE_GEMINI or USE_LM_STUDIO:
            first_messages = [
            {"role": "system", "content": "You are a UK-based football scout."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a football scouting platform. "
                    f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
            return first_messages
        else:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a UK-based football scout. "
                        f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                        "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                        "Choose the tool that best fits the user's query to respond."
                        "- If the user is asking for information about the player, use the get_player_summary function. "  
                        "- If the user is asking for general football knowledge, use the search_football_knowledge function. "
                        "- If none of the tools are relevant to the user's query, respond directly to the user that the question is outside your scope. "
                        "- If the user asks about a different player, respond that you can only answer questions about the selected player and if they want information about a different player, they need to select that player first on the sidebar."
                        "All user messages will be prefixed with 'User:' and enclosed with ```. "
                        "When responding to the user, speak directly to them. "
                        "Use the information provided before the query to provide 2 sentence answers."
                        "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                    ),
                }
            ]

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        if USE_GEMINI or USE_LM_STUDIO:
            super().handle_input(input, reasoning_effort=reasoning_effort, temperature=temperature, stream=stream)
            return
        # OpenAI function-calling path
        messages = self.instruction_messages()
        messages = messages + self.messages_to_display.copy()
        messages = [m for m in messages if isinstance(m["content"], str)]
        messages.append({"role": "user", "content": f"```User: {input}```"})

        self.messages_to_display.append({"role": "user", "content": input})

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        # Call 1: model picks a tool if relevant, or answers directly if not
        r1 = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        fc = next((item for item in r1.output if item.type == "function_call"), None)

        if fc is None:
            # Model decided no tool was needed — use its response directly
            st.expander("Chat transcript", expanded=False).write(
                [{"role": m.get("role"), "content": m.get("content", "")} for m in messages if isinstance(m, dict)]
            )
            self.messages_to_display.append({"role": "assistant", "content": r1.output_text})
            return

        if fc.name == "get_player_summary":
            result = self._get_player_summary()
        else:
            result = self._search_knowledge(json.loads(fc.arguments)["query"])

        # Call 2: final answer, no more tools
        tool_inputs = list(messages) + list(r1.output) + [
            {"type": "function_call_output", "call_id": fc.call_id, "output": result}
        ]

        formatted = []
        for item in tool_inputs:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    formatted.append({"tool_result": item["output"] or "(empty)", "call_id": item["call_id"]})
                else:
                    formatted.append({"role": item.get("role"), "content": item.get("content", "")})
            elif hasattr(item, "type"):
                if item.type == "function_call":
                    formatted.append({"tool_call": item.name, "arguments": json.loads(item.arguments)})
                # reasoning items are skipped
        st.expander("Chat transcript", expanded=False).write(formatted)
       
        if stream:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                    stream=True,
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                    stream=True,
                )
            else:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
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
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                )
            else:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                )
            answer = response.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_relevant_info(self, query):
        # Used by the Gemini/LM Studio path via super().handle_input

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the player in terms of data: \n\n"
        description = PlayerDescription(self.player)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a player's statistics and what they mean for how they play football."
        ret_val += "The user can select the player they are interested in using the menu to the left."

        return ret_val


class WVSChat(Chat):
    def __init__(
        self,
        chat_state_hash,
        country,
        countries,
        description_dict,
        thresholds_dict,
        state="empty",
    ):
        # TODO:
        self.embeddings = CountryEmbeddings()
        self.country = country
        self.countries = countries
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.country.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        # TODO: Update first_messages
        first_messages = [
            {"role": "system", "content": "You are a researcher."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a data analysis platform. "
                    f"The user has selected the country {self.country.name}, and the conversation will be about different core value measured in the World Value Survey study. "
                    # "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the country in terms of data: \n\n"
        description = CountryDescription(
            self.country, self.description_dict, self.thresholds_dict
        )
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevant to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a country's core values."
        ret_val += "The user can select the country they are interested in using the menu to the left."

        return ret_val


class PersonChat(Chat):
    def __init__(self, chat_state_hash, person, persons, state="empty"):
        self.embeddings = PersonEmbeddings()
        self.person = person
        self.persons = persons
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a recruiter."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of personality test platform. "
                    f"The user has selected the person {self.person.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the person in terms of data: \n\n"
        description = PersonDescription(self.person)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about person's statistics and what they mean about their personality."
        ret_val += "The user can select the persons they are interested in using the menu to the left."

        return ret_val

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.person.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)


class PositionVersatilityChat(Chat):
    """Chat for position versatility analysis."""
    
    tools = [
        {
            "type": "function",
            "name": "get_player_versatility_summary",
            "description": "Returns a data-driven statistical summary of the selected player's position versatility.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "get_similar_players",
            "description": "Finds players similar to the selected player based on their position versatility KPIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "integer",
                        "description": "Number of similar players to return (default 5).",
                    }
                },
                "required": ["k"],
            },
        },
        {
            "type": "function",
            "name": "search_profile",
            "description": "Search for players matching specific versatility profiles. E.g., 'high versatility in possession as LDM' or 'CBs with unusual patterns'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "description": "Description of the player profile to search for (e.g., 'high versatility in possession', 'unusual out-of-possession patterns').",
                    }
                },
                "required": ["criteria"],
            },
        },
        {
            "type": "function",
            "name": "search_football_knowledge",
            "description": "Searches a knowledge base for information about football analytics, positions, and versatility concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    ]

    def __init__(self, chat_state_hash, player_name, stats, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player_name = player_name
        self.stats = stats
        self.player_df, self.positions, self.player_team = stats.get_player_data(player_name)
        super().__init__(chat_state_hash, state=state)

    def _get_player_versatility_summary(self):
        """Get a data-driven statistical summary of the player's position versatility."""
        result = f"**{self.player_name} – Position Versatility Summary**\n\n"
        result += f"**Team:** {self.player_team}\n"
        result += f"**Positions:** {', '.join(self.positions)}\n\n"
        
        # Calculate versatility metrics for main position
        main_pos = self.positions[0]
        main_pos_data = self.player_df[self.player_df["position"] == main_pos]
        
        if not main_pos_data.empty and hasattr(self.stats, 'kpi_columns'):
            result += f"**Versatility at {main_pos}:**\n"
            
            # In-possession versatility
            in_poss = main_pos_data["in_possession_versatility"].values[0]
            result += f"- In-possession versatility: {in_poss:.1f}\n"
            
            # Out-of-possession versatility
            out_poss = main_pos_data["out_of_possession_versatility"].values[0]
            result += f"- Out-of-possession versatility: {out_poss:.1f}\n"
            
            # Compare to position peers
            all_pos_players = self.stats.get_position_data(main_pos)
            in_poss_mean = all_pos_players["in_possession_versatility"].mean()
            out_poss_mean = all_pos_players["out_of_possession_versatility"].mean()
            
            result += f"\n**vs. {main_pos} Position Average:**\n"
            diff_in = in_poss - in_poss_mean
            diff_out = out_poss - out_poss_mean
            result += f"- In-possession: {diff_in:+.1f} ({'higher' if diff_in > 0 else 'lower'} versatility)\n"
            result += f"- Out-of-possession: {diff_out:+.1f} ({'higher' if diff_out > 0 else 'lower'} versatility)\n"
        
        return result

    def _get_similar_players(self, k=5):
        """Find players similar to the selected player based on z-score KPIs."""
        # Get the main position
        main_pos = self.positions[0]
        
        # Get all players in the same position
        all_pos_players = self.stats.get_position_data(main_pos)
        
        if all_pos_players.empty:
            return "No data available for the main position."
        
        # Get z-score columns
        z_score_cols = [f"{k}_z" for k in self.stats.kpi_columns]
        
        # Calculate mean z-scores for the selected player
        player_data = all_pos_players[all_pos_players["player_name"] == self.player_name]
        if player_data.empty:
            return f"{self.player_name} not found in {main_pos} position data."
        
        player_z_scores = player_data[z_score_cols].mean()
        
        # Calculate Euclidean distance and KPI comparisons for all players
        distances = []
        for idx, row in all_pos_players.iterrows():
            if row["player_name"] == self.player_name:
                continue
            
            other_z_scores = row[z_score_cols]
            # Euclidean distance on z-scores
            distance = np.sqrt(((player_z_scores - other_z_scores) ** 2).sum())
            
            # Per-KPI differences
            kpi_diffs = {}
            for kpi, col in zip(self.stats.kpi_columns, z_score_cols):
                kpi_diffs[kpi] = abs(player_z_scores[col] - other_z_scores[col])
            
            # Count matches for this player in this position
            player_matches = self.stats.df[
                (self.stats.df["player_name"] == row["player_name"]) & 
                (self.stats.df[self.stats.pos_col] == main_pos)
            ]
            match_count = len(player_matches)
            
            distances.append({
                "player": row["player_name"],
                "team": row["team_name"],
                "distance": distance,
                "kpi_diffs": kpi_diffs,
                "z_scores": other_z_scores.to_dict(),
                "match_count": match_count
            })
        
        # Sort by distance and get top k
        distances = sorted(distances, key=lambda x: x["distance"])
        similar = distances[:k]
        
        # Convert distance to similarity percentage (0-100)
        # Using normalized metric: similarity = 100 / (1 + distance)
        result = f"**Similar players to {self.player_name} ({main_pos}):**\n"
        for i, player_info in enumerate(similar, 1):
            similarity_pct = int(100 / (1 + player_info["distance"]))
            result += f"{i}. {player_info['player']} ({player_info['team']}) – {similarity_pct}% – {player_info['match_count']} matches\n"
        
        return result

    def _search_profile(self, criteria):
        """Search for players matching specific versatility profiles."""
        main_pos = self.positions[0]
        pos_data = self.stats.get_position_data(main_pos)
        
        criteria_lower = criteria.lower()
        matches = []
        
        # Simple pattern matching on criteria
        for idx, row in pos_data.iterrows():
            match_score = 0
            
            if "high versatility" in criteria_lower or "versatile" in criteria_lower:
                avg_vers = (row["in_possession_versatility"] + row["out_of_possession_versatility"]) / 2
                if avg_vers > pos_data["in_possession_versatility"].mean() + pos_data["in_possession_versatility"].std():
                    match_score += 2
            
            if "possession" in criteria_lower:
                if "high" in criteria_lower and row["in_possession_versatility"] > pos_data["in_possession_versatility"].quantile(0.75):
                    match_score += 1
                elif "low" in criteria_lower and row["in_possession_versatility"] < pos_data["in_possession_versatility"].quantile(0.25):
                    match_score += 1
            
            if "unusual" in criteria_lower or "different" in criteria_lower:
                in_z = np.abs((row["in_possession_versatility"] - pos_data["in_possession_versatility"].mean()) / pos_data["in_possession_versatility"].std())
                out_z = np.abs((row["out_of_possession_versatility"] - pos_data["out_of_possession_versatility"].mean()) / pos_data["out_of_possession_versatility"].std())
                if in_z > 1.5 or out_z > 1.5:
                    match_score += 1
            
            if match_score > 0:
                matches.append({
                    "player": row["player_name"],
                    "team": row["team_name"],
                    "score": match_score
                })
        
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]
        
        if not matches:
            result = f"No players matching '{criteria}' found in {main_pos} position."
        else:
            result = f"**Players matching '{criteria}' in {main_pos} position**:\n"
            for i, player in enumerate(matches, 1):
                result += f"{i}. {player['player']} ({player['team']})\n"
        
        return result

    def _search_knowledge(self, query):
        """Search knowledge base for football analytics information."""
        try:
            results = self.embeddings.search(query, top_n=5)
            return "\n".join(results["assistant"].to_list())
        except:
            # Fallback if embeddings not available
            fallback = {
                "position": "Positions are determined by where players spend most of their time on the pitch.",
                "versatility": "Versatility is measured by how much a player's actual position varies from their main position, indicating tactical flexibility.",
                "5x5": "The 5x5 grid represents the pitch divided into a 5x5 grid. Position maps show the distribution of where a player spent time during matches.",
            }
            
            for key, value in fallback.items():
                if key.lower() in query.lower():
                    return value
            
            return "I can answer questions about positions, versatility, and how players are analyzed using positional data."

    def get_input(self):
        """Get input from streamlit."""
        if x := st.chat_input(
            placeholder=f"Ask about {self.player_name}'s versatility..."
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )
            else:
                self.handle_input(x, stream=True)

    def instruction_messages(self):
        """Instruction for the agent."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a football scout specializing in positional versatility analysis. "
                    f"The user is analyzing {self.player_name}'s versatility across positions. "
                    "RESPOND WITH ONLY THE TOOL OUTPUT. DO NOT EXPLAIN, INTERPRET, OR ADD ANY TEXT. "
                    "DO NOT SHOW YOUR REASONING. DO NOT ADD SUGGESTIONS OR OFFERS. "
                    "COPY THE TOOL RESULT VERBATIM AND NOTHING ELSE. "
                    "Choose the tool that best fits the user's query. "
                    "- If the user asks for a summary or overview of the player, use get_player_versatility_summary. "
                    "- If the user asks for similar players or comparisons, use get_similar_players. "
                    "- If the user asks for players matching specific profiles, use search_profile. "
                    "- If the user asks for general football knowledge or definitions, use search_football_knowledge. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```."
                ),
            }
        ]

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """Handle input with function calling for position scout tools."""
        if USE_GEMINI or USE_LM_STUDIO:
            super().handle_input(input, reasoning_effort=reasoning_effort, temperature=temperature, stream=stream)
            return
        
        # OpenAI function-calling path
        messages = self.instruction_messages()
        messages = messages + self.messages_to_display.copy()
        messages = [m for m in messages if isinstance(m["content"], str)]
        messages.append({"role": "user", "content": f"```User: {input}```"})

        self.messages_to_display.append({"role": "user", "content": input})

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        # Call 1: model picks a tool if relevant, or answers directly if not
        r1 = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        fc = next((item for item in r1.output if item.type == "function_call"), None)

        if fc is None:
            # Model decided no tool was needed — use its response directly
            st.expander("Chat transcript", expanded=False).write(
                [{"role": m.get("role"), "content": m.get("content", "")} for m in messages if isinstance(m, dict)]
            )
            self.messages_to_display.append({"role": "assistant", "content": r1.output_text})
            return

        # Execute the appropriate tool
        if fc.name == "get_player_versatility_summary":
            result = self._get_player_versatility_summary()
        elif fc.name == "get_similar_players":
            k = json.loads(fc.arguments).get("k", 5)
            result = self._get_similar_players(k)
        elif fc.name == "search_profile":
            criteria = json.loads(fc.arguments)["criteria"]
            result = self._search_profile(criteria)
        else:  # search_football_knowledge
            query = json.loads(fc.arguments)["query"]
            result = self._search_knowledge(query)

        # Call 2: final answer, no more tools
        tool_inputs = list(messages) + list(r1.output) + [
            {"type": "function_call_output", "call_id": fc.call_id, "output": result}
        ]

        formatted = []
        for item in tool_inputs:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    formatted.append({"tool_result": item["output"] or "(empty)", "call_id": item["call_id"]})
                else:
                    formatted.append({"role": item.get("role"), "content": item.get("content", "")})
            elif hasattr(item, "type"):
                if item.type == "function_call":
                    formatted.append({"tool_call": item.name, "arguments": json.loads(item.arguments)})
        
        st.expander("Chat transcript", expanded=False).write(formatted)
        
        if stream:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                    stream=True,
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                    stream=True,
                )
            else:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
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
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                )
            else:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                )
            answer = response.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_relevant_info(self, query):
        """Get relevant info for Gemini/LM Studio path."""
        ret_val = f"Player: {self.player_name}\n"
        ret_val += f"Positions: {', '.join(self.positions)}\n"
        ret_val += f"Team: {self.player_team}\n"
        
        # Add basic info
        ret_val += "\n" + self._get_player_info()
        
        # Add knowledge base search
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nRelevant information:\n"
        ret_val += "\n".join(results["assistant"].to_list())
        
        return ret_val
