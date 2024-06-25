import glob
import os.path

import flask
import asyncio

import librosa
from flask import Flask, render_template, request, session
from flask_session import Session
from flask_cors import CORS
import dspy
class StudentFeedback(dspy.Signature):
    """A student is learning English. You are assessing a spoken utterance. In at most two sentences, summarize (1) their specific strengths in English skills and (2) things they can work on to improve. Address the student in the second person. Include specific examples that the student can learn from. Be colloquial, as if in spoken conversation."""

    convo = dspy.InputField()
    output = dspy.OutputField(desc="Treat this as a spoken conversation, so be succinct, colloquial, and empathetic.")

class OfferFeedback(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_feedback = dspy.ChainOfThought(StudentFeedback)

    def forward(self, convo):
        answer = self.generate_feedback(convo=convo)
        return answer
from empathy_generation import OfferFeedback, StudentFeedback, call_empathy_gen
from ehcalabres_wav2vec_zeroshot import call_frustration
import logging
import argparse
from nce_v7_llama_api import send
import sys
import random
import requests
from query_response import classify_query, respond_to_user, summarize_topic
from summarize_convo import summarize_conversation
import re
import json
from personalization import mandarin_translation, feedback_style_update

from openai import OpenAI


parser = argparse.ArgumentParser(description="Simple API for chat bot")
parser.add_argument('--serving_hostname', default="0.0.0.0", help="API web server hostname.")
parser.add_argument('--serving_port', type=int, default=8080, help="API web server port.")

args = parser.parse_args()

serving_hostname = args.serving_hostname
serving_port = args.serving_port


# Create the Flask app instance
app = Flask(__name__)

LOGGER = logging.getLogger('gunicorn.error')

SECRET_KEY = 'YOURKEY'
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)

Session(app)
CORS(app)
blueprint = flask.Blueprint('parlai_api', __name__, template_folder='templates')

import json
# LOAD THE PREDEFINED UTTERANCE FILES
EMPATHY_UTTS = json.load(open("utterances/empathy.json"))["empathetic_utts"]
ERROR_REPHRASES = json.load(open("utterances/error_rephrase.json"))["rephrasers"]


FRUST_THRESHOLD = 0.4

empathy_response_storage = {}
grammar_feedback_storage = {}
feedback_buffer = {}
query_convo_buffer = {}
history_counts = {}
recent_three_utts = {}
recap_topic = {}
pre_survey_preferences = {}




@blueprint.route("/pre_survey", methods=["POST"])
def update_survey():
    data = request.get_json()
    print(data)
    pre_survey_preferences.update({
        data["id"]: data
    })
    return "OK"


# Define a route for the root URL
@blueprint.route('/call', methods=["POST"])
def call_empathy_responses():
    print("CALLING EMPATHY")
    data = request.form.get('json')
    if data:
        data = json.loads(data)
    print(data)

    audio_file = request.files.get('audio')
    print(audio_file)
    if audio_file:
        # Save the audio file or process it as needed
        audio_file.save("audio_cache/audio.wav")
        print("Saved audio file!")
    else:
        print("NO AUDIO FILE")


    client = OpenAI()

    audio_file = open("audio_cache/audio.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    text = transcription.text
    print(transcription.text)

    history, parameters, env_type, unit, uid = data.get('updated_hist', []), data.get(
        'parameters', {"empathy_mode": "0", "unit": "unit1"}), data.get('env_type', ''),  data.get('unit', 'unit1'), data.get("experimentID", "")
    # Empathy_mode: 0 = no empathy; 1 = random selection; 2 = generation
    print("TEXT", text)
    if uid not in recap_topic:
        recap_topic.update({uid: None})

    if uid not in history_counts:
        history_counts.update({uid: 2})
    else:
        history_counts[uid] = history_counts[uid] + 2

    if uid not in recent_three_utts:
        recent_three_utts.update({uid: {"user": [text], "bot": []}})
    else:
        recent_three_utts[uid]["user"].append(text)
        recent_three_utts[uid]["user"] = recent_three_utts[uid]["user"][-3:]

    if history_counts[uid] >= 20 or "bye" in text.lower():
        ep_done = True
    else:
        ep_done = False

    print("Through initialization")
    if uid in feedback_buffer and feedback_buffer[uid]:
        # Prevent timing out for llama
        response_vicuna = send(text, [], parameters, unit, env_type)
        query_convo_buffer[uid] = query_convo_buffer[uid] + "\nUser: " + text
        is_relevant, bot_resp = classify_query(query_convo_buffer[uid])
        query_convo_buffer[uid] = query_convo_buffer[uid] + "\nAssistant: " + bot_resp
        # print(query_convo_buffer[uid])
        if is_relevant and len(history) > 3:
            return {
                "response": mandarin_translation(bot_resp, pre_survey_preferences[uid]["mandarin_translation"]),
                "updated_hist": history,
                "parameters": parameters,
                "episode_done": ep_done
            }
        texts = feedback_buffer[uid].split(" | ")

        curr_topic = recap_topic[uid]
        if curr_topic:
            prefix = random.choice(
                [f"Alright, let's continue our conversation about {curr_topic}.", f"Let's get back to our chat on {curr_topic}!",
                 f"Okay let's go back to our conversation about {curr_topic}.", f"Now back to our conversation with respect to {curr_topic}.",
                 f"Lets' go back to our chat. We just talked about {curr_topic}.", f"Let's keep chatting about {curr_topic}."])
        else:
            prefix = random.choice(
                ["Okay, let's keep chatting.", "Let's go back to our conversation!", "Let's continue our chat!"]
            )

        text, vicuna = texts[0], texts[1]
        feedback_buffer.update({uid: False})
        query_convo_buffer.update({uid: False})

        recent_three_utts[uid]["bot"].append(vicuna)
        recent_three_utts[uid]["bot"] = recent_three_utts[uid]["bot"][-3:]

        return {
            "response": mandarin_translation(bot_resp + " " + prefix + " " + vicuna, pre_survey_preferences[uid]["mandarin_translation"]),
            "updated_hist": history + [text, vicuna],
            "parameters": parameters,
            "episode_done": ep_done
        }



    response_vicuna = send(text, history, parameters, unit, env_type)

    frust, _ = call_frustration()
    print(frust, ">>> FRUSTRATION LEVEL")

    if uid not in empathy_response_storage:
        empathy_response_storage.update({uid: 0})
    else:
        empathy_response_storage[uid] = empathy_response_storage[uid] - 1

    print("Through checkpoint 1")

    if parameters["empathy_mode"] == "3":
        # Forced
        empathetic_response = call_empathy_gen(recent_three_utts[uid]["user"], pre_survey_preferences[uid]["feedback_pref"])
    elif "?" in text:
        empathetic_response = ""
    elif frust < FRUST_THRESHOLD or parameters["empathy_mode"] == "0" or empathy_response_storage[uid] > 0:
        empathetic_response = ""
    else:
        # Only provide grammar correctness feedback if there is no need for empathetic feedback
        if parameters["empathy_mode"] == "1":
            empathetic_response = random.choice(EMPATHY_UTTS)
        else:
            empathetic_response = call_empathy_gen(recent_three_utts[uid]["user"], pre_survey_preferences[uid]["feedback_pref"])
        print("Through empathy generation")
        empathy_response_storage.update({uid: 4})

    print("Through checkpoint 2")

    concat_resp_string = None
    if len(empathetic_response):
        feedback_buffer.update({uid: text + " | " + response_vicuna["response"]})
        if parameters["empathy_mode"] == "1":
            concat_resp_string = empathetic_response
        else:
            concat_resp_string = empathetic_response + "  " + random.choice(["How does that sound?", "Does that sound alright to you?", "", "Does that sound good?"])
        concat_resp_string = concat_resp_string.strip(" ").replace("    ", "  ")

        # Update conversation history for the feedback
        conv_hist = "User: " + text + "\nAssistant: " + concat_resp_string
        if len(empathetic_response):
            concat_resp_string = concat_resp_string + "    "
        # Add a functionality for recapping
        if history_counts[uid] > 8:
            topic_hist = "Assistant: " + recent_three_utts[uid]["bot"][0]\
                         + "\nYou: " + recent_three_utts[uid]["user"][0]\
                         + "\nAssistant: " + recent_three_utts[uid]["bot"][1]\
                         + "\nYou: " + recent_three_utts[uid]["user"][1]\
                         + "\nAssistant: " + recent_three_utts[uid]["bot"][2] \
                         + "\nYou: " + recent_three_utts[uid]["user"][2]
            recap_topic.update({uid: summarize_topic(topic_hist)})
        query_convo_buffer.update({uid: conv_hist})
    else:
        feedback_buffer.update({uid: False})

    print("Through checkpoint 3")

    # concat_resp_string = grammar_correct + "  " + empathetic_response + "  " + response_vicuna["response"]
    # concat_resp_string = concat_resp_string.strip(" ").replace("    ", "  ")

    if concat_resp_string:
        print("Through checkpoint 4")
        return {
            "response": mandarin_translation(concat_resp_string, pre_survey_preferences[uid]["mandarin_translation"]),
            "updated_hist": history,
            "parameters": parameters,
            "episode_done": ep_done
        }
    else:
        # Only do the conversation summarization here
        if len(history) > 14:
            pref_hist = history[0].split("\n\n\n ")[0] + "\n\n\n"
            summ_all = summarize_conversation(history)
            summ_all = pref_hist + summ_all + " " + text
            print("Through checkpoint 4 summarized conversation")
            recent_three_utts[uid]["bot"].append(response_vicuna["response"])
            recent_three_utts[uid]["bot"] = recent_three_utts[uid]["bot"][-3:]
            return {
                "response": mandarin_translation(response_vicuna["response"], pre_survey_preferences[uid]["mandarin_translation"]),
                "updated_hist": [summ_all, response_vicuna["response"]],
                "parameters": parameters,
                "episode_done": ep_done
            }
        print("Through checkpoint 4")
        recent_three_utts[uid]["bot"].append(response_vicuna["response"])
        recent_three_utts[uid]["bot"] = recent_three_utts[uid]["bot"][-3:]
        return {
            "response": mandarin_translation(response_vicuna["response"], pre_survey_preferences[uid]["mandarin_translation"]),
            "updated_hist": history + [text, response_vicuna["response"]],
            "parameters": parameters,
            "episode_done": ep_done
        }


@blueprint.route('/health', methods=['GET'])
def get_health():
    return "OK"


async def main():
    app.register_blueprint(blueprint)
    app.run(host=serving_hostname, port=serving_port)

main_loop = asyncio.get_event_loop()
main_loop.run_until_complete(main())