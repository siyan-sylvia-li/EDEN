#!/usr/bin/env python
# -*- coding: utf-8 -*-
import uuid

from flask import Flask
from flask import render_template, session
from flask_session import Session

import numpy as np
import librosa
from flask_socketio import SocketIO, emit
import boto3
from flask_cors import CORS
import logging
import requests, json


logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.DEBUG)



RESPONSE_URL = "<GPU_SERVER>/call"
PRE_SURVEY_URL = "<GPU_SERVER>/pre_survey"

stream_config_dict = {}
recognizer_dict = {}
storage_dict = {}
asr_result_dict = {}
timestamp_dict = {}
agent_buffer = {}

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app)

NUM_WORDS = 30


polly_client = boto3.Session(
    aws_access_key_id="<AWS_ACCESS_KEY>",
    aws_secret_access_key="<AWS_ACESS_SECRET>",
    region_name='us-west-2').client('polly')
print("started_polly")

headers={
    'Content-type':'application/json',
    'Accept':'application/json'
}


def speak_text(text):
    og_text = text
    if og_text.endswith("//"):
        og_text = og_text.replace("//", "")
    if " // " in text:
        text = text.split(" // ")[0]
    if "/" in text:
        text = text.replace("/", "")
    if text == "":
        return 404
    try:
        response = polly_client.synthesize_speech(VoiceId='Joanna',
                                                  OutputFormat='mp3',
                                                  Text=text,
                                                  Engine='standard')

        cont = response['AudioStream'].read()
        if len(cont) % 2 == 1:
            cont = cont + b"\x00"
        file = open('speech.mp3', 'wb')
        file.write(cont)
        file.close()
        output_agent(cont)
        session["agent_speaking"] = True
        user_asr("BOT: " + og_text)
    except RuntimeError as e:
        print(e)


def predict(user_input):
    user_input = bytearray(user_input.replace(b"RIFF\xce\n\x00\x00WAVEfmt ", b""))
    session['buffer'].extend(user_input)
    if not session["agent_speaking"] and not session["waiting"]:
        buff = session['buffer'][:]
        wav = np.frombuffer(buff, dtype=np.int16).astype(np.float32)

        # Scale the audio
        scale = 1. / float(1 << 15)  # from librosa
        wav *= scale

        # Check if the trailing silence is longer than 2s
        trimmed, index = librosa.effects.trim(wav, top_db=10)
        from scipy.io.wavfile import write
        write('test.wav', 16000, trimmed)

        # If there is longer than 2s of silence
        if len(wav) - index[1] > 32000:
            with open("test.wav", 'rb') as audio_file:
                # Prepare the files dictionary
                files = {
                    'audio': ('file.wav', audio_file, 'audio/wav')
                }

                # Prepare the data dictionary
                data = {
                    'json': (None, json.dumps({
                        "experimentID": session["uid"],
                        "parameters": session["parameters"],
                        "updated_hist": session["history"],
                    }), 'application/json')
                }

                # Send the request
                session["waiting"] = True
                user_asr("...Thinking...")
                response = requests.post(RESPONSE_URL, files=files, data=data)

                resp_data = response.json()
                session["history"] = resp_data["updated_hist"]
                session["waiting"] = False
                # First call the URL


            session['buffer'] = bytearray(b"RIFF\xce\n\x00\x00WAVEfmt ")

            output = resp_data["response"]
            speak_text(output)


@app.route("/", methods=['POST', 'GET'])
def index():
    session['transmit_count'] = 0
    session["running_asr"] = False
    session["agent_speaking"] = False
    session["history"] = []
    session["parameters"] = json.load(open("settings.json"))
    session["uid"] = str(uuid.uuid4())
    session["waiting"] = False

    session['buffer'] = bytearray(b"RIFF\xce\n\x00\x00WAVEfmt ")
    pre_survey = json.load(open("pre_survey.json"))
    pre_survey.update({"id": session["uid"]})
    r = requests.post(PRE_SURVEY_URL, json=pre_survey, headers=headers)
    print("POSTING PRE-SURVEY RESULT", r)


    return render_template("index.html")


"""
    Receiving audio and sending audio
"""


@socketio.on("user-audio")
def handle_user(data):
    predict(data)
    session['transmit_count'] += 1


@socketio.on("agent-said")
def update_speech_state(speech_state):
    session["agent_speaking"] = speech_state
    if speech_state is False:
        session["buffer"] = bytearray(b"RIFF\xce\n\x00\x00WAVEfmt ")


@socketio.on("agent-audio")
def output_agent(data):
    session['agent_speaking'] = True
    # logging.info("Emission" + str(time.time() * 1000))
    emit("agent-audio", data)


# @socketio.on("agent-timestamp")
# def output_timestamp(id, pred):
#     # logging.info("Emission Timestamp" + str(time.time() * 1000))
#     emit("agent-timestamp", (id, pred))


# If user interrupts in the middle, we stop the transmission

@socketio.on("user-asr")
def user_asr(resp):
    # This goes onto the display
    emit("user-asr", resp)


@socketio.on("start-convo")
def start_convo(msg):
    print(msg)
    index()
    speak_text("Hi, I am a chatbot acting as your English tutor today. How are you doing?")
    user_asr("BOT: Hi, I am a chatbot acting as your English tutor today. How are you doing?")
    session["agent_speaking"] = True


@socketio.on("stop-convo")
def stop_convo(msg):
    print(msg)
    session["running_asr"] = False


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5023)
    # app.run()
