import os.path

from transformers import AutoModelForAudioClassification
import urllib
import librosa
from torch import nn
import torch
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support
from argparse import ArgumentParser
from noise_pause_process import speech_only_audio


MODEL_CHECKPOINT = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

WAV2VEC_MODEL = AutoModelForAudioClassification.from_pretrained(MODEL_CHECKPOINT)

# Reinitialize the classifier weights to help classification
WAV2VEC_MODEL.projector = nn.Linear(1024, 1024, bias=True)
WAV2VEC_MODEL.classifier = nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('model_storage/pytorch_model.bin', map_location=torch.device('cpu'))

WAV2VEC_MODEL.projector.weight.data = torch_state_dict['classifier.dense.weight']
WAV2VEC_MODEL.projector.bias.data = torch_state_dict['classifier.dense.bias']

WAV2VEC_MODEL.classifier.weight.data = torch_state_dict['classifier.output.weight']
WAV2VEC_MODEL.classifier.bias.data = torch_state_dict['classifier.output.bias']
# ==============

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
max_duration = 15.0
WAV2VEC_MODEL.cuda()
WAV2VEC_MODEL.eval()

SETUP = 2


def call_frustration():
    wav, silence_ratio = speech_only_audio()
    if silence_ratio >= 0.5:
        return 1, f"student is pausing more than usual => {silence_ratio}"
    inputs = feature_extractor(
        wav,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).input_values
    inputs = inputs.cuda()
    # EHCALABRES MODEL SPECIFIC ================
    neg_emotion = WAV2VEC_MODEL(inputs)[0]
    neg_emotion = torch.softmax(neg_emotion, dim=1)
    return_str = " ".join(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', "===>", str(neg_emotion)])
    if SETUP == 0:
        neg_emotion = neg_emotion[0, 0] + neg_emotion[0, 2] + neg_emotion[0, 3] + neg_emotion[0, 6]
        neg_emotion = neg_emotion.detach().cpu().item()
    elif SETUP == 1:
        neg_emotion = neg_emotion[0, 0] + neg_emotion[0, 2] + neg_emotion[0, 3]
        neg_emotion = neg_emotion.detach().cpu().item()
    elif SETUP == 2:
        # Only the ANGER setup, as specified in the paper
        neg_emotion = neg_emotion[0, 0]
        neg_emotion = neg_emotion.detach().cpu().item()
    elif SETUP == 3:
        neg_emotion = neg_emotion[0, 2] + neg_emotion[0, 3]
        neg_emotion = neg_emotion.detach().cpu().item()
    elif SETUP == 4:
        neg_emotion = neg_emotion[0, 2] + neg_emotion[0, 0]
        neg_emotion = neg_emotion.detach().cpu().item()
    elif SETUP == 5:
        neg_emotion = neg_emotion[0, 3] + neg_emotion[0, 0]
        neg_emotion = neg_emotion.detach().cpu().item()
    else:
        neg_emotion = int(int(torch.argmax(neg_emotion, dim=1).detach().cpu().item()) in [0, 2, 3])

    return neg_emotion, return_str
