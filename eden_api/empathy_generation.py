import dspy

import os
import random
import pickle

from personalization import feedback_style_update

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

import openai
openai.api_key = "<OPENAI_API_KEY>"


def create_convo(history):
    # Reformatting past utterances into the acceptable format
    all_user_utts = ["- " + t for t in history]
    return "\n".join(all_user_utts)


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


def generate_gpt_empathy_rewrite(output):
    prompt = f"""Shorten and rewrite this utterance to sound simple, natural, and engaging; remove any assessment of speech including pronunciation and intonation. Don't use the word \"basic\":\n\n{output}"""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    msgs.append({"role": "assistant", "content": response.choices[0].message.content})
    msgs.append({"role": "system", "content": "Make your response different and casual, and shorten to at most 3 - 4 sentences."})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


client = openai.OpenAI(api_key="<OPENAI_API_KEY>")
turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=1000)
dspy.configure(lm=turbo)
reload_chain = OfferFeedback()
reload_chain.load("emp_bot.json")


def call_empathy_gen(history, feedback_pref={"short": False, "example": False}):
    if len(history) < 3:
        return ""
    conv = create_convo(history)
    outs = reload_chain.forward(conv)
    if feedback_pref["short"] or feedback_pref["example"]:
        rewrite = feedback_style_update(outs.output, conv, feedback_pref)
    else:
        rewrite = generate_gpt_empathy_rewrite(outs.output)
    return rewrite

