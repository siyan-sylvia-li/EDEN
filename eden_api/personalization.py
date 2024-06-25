import openai
openai.api_key = "<OPENAI_API_KEY>"
from datetime import datetime

client = openai.OpenAI(api_key="<OPENAI_API_KEY>")


def mandarin_translation(output, mandarin_pref):
    if mandarin_pref:
        prompt = f"""Translate the following sentence into Mandarin:\n\n{output.strip()}"""
        msgs = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=msgs
        )
        translation = response.choices[0].message.content.strip().replace("\n", "")
        return output + " // " + translation
    else:
        return output


def add_context_examples(output, convo):
    prompt = f"""Given the following utterances by a student learning English as the context:\n\n{convo}\n\nAnd a piece of feedback:\n\n{output}\n\nCreate a new piece of feedback with more context-specific examples supporting the feedback. Make the feedback colloquial, as if spoken in conversation. Don't use the word \"basic\"."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    msgs.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    msgs.append({
        "role": "system",
        "content": "Shorten your response to 3 - 4 sentences while retaining necessary information and detail."
    })
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


def add_reasoning(output, convo):
    prompt = f"""Given the following utterances by a student learning English as the context:\n\n{convo}\n\nAnd a piece of feedback:\n\n{output}\n\nCreate a new piece of feedback by elaborating on the reasoning and logic behind the feedback. Make the feedback colloquial."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


def add_details(output, convo):
    prompt = f"""Given the following utterances by a student learning English as the context:\n\n{convo}\n\nAnd a piece of feedback:\n\n{output}\n\nCreate a new piece of feedback with more details. Make the feedback colloquial."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


def remove_encouragement(output, convo):
    prompt = f"""Given the following utterances by a student learning English as the context:\n\n{convo}\n\nAnd a piece of feedback:\n\n{output}\n\nRemove any unnecessary encouragement or praise from this piece of feedback. Make the feedback colloquial."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


def shorten(output, convo):
    prompt = f"""Given the following utterances by a student learning English as the context:\n\n{convo}\n\nAnd a piece of feedback:\n\n{output}\n\nMake it more succinct and concise while retaining the original examples with their full context. Make the feedback colloquial and succinct. Don't use the word \"basic\". Try to shorten to at most 3 sentences."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


def feedback_style_update(output, convo, feedback_prefs):
    if feedback_prefs.get("example", False):
        output = add_context_examples(output, convo)
    if feedback_prefs.get("detailed", False):
        output = add_details(output, convo)
    if feedback_prefs.get("reasoning", False):
        output = add_reasoning(output, convo)
    if feedback_prefs.get("no_praise", False):
        output = remove_encouragement(output, convo)
    if feedback_prefs.get("short", False):
        output = shorten(output, convo)

    return output

