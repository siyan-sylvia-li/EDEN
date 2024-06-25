"""
Based on the user's following utterance: " I would like to talk a movie that I have watched several days ago which is called Pegasus the second and is directed by a Chinese director Hang Han. The main character of the movie was a very famous super racing driver and in his last driving race competition he after crossed the finish line he had a very serious car accident and one of the important mechanical parts of his car has been lost which make it impossible to confirm whether his result was legal. So therefore his results was cancelled and he also was seriously injured in that car accident and could not drive a race car anymore. So he had to make a living as a driving school teacher until one day the owner of a car factory found him and wanted to sponsor him to form a team to play in a competition again. The owner of the car factory wants to promote his car factory and the main character wants to trade again for his own dream so they had a deal and collaboration and after all the result proved himself to the world again.", answer the user's following query: "Am I making grammar mistakes?" Answer in a spoken utterance. Provide specific feedback, but be succinct.

"""
import openai
import numpy as np
import nltk


def summarize_topic(convo):
    client = openai.OpenAI(api_key="<OPENAI_API_KEY>")
    prompt = f"""Given the following conversation history:\n\n{convo}\n\nDescribe the current general topic with ONE SHORT PHRASE."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content.lower()


def classify_query(convo_history):
    # if "?" in user_utt and np.any([x in user_utt for x in ["grammar", "grammatical", "vocab", "English", "mistake", "example", "sentence"]]):
    #     return True
    updated_convo_hist = convo_history
    is_relevant, bot_response = respond_to_user(updated_convo_hist)
    if not is_relevant and "?" in bot_response:
        # Get rid of questions in the response
        sents = nltk.sent_tokenize(bot_response)
        bot_response = " ".join([x for x in sents if "?" not in x])
    return is_relevant, bot_response


# TODO: Change to accommodate situation where user asks a series of questions
def respond_to_user(convo_history):
    client = openai.OpenAI(api_key="<OPENAI_API_KEY>")
    prompt = f"""Given the following user-chatbot exchange:\n\n{convo_history}\n\nIs the latest user utterance asking for clarifications or English learning advice? Answer with yes or no."""
    # prompt = f"""Based on the following conversation history:\n\n{convo}, answer the user's following query: \"{user_query}\" Answer in a spoken utterance. Provide specific feedback, but be succinct."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    query_class = response.choices[0].message.content
    msgs.append({"role": "assistant", "content": query_class})
    query_class = query_class.lower().startswith("yes")
    msgs.append({"role": "system", "content": "Respond to the last user utterance as the Assistant based on the conversation context. Be colloquial and helpful. You only know English and Mandarin."})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return query_class, response.choices[0].message.content

