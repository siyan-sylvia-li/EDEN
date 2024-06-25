"""
Based on the user's following utterance: " I would like to talk a movie that I have watched several days ago which is called Pegasus the second and is directed by a Chinese director Hang Han. The main character of the movie was a very famous super racing driver and in his last driving race competition he after crossed the finish line he had a very serious car accident and one of the important mechanical parts of his car has been lost which make it impossible to confirm whether his result was legal. So therefore his results was cancelled and he also was seriously injured in that car accident and could not drive a race car anymore. So he had to make a living as a driving school teacher until one day the owner of a car factory found him and wanted to sponsor him to form a team to play in a competition again. The owner of the car factory wants to promote his car factory and the main character wants to trade again for his own dream so they had a deal and collaboration and after all the result proved himself to the world again.", answer the user's following query: "Am I making grammar mistakes?" Answer in a spoken utterance. Provide specific feedback, but be succinct.

"""
import openai
import numpy as np


def create_convo_history(history):
    output_str = ""
    roles = ["User: ", "Assistant: "]
    for i, h in enumerate(history):
        output_str += roles[i % 2] + h + "\n"
    return output_str


def summarize_conversation(convo_history):
    convo_str = create_convo_history(convo_history)
    client = openai.OpenAI(api_key="<OPENAI_API_KEY>")
    prompt = f"""Given the following conversation history:\n\n{convo_str}\n\nPretend you are the user and summarize what you have said in this conversation. Use the first person when summarizing the conversation. Mimic the way the user talks."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )

    return response.choices[0].message.content

