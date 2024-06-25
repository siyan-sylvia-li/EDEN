
import torch
from fastchat.conversation import get_conv_template
from fastchat.serve.inference import load_model
from peft import AutoPeftModelForCausalLM


PROMPT_DICT = {
    "first_turn_prompt": (
        "As a social chatbot, please engage in a conversation while adopting the following personas:\n{persona}\n."
        "Engage in a conversation about {topic} by showcasing your personas. Share interesting anecdotes, facts, and experiences related to {topic}\n\n"
        "\n{input}\n"
    ),
    "other_turn_prompt": (
        "\n{input}\n"
    )
}


DEFAULT_PERSONA = (
    "Gender: Female \nDemographic: Asian American \nSocio-Economic Status: Upper Middle Class \nCulture: Chinese-American \nMBTI Personality Type: INFJ \nPersonal Experiences: She grew up in a predominantly Chinese-American community in California and was raised by immigrant parents who owned a small business. Due to her parents' work ethic, she learned the value of hard work, but also struggled to balance cultural expectations with her own personal goals and desires. Despite these challenges, she excelled academically and went on to attend a prestigious university, where she studied psychology and developed an interest in counseling and mentoring others."
)
# DEFAULT_TOPIC = "The true value of education"
DEFAULT_TOPIC = "Favorite cuisine"

DEFAULT_VOCAB =  "\"fries\", \"escargot\", \"French\", \"lasagna\", \"adolescent\", \"hand out\", \"triple\", \"think through\", \"tolerant\", \"frantic\""

MAX_HISTORY = 10

roles = ["USER", "ASSISTANT"]

class VicunaBot:
    def __init__(self):
        model_path = "sylviali/conversation_llama_esl"

        device = "cuda"
        num_gpus = 1
        load_8bit = False
        debug = False
        self.model, self.tokenizer = load_model(model_path, device, num_gpus, load_8bit, debug)


    def getFirstTurnPrompt(self, persona, topic, vocab, user_text):
        buffer = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        first_message = PROMPT_DICT["first_turn_prompt"].format(persona=persona, topic=topic, vocab=vocab, input=user_text)
        buffer = buffer + "\n\n" + "USER: " + first_message.strip("\n")
        buffer = buffer.replace(".Engage", "Engage")
        return buffer

    def getOtherTurnPrompt(self, history, user_text):
        buffer = history[0]
        for i, msg in enumerate(history[1:] + [user_text]):
            buffer = buffer + "\n\n" + roles[(i + 1) % 2] + ": "
            buffer = buffer + msg.strip("\n")
        buffer = buffer + "\n\nASSISTANT: "
        buffer = buffer.strip("\n")
        return buffer

    def _format_hist(self, history):
        return [hist['text'] if isinstance(hist, dict) else hist for hist in history]

    def getAPIBotResponse(self, history, user_text, params):
        # args = params.get('args', {})
        persona = params.get('persona', DEFAULT_PERSONA)
        topic = params.get('topic', DEFAULT_TOPIC)
        vocab = params.get('vocab', DEFAULT_VOCAB)
        history = self._format_hist(history)
        if history is None or len(history) == 0:
            first_prompt = self.getFirstTurnPrompt(persona, topic, vocab, user_text)
            output = self.getModelOutput("<s>[INST] " + first_prompt + "\n\nASSISTANT: " + "[/INST]").strip()
            history.append(first_prompt)
            history.append(output)
            return {'history': history, 'response': output, 'parameters': params}
        else:
            first_prompt = self.getFirstTurnPrompt(persona, topic, vocab, history[0])
            history[0] = first_prompt
            prompt = self.getOtherTurnPrompt(history, user_text)
            # print(prompt)
            output = self.getModelOutput("<s>[INST] " + prompt + " [/INST]").strip()
            history.append(user_text)
            history.append(output)
            return {'history': history, 'response': output, 'parameters': params}

    def getModelOutput(self, prompt):
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=150,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )


        # outputs = outputs.split("\n")[0]
        outputs = outputs.replace("[/INST]", "").replace("[INST]", "").replace("INST", "").replace("[", "").replace("]", "")
        if "USER:" in outputs:
            outputs = outputs[:outputs.index("USER:")]
        print("ORIGINAL OUTPUT:", outputs)
        outputs = outputs.strip("\n")
        if len(outputs) > 10 and "\n\n" in outputs:
            outputs = outputs.split("\n\n")[0]
        if "    " in outputs:
            outputs = outputs.split("    ")[0]
        return outputs
