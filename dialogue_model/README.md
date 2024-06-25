## Conversation Model

Our conversation model is trained on synthesized data between a generic persona and a persona that represents an ESL (English as a second language) speaker. We include our training data for this model, as well as the training script for this model, in this directory.

Our model is also hosted on Huggingface [here](https://huggingface.co/sylviali/conversation_llama_esl). You can refer to the `nce_v7_llama.py` filie in the `eden_api` directory for how to use the conversation model. 

### Training the Conversation Model
Our model is trained using PEFT on a single GPU.

#### Install Dependencies
```shell
pip install transformers torch trl peft bitsandbytes
```

#### Train Model
1. Run the training script
```shell
python3 train_sft_llama.py
```
2. Merge the PEFT adapter into the original model
```shell
python3 merge_peft.py
```