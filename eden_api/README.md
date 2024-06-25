# EDEN Server API

The code in this directory should be placed on a GPU server!

## Setup
1. Replace all the <OPENAI_API_KEY> with your own API keys
2. Create a conda environment with the `.yml` file
3. Download the `pytorch_model.bin` file from [this huggingface wav2vec2 model](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition/tree/main), and place it in the `model_storage` directory

## Run The Server
```shell
python3 app.py --serving_port=<PORT_NUMBER>
```

Make sure to update the URL for the GPU server and the port you are running the Flask application on in the front-end UI code, located under `local_ui/`.