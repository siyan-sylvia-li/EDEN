## Barebone Streaming UI for EDEN

This is not the user interface we used for our user studies, but instead a minimally functional version of the UI so that people can run our code themselves and try out the chatbot with their own GPU! 

People are also more than welcome to use this UI for their own chatbots!

### General Workflow


### Setup
1. Install all dependencies by creating a conda environment by using the `environment_ui.yml` file
2. Write your preferences for having Mandarin translations and for the style of adaptive empathetic feedback in `pre_survey.json`
   1. `mandarin_translation`: do you want Mandarin translations of chatbot utterances? If yes, put down `true`, otherwise `false`
   2. `feedback_pref`: `short` - do you prefer short and succinct utterances? `example` - do you want your feedback to contain specific examples?
3. Write your desired topic and empathy mode in `settings.json`
   1. `empathy_mode`: `0` - no empathetic feedback, `1` - fixed empathetic feedback, `2` - adaptive empathetic feedback
   2. `topic`: see the appendix in our paper for a complete list of topic that the chatbot can discuss, defaults to "Favorite movie"

### Running the UI server
```shell
python3 app_stream.py
```

The app defaults to running on port 5023, but you can set it to run on any port by modifying `app_stream.py` and `static/js/recorder.js`, as long as the two ports are consistent to enable sockets. Then you can go to `localhost:5023` to access the UI!

Make sure to have your API (`eden_api/`) running on your GPU server first, and make sure you have updated the corresponding URLs in your `app_stream.py` file.