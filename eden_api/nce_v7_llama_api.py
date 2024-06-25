"""
    Code credit: Yu Li
"""

import asyncio

from nce_v7_llama import VicunaBot

vc = VicunaBot()

class ParlaiAPI:
    @staticmethod
    async def send_message(text, history, params):
        return vc.getAPIBotResponse(text, history, params)


def send(text, history, parameters, unit, env_type):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if parameters.get('unit', None) is None:
        parameters['unit'] = unit

    result = loop.run_until_complete(ParlaiAPI.send_message(history, text, parameters))
    print(result)

    ep_done = False
    if env_type == "EXPERIMENT":
        if len(result['history']) >= 22:
            ep_done = True
    elif len(result['history']) >= 12:
        ep_done = True

    result['episode_done'] = ep_done
    return result
