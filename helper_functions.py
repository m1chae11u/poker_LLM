import os
import json

def vllm_env_setup():
    dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
    path_to_config = os.path.join(dir_of_this_script, 'configs', 'config.json')
    with open(path_to_config, 'r') as config_file:
        config_data = json.load(config_file)
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]