#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


# Config that serves all environment
GLOBAL_CONFIG = {
    "HOST": "http://localhost:8000", # default "localhost", "django" if using docker
    "USE_CUDA_IF_AVAILABLE": True,
    "ROUND_DIGIT": 6
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True,
    },

    "staging": {
        "DEBUG": True,
        "HOST": "http://django:8000", # default "localhost", "django" if using docker
    },

    "production": {
        "DEBUG": False,
        "HOST": "https://it245151.uni-graz.at", # default "localhost", "django" if using docker
        "ROUND_DIGIT": 3
    }
}


def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ['ML_ENV'] if 'ML_ENV' in os.environ else 'development'
    ENV = ENV or 'development'
    print(ENV)

    rd_pw = os.environ['REDIS_PASSWORD'] if 'REDIS_PASSWORD' in os.environ else None

    # raise error if environment is not expected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV
    config['REDIS_PASSWORD'] = rd_pw
    # config['DEVICE'] = 'cpu'

    return config

# load config for import
CONFIG = get_config()

if __name__ == '__main__':
    # for debugging
    import json
    print(json.dumps(CONFIG, indent=4))