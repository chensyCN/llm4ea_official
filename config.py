
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config(object):
    # for data processing
    simplify_url = True
    remove_ID_attr = False

    # for active selection
    query_scheme = "aggregated"

    # for annotation
    simulate = False
    gpt_api_key = os.getenv('GPT_API_KEY')
    gpt_model = 'gpt-3.5-turbo'

    # for probabilistic inference
    init_with_attr = False # enhance the label refinement with identical attributes
    label_refine = True
    delta_0 = 0.5 
    delta_1 = 0.9 

    ## print control during exps
    print_during_exp = {
        'paris': False
    }

    @classmethod
    def __repr__(cls):
        return '\n'.join(f'{k}: {v}' for k, v in cls.__dict__.items() if not k.startswith('__') and k != 'gpt_api_key')