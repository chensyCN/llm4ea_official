
class Config(object):
    # for probabilistic inference
    init_with_attr = False
    label_refine = True

    # for inference
    query_scheme = "aggregated"
    simulate = False
    topk = True
    num_selected_ent = 20

    # for annotation
    simplify_url = True
    remove_ID_attr = False
    gpt_api_key = None # please specify your own GPT API key before running the code
    gpt_model = 'gpt-3.5-turbo'
    initial_alignment_score = 0.5 # delta_1 in the paper

    ## print control during exps
    print_during_exp = {
        'paris': False
    }


    @classmethod
    def __repr__(cls):
        return '\n'.join(f'{k}: {v}' for k, v in cls.__dict__.items() if not k.startswith('__') and k != 'gpt_api_key')