config_dict_2_1 = {
    0: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a green square']
    },
    1: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a blue triangle behind a green square']
    },
    2: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red circle behind a green square']
    },
    3: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red square behind a green square']
    },
    4: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a blue square']
    },
    5: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a green circle']
    },
    6: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a green triangle']
    },
    7: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a blue circle behind a green square']
    },
    8: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a blue square behind a green square']
    },
    9: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a blue circle']
    },
    10: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red triangle behind a blue triangle']
    },
    11: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a green triangle behind a red square']
    },
    12: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a blue triangle behind a red square']
    },
    13: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a green triangle behind a blue square']
    },
    14: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red circle behind a green triangle']
    },
    15: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red square behind a green circle']
    },
    16: {
        'prompts_plus': ['a red triangle behind a green square'],
        'prompts_minus': ['a red circle behind a green circle']
    }
}


def sample_prompts(model, config_dict):
    sampled_prompts = []

    for key in config_dict.keys():
        prompts_plus = config_dict[key]['prompts_plus'][0]
        prompts_minus = config_dict[key]['prompts_minus'][0]

        if prompts_plus not in sampled_prompts:
            model.sample(100, prompts_plus, return_img=False)
            sampled_prompts.append(prompts_plus)

        if prompts_minus not in sampled_prompts:
            model.sample(100, prompts_minus, return_img=False)
            sampled_prompts.append(prompts_minus)
