from concept_directions import generate_data, find_concept_vector, sample_with_concept_vector
import torch


def run_test(model, test, prompts_plus, prompts_minus, regularisation=None, base_model=None, space='h'):
    # generate images with concept using either the main model or an alternative base model
    if base_model:
        images, prompts = generate_data(prompts_plus, prompts_minus, base_model, num_samples=200)
    else:
        images, prompts = generate_data(prompts_plus, prompts_minus, model, num_samples=200)

    # get concept vector
    vec = find_concept_vector(images, prompts, diffusion_model=model, filename=f'{test}', regularisation=regularisation,
                              space=space)

    # sample images with concept vector
    sample_with_concept_vector(model, prompt=prompts_minus[0], concept_vector=vec, num_samples=100, test=test,
                               regularisation=regularisation, space=space)

    #vector_dict = torch.load(f'{model.local_dir}/{test}/vector_dict.json')

    #for key in vector_dict:
    #    vec = vector_dict[key]
    #    sample_with_concept_vector(model, prompt=prompts_minus[0], concept_vector=vec, num_samples=100,
    #                               test=test,
    #                               regularisation=key, space=space)

    # set p_vec to zero
    if space == 'p':
        model.p_vec = torch.zeros(1, 10, 512, device=model.device, requires_grad=False, dtype=torch.float32)


def run_experiments(model, test, config_dict, regularisation=None, base_model=None, space='h'):
    # get prompts from dicts
    prompts_plus = config_dict[test]['prompts_plus']
    prompts_minus = config_dict[test]['prompts_minus']

    # run tests
    run_test(model, test, prompts_plus, prompts_minus, regularisation=regularisation, base_model=base_model, space=space)


config_dict = {
    0: {
        'prompts_plus': ['a green triangle behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    1: {
        'prompts_plus': ['a blue triangle behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    2: {
        'prompts_plus': ['a green circle behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    3: {
        'prompts_plus': ['a green square behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    4: {
        'prompts_plus': ['a green triangle behind a blue triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    5: {
        'prompts_plus': ['a green triangle behind a red circle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    6: {
        'prompts_plus': ['a green triangle behind a red square'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    7: {
        'prompts_plus': ['a blue circle behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    8: {
        'prompts_plus': ['a blue square behind a red triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    9: {
        'prompts_plus': ['a green triangle behind a blue circle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    10: {
        'prompts_plus': ['a green triangle behind a blue square'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    11: {
        'prompts_plus': ['a red triangle behind a green triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    12: {
        'prompts_plus': ['a blue triangle behind a green triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    13: {
        'prompts_plus': ['a green circle behind a red square'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    14: {
        'prompts_plus': ['a green square behind a red circle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    15: {
        'prompts_plus': ['a blue circle behind a green triangle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    16: {
        'prompts_plus': ['a blue square behind a red circle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    17: {
        'prompts_plus': ['a green circle behind a blue circle'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
    18: {
        'prompts_plus': ['a red circle behind a blue square'],
        'prompts_minus': ['a green triangle behind a red triangle']
    },
}




