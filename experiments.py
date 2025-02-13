from generate_dataset import generate_data_for_steering
from concept_directions import find_concept_vector, sample_with_concept_vector
from evaluate_images import load_classifiers, evaluate_accuracy_batch


def run_test(model, test, end_prompt, starting_prompt, space='p'):
    """
    Implements the optimisation of a steering vector between the target concepts and starting prompt and uses the vector
    to sample with steering
    Args:
        model: Diffusion, diffusion model instance
        test: int, test number corresponding to the starting prompt and target concept combination
        end_prompt: str, prompt describing target concept combinations
        starting_prompt: str, starting prompt
        space: str, either 'p' for prompt space or 'h' for h-space
    Returns:
    """
    # Create target images and list of corresponding starting prompts
    images, prompts = generate_data_for_steering(end_prompt, starting_prompt)

    # Optimise concept vector that will be used to steer model outputs
    vec = find_concept_vector(images, prompts, diffusion_model=model, save_dir=f'{test}', space=space)

    # Sample images using optimised concept vector
    images = sample_with_concept_vector(model, prompt=starting_prompt, concept_vector=vec, num_samples=100, test=test,
                                        space=space)

    # Evaluate steering accuracy using pre-trained classifiers
    model_list = load_classifiers()
    evaluate_accuracy_batch(images, model_list, ['colour', 's1', 's2'], end_prompt,
                            save_directory=f'{model.local_dir}/{test}/samples_{starting_prompt}', space=space)

    # Reset the prompt vector if steering was applied in the prompt space
    if space == 'p':
        model.p_vec.zero_()


def run_experiments(model, test, config_dict, space='p'):
    """
    Extracts starting prompt and prompt describing the target combinations and implements a test for steering
    Args:
        model: Diffusion, diffusion model instance
        test: int, test number corresponding to the starting prompt and target concept combination
        config_dict: dict, a configuration dictionary defining the prompts for each test case. It should follow the
                    structure
                    {<test_number> (int) :
                    {'end_prompt' (str) : <prompt describing target concept combination> (str),
                    'starting_prompt' (str) : <prompt describing starting prompt]> (str)}
                    }
        space: str, either 'p' for prompt space or 'h' for h-space
    Returns:
    """
    if test not in config_dict:
        raise KeyError(f"Test number {test} not found in config_dict.")

    # Extract prompts from config_dict
    end_prompt = config_dict[test]['end_prompt']
    starting_prompt = config_dict[test]['starting_prompt']

    # Run the steering test
    run_test(model, test, end_prompt, starting_prompt, space=space)


def sample_prompts(model, config_dict):
    """
    Samples images from diffusion model according to the prompts in a configuration dictionary
    Args:
        model: Diffusion, diffusion model instance
        config_dict: dict, a configuration dictionary defining the prompts for each test case. It should follow the
                    structure
                    {<test_number> (int) :
                    {'end_prompt' (str) : <prompt describing target concept combination> (str),
                    'starting_prompt' (str) : <prompt describing starting prompt> (str)}
                    }
    Returns:
    """
    # Load classifiers for evaluating sampled images
    model_list = load_classifiers()

    # Track sampled prompts to avoid redundant sampling
    sampled_prompts = set()

    # Iterate through prompts in config_dict
    for key, prompts in config_dict.items():
        # Sample target concept combination
        prompt = prompts['end_prompt']

        if prompt not in sampled_prompts:
            # Sample 100 images from the model
            images = model.sample(100, prompt, return_img=True)

            # Evaluate accuracy of the generated images
            evaluate_accuracy_batch(images, model_list, ['colour', 's1', 's2'], prompt,
                                    save_directory=f'{model.local_dir}/samples_{prompt}')

            # Mark prompt as sampled to prevent re-sampling
            sampled_prompts.add(prompt)

