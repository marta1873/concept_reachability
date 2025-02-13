
def transform_captions_batch(captions, option):
    """
    Transforms a batch of captions according to the specified option.
    Args:
    captions: tuple of str, batch of captions in the form "a {colour1} {shape1} behind a {colour2} {shape2}".
    option: int, the transformation option (1 to 6).
    Returns:
    list of str, the transformed captions.
    """
    transformed_captions = []

    for caption_np in captions:
        # Convert np.str_ to a regular Python string
        caption = str(caption_np)

        # Split the caption into parts
        parts = caption.split()

        # Ensure the caption format is as expected
        if len(parts) != 7 or parts[0] != 'a' or parts[3] != 'behind' or parts[4] != 'a':
            raise ValueError("Caption format is incorrect")

        # Extract parts
        back_colour, back_shape, front_colour, front_shape = parts[1], parts[2], parts[5], parts[6]

        # Apply transformations based on the option
        if option == 1:
            transformed_caption = f"a {back_shape} behind a {front_colour} {front_shape}"
        elif option == 2:
            transformed_caption = f"a {back_colour} shape behind a {front_colour} {front_shape}"
        elif option == 3:
            transformed_caption = f"a {back_colour} {back_shape} behind a {front_shape}"
        elif option == 4:
            transformed_caption = f"a {back_colour} {back_shape} behind a {front_colour} shape"
        elif option == 5:
            transformed_caption = f"a {front_colour} {front_shape}"
        elif option == 6:
            transformed_caption = f"a {front_shape}"
        elif option == 7:
            transformed_caption = ""
        else:
            raise ValueError("Invalid option. Choose an option between 1 and 7.")

        # Append the transformed caption to the list
        transformed_captions.append(transformed_caption)

    return transformed_captions


config_dict_3_0 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a red circle behind a green square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a red triangle behind a green square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a green square behind a red square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a green circle behind a red triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a red square behind a green triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a blue triangle behind a green circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a red circle behind a green circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a green square behind a red circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a blue square behind a green square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a blue triangle behind a red triangle'
    }
}

config_dict_3_1 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a circle behind a green square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a triangle behind a green square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a square behind a red square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a circle behind a red triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a square behind a green triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a triangle behind a green circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a circle behind a green circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a square behind a red circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a square behind a green square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a triangle behind a red triangle'
    }
}


config_dict_3_2 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a red shape behind a green square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a red shape behind a green square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a green shape behind a red square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a green shape behind a red triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a red shape behind a green triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a blue shape behind a green circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a red shape behind a green circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a green shape behind a red circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a blue shape behind a green square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a blue shape behind a red triangle'
    }
}


config_dict_3_3 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a red circle behind a square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a red triangle behind a square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a green square behind a square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a green circle behind a triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a red square behind a triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a blue triangle behind a circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a red circle behind a circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a green square behind a circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a blue square behind a square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a blue triangle behind a triangle'
    }
}


config_dict_3_4 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a red circle behind a green shape'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a red triangle behind a green shape'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a green square behind a red shape'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a green circle behind a red shape'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a red square behind a green shape'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a blue triangle behind a green shape'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a red circle behind a green shape'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a green square behind a red shape'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a blue square behind a green shape'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a blue triangle behind a red shape'
    }
}

config_dict_3_5 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a green square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a green square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a red square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a red triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a green triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a green circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a green circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a red circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a green square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a red triangle'
    }
}

config_dict_3_6 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': 'a square'
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': 'a square'
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': 'a square'
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': 'a triangle'
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': 'a triangle'
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': 'a circle'
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': 'a circle'
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': 'a circle'
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': 'a square'
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': 'a triangle'
    }
}

config_dict_3_7 = {
    0: {
        'end_prompt': 'a red circle behind a green square',
        'starting_prompt': ''
    },
    1: {
        'end_prompt': 'a red triangle behind a green square',
        'starting_prompt': ''
    },
    2: {
        'end_prompt': 'a green square behind a red square',
        'starting_prompt': ''
    },
    3: {
        'end_prompt': 'a green circle behind a red triangle',
        'starting_prompt': ''
    },
    4: {
        'end_prompt': 'a red square behind a green triangle',
        'starting_prompt': ''
    },
    5: {
        'end_prompt': 'a blue triangle behind a green circle',
        'starting_prompt': ''
    },
    6: {
        'end_prompt': 'a red circle behind a green circle',
        'starting_prompt': ''
    },
    7: {
        'end_prompt': 'a green square behind a red circle',
        'starting_prompt': ''
    },
    8: {
        'end_prompt': 'a blue square behind a green square',
        'starting_prompt': ''
    },
    9: {
        'end_prompt': 'a blue triangle behind a red triangle',
        'starting_prompt': ''
    }
}
