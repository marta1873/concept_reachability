
def transform_captions_batch(captions, option):
    """
    Transforms a batch of captions according to the specified option
    Args:
    captions: tuple of str, batch of captions in the form "a {colour1} {shape1} behind a {colour2} {shape2}"
    option: int, the transformation option (1 to 6)
    Returns:
    list of str, the transformed captions
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
