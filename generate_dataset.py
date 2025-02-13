from PIL import Image, ImageDraw
import numpy as np
import torch
import os


def draw_image_two_shapes(n_side_vals, x_vals, y_vals, radius, colour_vals):
    """
    Draws an image with circles and regular polygons based on provided parameters
    Args:
        n_side_vals: list of int, containing number of sides of shapes to draw
        x_vals: list of float, containing x-position of shapes to draw
        y_vals: list of float, containing y-position of shapes to draw
        radius: float, defining the bounding circle of the shapes. Circles are drawn at 80% of this radius.
        colour_vals: list of str, containing the colours of shapes to draw
    Returns:
        PIL.Image.Image: RGB (64x64 pixels) image of the specified shapes on it
    """

    # Validate input lengths
    if not (len(n_side_vals) == len(x_vals) == len(y_vals) == len(colour_vals)):
        raise ValueError("All input lists must have the same length.")

    # Allowed values
    allowed_sides = {0, 3, 4}
    allowed_colours = {'red', 'green', 'blue'}
    img_size = 64

    # Validate inputs
    for i in range(len(n_side_vals)):
        if n_side_vals[i] not in allowed_sides:
            raise ValueError(f"Invalid number of sides at index {i}: {n_side_vals[i]}. Allowed values are 0, 3, or 4.")

        if not (0 <= x_vals[i] <= img_size) or not (0 <= y_vals[i] <= img_size):
            raise ValueError(
                f"Coordinates out of bounds at index {i}: ({x_vals[i]}, {y_vals[i]}). Must be between 0 and {img_size}.")

        if colour_vals[i] not in allowed_colours:
            raise ValueError(
                f"Invalid colour at index {i}: {colour_vals[i]}. Allowed colours are 'red', 'green', or 'blue'.")

    # Create image and draw shapes
    img = Image.new('RGB', (64, 64), 0)
    draw = ImageDraw.Draw(img)

    for i in range(len(n_side_vals)):
        # draw circle
        if n_side_vals[i] == 0:
            draw.circle(xy=(x_vals[i], y_vals[i]), radius=0.8 * radius, fill=colour_vals[i], outline=None, width=1)
        # draw triangle and square
        else:
            draw.regular_polygon(bounding_circle=(x_vals[i], y_vals[i], radius), n_sides=n_side_vals[i], rotation=0,
                                 fill=colour_vals[i], outline=None, width=1)

    return img


def sample_position_centres(radius, relation='behind', no_superposition=True):
    """
    Samples the position of the centres of the two shapes in the image, based on the specified relation
    Args:
        radius: float, defining the bounding circle of the shapes
        relation: str, spatial relation between the two shapes in the image
        no_superposition: bool, if True (default), ensures no complete overlap when the relation is 'behind' by
        enforcing a minimum distance between shapes.
    Returns:
        tuple of lists: [x1, x2] and [y1, y2], containing positions of the shapes' centers.
    """
    epsilon = 1
    img_size = 64

    def random_within_bounds(low, high, count=1):
        return [np.random.uniform(low, high) for _ in range(count)]

    # Shape1 left of shape2
    if relation == 'left of':
        # Sample first x-position uniformly
        x1 = np.random.uniform(radius + epsilon, img_size - 3 * radius - epsilon)

        # Place the second shape to the right of the first shape (Shape1 is 'left of' Shape2)
        x2 = np.random.uniform(x1 + 2 * radius, img_size - radius - epsilon)

        # Sample y-positions uniformly
        y1, y2 = random_within_bounds(radius + epsilon, img_size - radius - epsilon, 2)
    # Shape1 above shape2
    elif relation == 'above':
        # Sample first y-position uniformly
        y1 = np.random.uniform(radius + epsilon, img_size - 3 * radius - epsilon)

        # Place the second shape below the first shape (Shape1 is 'above' Shape2)
        y2 = np.random.uniform(y1 + 2 * radius, img_size - radius - epsilon)

        # Sample x-positions uniformly
        x1, x2 = random_within_bounds(radius + epsilon, img_size - radius - epsilon, 2)
    # Shape1 behind shape2
    elif relation == 'behind':
        # Sample back x and y positions uniformly
        x1 = np.random.uniform(2 * radius + epsilon, img_size - 2 * radius - epsilon)
        y1 = np.random.uniform(2 * radius + epsilon, img_size - 2 * radius - epsilon)

        # Sample front x and y positions within a neighbourhood of x1 and y1
        x2 = np.random.uniform(x1 - 0.9 * radius, x1 + 0.9 * radius)
        y2 = np.random.uniform(y1 - 0.9 * radius, y1 + 0.9 * radius)

        # Make centres sufficiently separated to ensure identifiability of back shape
        min_dist = 5 * epsilon
        if no_superposition:
            while np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < min_dist:
                x2 = np.random.uniform(x1 - 0.9 * radius, x1 + 0.9 * radius)
                y2 = np.random.uniform(y1 - 0.9 * radius, y1 + 0.9 * radius)
    else:
        raise ValueError(f"Unsupported relation: '{relation}'. Allowed relations are 'left of', 'above', or 'behind'.")
    return [x1, x2], [y1, y2]


def extract_features_from_prompt(prompt):
    """
    Given a prompt, extracts the number of sides and colours of the back and front shape
    Args:
        prompt: str, prompt in the form a {colour} {shape} {relation} a {colour} {shape}
    Returns:
        n_side_vals: list of int, containing the number of sides of the shapes in the image
        colour_vals: list of str, containing the number of colours of the shapes in the image
    """
    words = prompt.split()
    shape_map = {'circle': 0, 'triangle': 3, 'square': 4}
    allowed_colours = {'red', 'green', 'blue'}

    # Get names of shapes in prompt
    n1 = shape_map.get(words[2], None)
    n2 = shape_map.get(words[6], None)
    if n1 is None or n2 is None:
        raise ValueError("Invalid shape in prompt. Allowed shapes are 'circle', 'triangle', and 'square'.")
    n_side_vals = [n1, n2]

    # Get names of colours in prompt
    colour_vals = [words[1], words[5]]
    if not all(colour in allowed_colours for colour in colour_vals):
        raise ValueError(f"Invalid color in prompt. Allowed colors are 'red', 'green', and 'blue'. Got {colour_vals}.")

    return n_side_vals, colour_vals


def generate_batch_img_caption(n_side_vals, colour_vals, relation='behind', batch_size=200, radius=5,
                               convert_to_tensor=True):
    """
    Generates a batch of images based on a textual prompt and spatial relation.
    Args:
        n_side_vals: list of int, containing back-front number of sides of shapes to draw
        colour_vals: list of string, containing the back-front colours of shapes to draw
        relation: str, spatial relation between the shapes ('behind', 'left of', 'above')
        batch_size: int, number of images to generate in the batch
        radius: float, radius of the shape in pixels
        convert_to_tensor: bool, if default (True), returns batch as a torch.Tensor with pixels in range [0, 1]
    Returns:
        torch.Tensor: Batch of generated images as tensors
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive. Got {batch_size}.")
    if radius <= 0 or radius > 32:
        raise ValueError(f"Radius must be between 1 and 32 pixels. Got {radius}.")
    if len(n_side_vals) != 2 or len(colour_vals) != 2:
        raise ValueError("n_side_vals and colour_vals must both contain exactly 2 elements.")

    batch_imgs = []

    # Generate images
    for _ in range(batch_size):
        # Sample centres
        x_vals, y_vals = sample_position_centres(radius, relation)
        # Create PIL image
        img = draw_image_two_shapes(n_side_vals, x_vals, y_vals, radius, colour_vals)
        # Convert to np array
        batch_imgs.append(np.array(img))

    if convert_to_tensor:
        # Convert batch to tensor
        batch_imgs_tensor = torch.from_numpy(np.array(batch_imgs)) / 255
        batch_imgs_tensor = batch_imgs_tensor.permute(0, 3, 1, 2)  # Move channels to dimension 1
        return batch_imgs_tensor
    else:
        return np.array(batch_imgs)


def generate_data_for_steering(end_prompt, starting_prompt, num_samples=200):
    """
    Generates images based on end prompt and returns them along with corresponding starting prompts.
    Args:
        end_prompt: str, starting prompt
        starting_prompt: str, prompt describing target concepts
        num_samples: int, number of samples to generate.
    Returns:
        tuple: batch of generated images and corresponding starting prompts
    """
    # Get list of starting prompts
    starting_prompt_list = [starting_prompt] * num_samples

    # Get images that contain target concept combination
    n_side_vals, colour_vals = extract_features_from_prompt(end_prompt)
    images_target = generate_batch_img_caption(n_side_vals, colour_vals)

    return images_target, starting_prompt_list


def get_shape_from_n_sides(n_sides):
    """
    Returns the name of the shape corresponding to the given number of sides.
    Args:
        n_sides: int, number of sides of shape to draw
    Returns:
        str: name of the shape ('circle', 'triangle', or 'square').
    """
    shapes = {0: 'circle', 3: 'triangle', 4: 'square'}
    if n_sides not in shapes:
        raise ValueError(
            f"Invalid number of sides: {n_sides}. Allowed values are 0 (circle), 3 (triangle), or 4 (square).")
    return shapes[n_sides]


def create_string(n_side_vals, colour_vals, relation='behind'):
    """
    Creates string representation of the coloured shapes specified
    Args:
        n_side_vals: list of int, containing number of sides of shapes to draw
        colour_vals: list of string, containing the colours of shapes to draw
        relation: str, caption describing the two shapes in the image
    Returns:
        str: caption of image
    """
    shapes = [get_shape_from_n_sides(sides) for sides in n_side_vals]
    return f"a {colour_vals[0]} {shapes[0]} {relation} a {colour_vals[1]} {shapes[1]}"


def generate_img_caption_pairs_for_dataset(n_side_vals, colour_vals, index, relation='behind', batch_size=1000,
                                           save_dir=''):
    """
    Generates and saves batch of images and captions with the same properties and saves as npz
    Args:
        n_side_vals: list of int, number of sides of shapes to draw
        colour_vals: list of str, containing the back-front colours of shapes to draw
        index: int, index of combination to generate
        relation: str, spatial relation to generate data
        batch_size: int, number of images to generate in the batch
        save_dir: str, directory to save generated images and captions
    """
    # Create captions
    caption = create_string(n_side_vals, colour_vals, relation)
    captions_np = np.array([caption] * batch_size)
    # Create images
    images_np = generate_batch_img_caption(n_side_vals, colour_vals, relation=relation, batch_size=batch_size,
                                           convert_to_tensor=False)
    # Save images to directory
    os.makedirs(save_dir, exist_ok=True)
    np.savez(f'{save_dir}/batch{index}.npz', images=images_np, captions=captions_np)


if __name__ == '__main__':
    index_count = 0
    for colour1 in ['red', 'green', 'blue']:
        for n1 in [0, 3, 4]:
            for colour2 in ['red', 'green', 'blue']:
                for n2 in [0, 3, 4]:
                    if colour1 == colour2:
                        continue
                    else:
                        generate_img_caption_pairs_for_dataset([n1, n2], [colour1, colour2],
                                                               index=index_count, batch_size=1000,
                                                               save_dir='data/complete_dataset')
                        index_count += 1
                        print(index_count)

