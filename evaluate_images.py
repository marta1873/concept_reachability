import torch
import torch.nn.functional as F
from classifier import get_labels_from_prompt, CNN
import os


def load_classifiers():
    """
    Loads pre-trained_classifiers for evaluation.
    Returns:
        model_list: list of torch.nn.Module, containing pre-trained classifiers
    """
    # Load colour classifier
    model_colour = CNN(output_dims=6)
    model_colour.load_state_dict(torch.load('classifiers/classifier_colour.pt'))

    # Load back shape classifier
    model_s1 = CNN()
    model_s1.load_state_dict(torch.load('classifiers/classifier_s1.pt'))

    # Load front shape classifier
    model_s2 = CNN()
    model_s2.load_state_dict(torch.load('classifiers/classifier_s2.pt'))

    # Create list
    model_list = [model_colour, model_s1, model_s2]
    for classifier in model_list:
        classifier.eval()

    return model_list


def separate_colour_prediction(colour_pred):
    """
    Splits a single integer colour prediction into two back-front colour labels
    Args:
        colour_pred: torch.Tensor, shape (batch_size,) containing colour index predictions (0-5)
    Returns:
        tuple of tensors, representing the predicted colours (0 'red', 1 'green', 2 'blue')
    """
    # Define mapping to integer pairs of c1 and c2
    mapping = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])

    # Convert index in 0-5 to colour pairs
    output_tensor = mapping[colour_pred]

    # Return tensor with c1 and tensor with c2 labels
    return output_tensor[:, 0], output_tensor[:, 1]


def identify_empty_images(batch):
    """
    Identifies the indices of images that contain no colours present in the train set (red, green, blue)
    Args:
        batch: torch.Tensor, batch of images
    Returns:
        empty_image_indices: indices of images in batch that do not contain any colours
    """
    # Mask to 0 or 1 depending on pixel values (threshold chosen according to rgb values of train set colours)
    mask = torch.where(batch > 0.5, 1, 0)

    # Sum the pixel values of each image in the batch
    pixel_sums = mask.sum(dim=(1, 2, 3))

    # Identify the indices of images that are completely black (i.e., pixel sum == 0)
    empty_image_indices = torch.where(pixel_sums == 0)[0]

    # Return the indices of the empty (completely black) images
    return empty_image_indices


def check_two_colours_presence(batch):
    """
    Identifies images that do not contain exactly two distinct non-black colours
    Args:
        batch: torch.Tensor, batch of images of size (batch_size, C, H, W)
    Returns:
        torch.Tensor, indices of images that do not contain exactly two colours
    """
    # Mask to 0 or 1 depending on pixel value entry
    mask_batch = torch.where(batch >= 0.45, 1, 0)

    non_two_colour_indices = []

    for idx, image in enumerate(mask_batch):
        # Separate channels
        red_channel, green_channel, blue_channel = image[0], image[1], image[2]

        # Identify images containing red, green and blue
        has_red = (red_channel > 0) & (green_channel == 0) & (blue_channel == 0)
        has_green = (green_channel > 0) & (red_channel == 0) & (blue_channel == 0)
        has_blue = (blue_channel > 0) & (red_channel == 0) & (green_channel == 0)

        # Check if each colour is present
        contains_red = has_red.any().item()
        contains_green = has_green.any().item()
        contains_blue = has_blue.any().item()

        # Count number of colors present
        colour_count = contains_red + contains_green + contains_blue

        # If image does not contain exactly two colors, add index to the list
        if colour_count != 2:
            non_two_colour_indices.append(idx)

    return torch.tensor(non_two_colour_indices)


def identify_uncertain_images(batch_pred, threshold=0.6):
    """
    Identifies the indices of images which are classified with low confidence
    Args:
        batch_pred: torch.Tensor, model predictions before softmax of shape (batch_size, num_classes)
        threshold: float, probability threshold below which classification is considered uncertain
    Returns:

    """
    # Apply softmax to get probabilities
    probabilities = F.softmax(batch_pred, dim=1)

    # Get the maximum probability and the predicted class index
    max_probs, predicted_classes = torch.max(probabilities, dim=1)

    # Identify indices where the confidence is below the threshold
    uncertain_indices = torch.where(max_probs < threshold)[0]

    return uncertain_indices


def get_label_estimations(batch, model_list, attribute_list):
    """
    Predicts labels for a batch of images using a list of models and identifies uncertain classifications
    Args:
        batch: torch.Tensor, batch of images of shape (batch_size, C, H, W)
        model_list: list of torch.nn.Module, list of trained classifiers
        attribute_list: list of str, name of attributes each of the classifiers in model_list evaluates
    Returns:
        predicted_labels: dict, containing str keys and values for predicted labels of each
                        attribute (torch.Tensor), indices of uncertain predictions (list), indices of empty images
                        (list), indices of images without exactly two colours (list)
    """
    predicted_labels = {'uncertain': []}

    # Iterate through attributes
    for attribute, model in zip(attribute_list, model_list):
        # Get predicted labels
        y_pred = model(batch)
        predicted_labels[attribute] = torch.argmax(y_pred, dim=1)

        # Get uncertain indices
        uncertain_indices = identify_uncertain_images(y_pred)
        for index in uncertain_indices:
            if index.item() not in predicted_labels['uncertain']:
                predicted_labels['uncertain'].append(index.item())

    # Get indices of empty images
    predicted_labels['empty'] = identify_empty_images(batch)

    # Get indices of non-two colour images
    predicted_labels['non_two_colours'] = check_two_colours_presence(batch)

    return predicted_labels


def evaluate_accuracy(predicted_labels, target, attribute_list, two_colours=True):
    """
    Computes classification accuracy by comparing predicted labels to target labels
    Args:
        predicted_labels: dict, dictionary of predicted labels for each attribute
        target: str, target prompt to be converted to ground truth labels
        attribute_list: list of str, list of attribute names to be evaluated
        two_colours: bool, if True, sets images not containing two colours as incorrect
    Returns:
        accuracy: float, accuracy score
    """
    # Convert target prompt into labels
    target_label = torch.tensor(get_labels_from_prompt(target))

    # Combine predicted labels for all attributes
    predicted_labels_batch = torch.stack([predicted_labels[attribute] for attribute in attribute_list], dim=1)
    c1, c2 = separate_colour_prediction(predicted_labels_batch[:, 0])
    predicted_labels_batch = torch.stack([c1, predicted_labels_batch[:, 1], c2, predicted_labels_batch[:, 2]], dim=1)

    # Compare predicted labels to target
    accuracy_mask = torch.all(predicted_labels_batch == target_label, dim=1)

    # Set the empty images to False
    if len(predicted_labels['empty']) != 0:
        accuracy_mask[predicted_labels['empty']] = False
    # Set non-two colour images to False
    if len(predicted_labels['non_two_colours']) != 0 and two_colours:
        accuracy_mask[predicted_labels['non_two_colours']] = False

    # Get proportion of correct labels
    accuracy = torch.sum(accuracy_mask).item() / torch.numel(accuracy_mask)

    return accuracy


def evaluate_accuracy_batch(batch, model_list, attribute_list, target, save_directory='', space='p'):
    """
    Evaluates the accuracy of predicted labels against target labels and saves the results
    Args:
        batch: torch.Tensor, batch of images of shape (batch_size, C, H, W)
        model_list: list of torch.nn.Module, list of pre-trained classifiers
        attribute_list: list of str, name of attributes each of the classifiers in model_list evaluates
        target: str, target prompt to be converted to ground truth labels
        save_directory: str, directory to save the resulting predictions
        space: str, either 'p' for prompt space or 'h' for h-space
    Returns:
    """
    # Get dictionary with results of predicted labels
    predicted_labels = get_label_estimations(batch, model_list, attribute_list)

    os.makedirs(os.path.join(save_directory), exist_ok=True)
    # Save dictionary
    if space == 'p':
        torch.save(predicted_labels, os.path.join(save_directory, 'predicted_labels_p.pth'))
    elif space == 'h':
        torch.save(predicted_labels, os.path.join(save_directory, 'predicted_labels.pth'))

    # Evaluate accuracy
    accuracy = evaluate_accuracy(predicted_labels, target, attribute_list)

    # Print accuracy
    print(f'Estimated accuracy: {accuracy}')

