import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data as data
from dataset import ShapesDataset
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, output_dims=3):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjusted for 64x64 input size
        self.fc2 = nn.Linear(128, output_dims)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer (with 0.3 dropout probability)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # First convolution + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Second convolution + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 32 * 16 * 16)  # Flatten after pooling

        # Fully connected layers + ReLU
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Final output layer (no ReLU, directly to softmax during loss calculation)
        x = self.fc2(x)

        return x


def get_labels_from_prompt(prompt):
    """
    Extracts and converts colour and shape labels from a descriptive text prompt
    Args:
        prompt: str, prompt describing image in the format "a <colour1> <shape1> <relation> a <colour2> <shape2>"
    Returns:
        list of int, containing the labels of the factors [c1, s1, c2, s2]
    """
    words = prompt.split()

    # Validate the prompt length (must have at least 7 words for the expected format)
    if len(words) < 7:
        raise ValueError(
            f"Invalid prompt format: '{prompt}'. Expected format like 'a <colour> <shape> behind a <colour> <shape>'.")

    # Extract colour and shape labels from the prompt
    colour1 = get_colour(words[1])
    shape1 = get_shape(words[2])
    colour2 = get_colour(words[5])
    shape2 = get_shape(words[6])

    return [colour1, shape1, colour2, shape2]


def get_colour(word):
    """
    Converts a colour name into its corresponding integer label
    Args:
        word: str, name of colour
    Returns:
        int, number corresponding to label
    """
    if word == 'red':
        return 0
    elif word == 'green':
        return 1
    elif word == 'blue':
        return 2
    else:
        raise ValueError(f"Invalid colour '{word}'. Expected 'red', 'green', or 'blue'.")


def get_shape(word):
    """
    Converts a colour name into its corresponding integer label
    Args:
        word: str, name of the shape
    Returns:
        int, number corresponding to label
    """
    if word == 'circle':
        return 0
    elif word == 'triangle':
        return 1
    elif word == 'square':
        return 2
    else:
        raise ValueError(f"Invalid shape '{word}'. Expected 'circle', 'triangle', or 'square'.")


def convert_to_integer_tensor(input_tensor_2d):
    """
    Converts (c1, c2) tuples into an integer from 0 to 5
    Args:
        input_tensor_2d: torch.Tensor, a 2D tensor of shape (N, 2) where each row represents a coordinate pair (c1, c2)
                        with elements c1, c2 taking values 0, 1, 2
    Returns:
        output_tensor: torch.Tensor, a 1D tensor of shape (N, ) containing integers from 0 to 5
    """
    # Define the reverse mapping from 2D coordinates to integers
    reverse_mapping = {
        (0, 1): 0,
        (0, 2): 1,
        (1, 0): 2,
        (1, 2): 3,
        (2, 0): 4,
        (2, 1): 5
    }

    # Create an empty list to store the corresponding integers
    output_list = []

    # Iterate over each pair in the input 2D tensor
    for pair in input_tensor_2d:
        # Convert the pair to a tuple and use it to look up the corresponding integer
        output_list.append(reverse_mapping[(pair[0].item(), pair[1].item())])

    # Convert the list to a tensor
    output_tensor = torch.tensor(output_list)

    return output_tensor


def get_dictionary(list_batch):
    """
    Converts a torch tensor of the labels for c1, s1, c2, s2 into a dictionary
    Args:
        list_batch: list, of shape (batch_size, 4), where each row represents [c1, s1, c2, s2] for one image in
                    the batch
    Returns:
        dictionary: dict, keys correspond to the columns in list_batch
    """
    if len(list_batch[0]) != 4:
        raise ValueError(f"Input must have shape (batch_size, 4), but got row of length {len(list_batch[0])}")

    dictionary = {0: torch.tensor([l[0] for l in list_batch]),
                  1: torch.tensor([l[1] for l in list_batch]),
                  2: torch.tensor([l[2] for l in list_batch]),
                  3: torch.tensor([l[3] for l in list_batch])}

    return dictionary


def calculate_accuracy(pred, target):
    """
    Calculates accuracy between prediction and target
    Args:
        pred: torch.Tensor, prediction tensor
        target: torch.Tensor, target tensor
    Returns:
        acc: torch.Tensor, accuracy value
    """
    # Predict class using highest probability assigned
    top_pred = pred.argmax(1, keepdim=True)
    # Compare prediction to ground truth
    correct = top_pred.eq(target.view_as(top_pred)).sum()
    # Get proportion of correct images
    acc = correct.float() / target.shape[0]
    return acc


def train_epoch(model, dataloader, optimizer, criterion, device, factor='colour'):
    """
    Implementation of one train epoch. Supports training on different classification tasks: 's1' (back shape), 's2'
    (front shape) and 'colour' (back-front colour pairs)
    Args:
        model: CNN, convolutional network
        dataloader: torch.utils.data.Dataloader, dataloader of train images
        optimizer: torch.optim.Optimizer, optimiser for updating model parameters
        criterion: function, loss function for training classifier
        device: torch.device, device to run the training on
        factor: str, factor for classifier being trained ('s1', 's2' or 'colour')
    Returns:
        tuple of floats: average loss over one epoch, average accuracy over one epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    model.to(device)

    for (x, y) in tqdm(dataloader, desc="Training", leave=False):
        # Move input to device
        x = x.to(device)
        y = list(y)

        # Zero gradients
        optimizer.zero_grad()

        # Convert captions into labels for each of the factors c1, s1, c2, s2
        for i in range(len(y)):
            y[i] = get_labels_from_prompt(y[i])
        y = get_dictionary(y)

        # Select the appropriate target labels based on the classification factor
        if factor == 's1':
            target = y[1]   # Back shape
        elif factor == 's2':
            target = y[3]   # Front shape
        elif factor == 'colour':
            target = convert_to_integer_tensor(torch.stack([y[0], y[2]], dim=1))    # Colour pairs

        # Compute predictions
        y_pred = model(x).to('cpu')

        # Evaluate loss and accuracy
        loss = criterion(y_pred, target)
        acc = calculate_accuracy(y_pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Add loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def test_epoch(model, dataloader, criterion, device, factor='colour'):
    """
    Implementation of one test epoch. Supported on different classification tasks: 's1' (back shape), 's2'
    (front shape) and 'colour' (back-front colour pairs)
    Args:
        model: CNN, convolutional network
        dataloader: torch.utils.data.Dataloader, dataloader of train images
        criterion: function, loss function for training classifier
        device: torch.device, device to run the training on
        factor: str, factor for classifier being trained ('s1', 's2' or 'colour')
    Returns:
        tuple of floats: average loss over one epoch, average accuracy over one epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for (x, y) in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move input to device
            x = x.to(device)
            y = list(y)

            # Convert captions into labels for each of the factors c1, s1, c2, s2
            for i in range(len(y)):
                y[i] = get_labels_from_prompt(y[i])
            y = get_dictionary(y)

            # Select the appropriate target labels based on the classification factor
            if factor == 's1':
                target = y[1]  # Back shape
            elif factor == 's2':
                target = y[3]  # Front shape
            elif factor == 'colour':
                target = convert_to_integer_tensor(torch.stack([y[0], y[2]], dim=1))  # Colour pairs

            # Compute predictions
            y_pred = model(x).to('cpu')

            # Evaluate loss and accuracy
            loss = criterion(y_pred, target)
            acc = calculate_accuracy(y_pred, target)

            # Add loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


if __name__ == "__main__":
    # Training set
    train_set1 = ShapesDataset(db_path=['data/complete_dataset'])
    train_set2 = ShapesDataset(db_path=['data/data_seed_0'], reorder=False)
    train_set3 = ShapesDataset(db_path=['data/data_seed_3'], reorder=False)
    test_set = ShapesDataset(db_path=['data/test_set'])
    train_set = torch.utils.data.ConcatDataset([train_set1, train_set2, train_set3])

    train_data, valid_data = torch.utils.data.random_split(train_set, [97200, 10800])

    pixel_size = 64
    BATCH_SIZE = 128
    INPUT_DIM = pixel_size * pixel_size * 3  # 4

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Colour model
    model = CNN(output_dims=6)
    SAVE_DIR = 'classifiers/classifier_colour.pt'

    # Optimiser and criterion
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 7  # 200
    best_valid_loss = float('inf')

    # Optimisation process
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc = test_epoch(model, valid_dataloader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'SAVE_DIR')

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_acc:.3f}')

    test_loss, test_acc = test_epoch(model, test_dataloader, criterion, device)
    print(f'\tTest Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')
