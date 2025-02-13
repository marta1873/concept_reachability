import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ShapesDataset(Dataset):
    def __init__(self, db_path=['data/complete_dataset'], reorder=True):
        """
        PyTorch Dataset for loading the coloured shapes dataset
        Args:
            db_path: list of str, list containing path to directory of images
            reorder: bool, if True, images are normalized to [0,1] and channels are moved to PyTorch format (C, H, W)
        """
        self.db_path = db_path
        self.db_files = [os.listdir(path) for path in db_path]
        self.images = None
        self.captions = None
        self.reorder = reorder

    def load_data(self):
        """
        Loads data from directory of images
        """
        images_list = []
        captions_list = []
        for path in self.db_path:
            # Iterate through files in path
            for file in sorted(self.db_files[self.db_path.index(path)],
                               key=lambda x: int(x.split('h')[1].split('.')[0])):

                # Get images and captions
                filename_path = os.path.join(path, file)
                img = np.load(filename_path)['images']
                captions = np.load(filename_path)['captions']

                # Add images and captions to list
                images_list.append(img)   # Scale image pixels to the range [0,1]
                captions_list.append(captions)

            # Convert images to a NumPy array
            images_np = np.concatenate(images_list, axis=0)  # Shape: (N, H, W, C)

            # Convert images to tensor
            image_tensor = torch.tensor(images_np, dtype=torch.float32)

            if self.reorder:
                self.images = torch.permute(image_tensor / 255.0, (0, 3, 1, 2))  # Normalise + move channels
            else:
                self.images = image_tensor

            self.captions = np.concatenate(captions_list, axis=0)

    def __len__(self):
        if self.images is None:
            self.load_data()
        return len(self.images)

    def __getitem__(self, idx):
        if self.images is None:
            self.load_data()
        image = self.images[idx]
        caption = self.captions[idx]
        return image, caption


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ShapesDataset(db_path=['data/complete_dataset'], reorder=True)
    index = 0

    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        index = np.random.randint(len(dataset))
        image, label = dataset.__getitem__(index)

        #ax[i].set_title(label)
        ax[i].imshow(image.permute(1, 2, 0))
        ax[i].axis('off')

        index += 1

    plt.tight_layout()
    plt.show()
