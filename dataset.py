import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision.utils import save_image


class ShapesDataset(Dataset):
    def __init__(self, db_path=['complete_dataset'], exclude=[]):
        self.db_path = db_path
        self.db_files = [os.listdir(path) for path in db_path]
        self.images = None
        self.labels = None
        self.batch = None
        self.exclude = exclude

    def load_data(self):
        images_list = []
        captions_list = []
        for path in self.db_path:
            # iterate through files in path
            for file in sorted(self.db_files[self.db_path.index(path)],
                               key=lambda x: int(x.split('h')[1].split('.')[0])):
                # skip file if we want to exclude certain batches
                if int(file.split('h')[1].split('.')[0]) in self.exclude:
                    print(f'File {file} not loaded')
                    continue

                # open file
                filename_path = os.path.join(path, file)

                # get images and captions
                img = np.load(filename_path)['images']
                labels = np.load(filename_path)['captions']

                # add images and captions to list, scale image pixels to the range [0,1]
                images_list.append(img / 255)
                captions_list.append(labels)
        # convert images to torch tensor
        image_tensor = torch.from_numpy(np.concatenate(images_list, axis=0)).to(torch.float32)

        # set final values of images captions
        self.images = torch.movedim(image_tensor, 3, 1)  # move channel dimension to the right position
        self.labels = np.concatenate(captions_list, axis=0)

    def __len__(self):
        if self.images is None:
            self.load_data()
        return len(self.images)

    def __getitem__(self, idx):
        if self.images is None:
            self.load_data()
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ShapesDataset(db_path=['conditional_unet_dataset_behind2'])

    index = 40000

    for i in range(10):
        image, label = dataset.__getitem__(index)

        #plt.title(label)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        plt.close()

        index += 1
