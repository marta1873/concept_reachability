import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDPMScheduler
from conditional_unet import build_unet
from transformers import AutoTokenizer, T5EncoderModel
from experiment_3 import transform_captions_batch


def set_device():
    """
    Sets device for PyTorch computations
    Returns:
        device: torch.device, 'cuda', 'mps' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


def create_save_directory(experiment_num, train_dir, seed, test, optimizer_name, lr):
    """
    Generates a local directory path based on the experiment number, training directory, seed, and test condition
    Args:
        experiment_num: int, experiment number identifier (1: baseline, 2: scarcity, 3: underspecification, 4: biases)
        train_dir: str, name of directory storing data to train model
        seed: int, random seed for reproducibility
        test: int, test identifier, used for experiment 3
    Returns:
        local_dir: str, local directory path
    """

    if experiment_num == 1 or experiment_num == 2 or experiment_num == 4:
        local_dir = os.path.join(f'experiment_{experiment_num}', f'diffusion_{train_dir}_{optimizer_name}_{lr}',
                                 f'seed_{seed}')
    elif experiment_num == 3:
        local_dir = os.path.join(f'experiment_{experiment_num}',
                                 f'diffusion_{train_dir}_{optimizer_name}_{lr}/test_{test}', f'seed_{seed}')
    else:
        local_dir = os.path.join(f'diffusion_{optimizer_name}_{lr}', f'seed_{seed}')

    # Create directory if it does not already exist
    os.makedirs(local_dir, exist_ok=True)

    return local_dir


class Diffusion(nn.Module):
    def __init__(self, args):
        super(Diffusion, self).__init__()

        # Device
        self.device = set_device()

        # Scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

        # U-net architecture
        self.unet = build_unet(num_channels=3).to(self.device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-small").to(self.device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer_name = args.optim
        self.optim = None

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.max_epochs = args.max_epoch
        self.T = args.T
        self.global_iter = 0

        # Directory to save files
        self.local_dir = create_save_directory(args.experiment_num, args.train_dir, args.seed, args.test,
                                               self.optimizer_name, self.lr)

        # Name of directory to store checkpoints
        self.ckpt_dir = 'checkpoints'

        # Vector to add in prompt space (set to 0 unless steering on the prompt space)
        self.p_vec = torch.zeros(1, 10, 512, device=self.device, requires_grad=False, dtype=torch.float32)

        # Caption modifications (used in Experiment 3: underspecification)
        self.transform_caption = None
        self.option = None
        if args.experiment_num == 3 and args.test != 0:
            self.transform_caption = transform_captions_batch
            self.option = args.test
            # Adjust dimension of p_vec to dimension of encoding of caption or prompt
            caption_lengths = [10, 9, 10, 9, 10, 5, 4]
            self.p_vec = torch.zeros(1, caption_lengths[args.test], 512, device=self.device, requires_grad=False,
                                     dtype=torch.float32)

    def encode_text(self, y):
        """
        Tokenizes input text and encodes it using a pre-trained text encoder
        Args:
            y: list of str, a list of strings representing the text prompts
        Returns:
            last_hidden_states: torch.Tensor, encoded text representations
        """
        # Pass to tokenizer
        inputs = self.tokenizer(y, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Pass to text encoder
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids.to(self.device),
                                        attention_mask=attention_mask.to(self.device))
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def predict_noise(self, x, t, prompt):
        """
        Predicts added noise using a U-net
        Args:
            x: torch.Tensor, input tensor containing batch of noisy images, shape (batch_size, channels, height, width)
            t: torch.Tensor, timestep tensor representing the diffusion step of each image in x, shape (batch_size, )
            prompt: list of str, list of text captions of each image in x
        Returns:
            torch.Tensor, the predicted noise tensor from the U-net, of the same dimension as x
        """
        # Encode text prompts
        last_hidden_states = self.encode_text(prompt)

        # Predict noise
        return self.unet(x, t, encoder_hidden_states=last_hidden_states + self.p_vec).sample

    def get_loss(self, x, t, prompt):
        """
        Computes the mean squared error loss between the true noise and the predicted noise during the diffusion process
        Args:
            x: torch.Tensor, clean batch of images, shape (batch_size, channels, height, width)
            t: torch.Tensor, timestep tensor representing the diffusion step for each image in x, shape (batch_size, )
            prompt: list of str, list of text captions of each image in x
        Returns:
            torch.Tensor, computed loss value (scalar)
        """
        # add noise
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, t)

        # get model prediction
        noise_pred = self.predict_noise(noisy_x, t, prompt)

        # return loss
        return F.mse_loss(noise, noise_pred, reduction='sum').div(noisy_x.size(0))

    def train_diffusion(self, train_set, test_set):
        """
        Training function of diffusion model
        Args:
            train_set: torch.utils.data.Dataset, train dataset
            test_set: torch.utils.data.Dataset, test dataset
        Returns:
        """
        # Set U-net to training mode
        self.unet.train()

        # Dataloader
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        # Optimizer
        if self.optimizer_name == 'adam':
            self.optim = Adam(self.unet.parameters(), self.lr)
        elif self.optimizer_name == 'rmsprop':
            self.optim = RMSprop(self.unet.parameters(), self.lr)
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'. Choose 'adam' or 'rmsprop'.")
        print(f'Using {self.optimizer_name} optimizer')

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.98)

        print('Begin training')

        losses = []
        test_losses = []
        test_its = []

        # Training loop
        for epoch in range(self.max_epochs):
            print(f'Epoch {epoch}')
            for x, y in tqdm(dataloader):
                # Pass data to device and map to (-1, 1)
                x = x.to(self.device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))

                # Get timesteps
                t = torch.randint(0, self.T, (x.shape[0],)).long().to(self.device)

                # Experiment 3
                if self.transform_caption:
                    y = self.transform_caption(y, self.option)

                # Calculate the loss
                loss = self.get_loss(x, t, y)

                # Backpropagation
                self.optim.zero_grad()
                loss.backward()

                # Update parameters
                self.optim.step()

                # Store loss values
                losses.append(loss.item())

                # Update global iteration count
                self.global_iter += 1

            print(f'Loss: {losses[-1]:.6f}')

            # Update scheduler
            print(f'Learning rate: {scheduler.get_last_lr()}')
            scheduler.step()

            if epoch % 10 == 0:
                # Save checkpoint
                self.save_checkpoint(str(self.global_iter))

                # Evaluate denoising ability on test set
                test_losses.append(self.test_diffusion(test_set))
                test_its.append(self.global_iter)

            # Print out the average of the last 100 loss values
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

        # Save final model
        self.save_checkpoint('last')

        # Store final loss
        test_losses.append(self.test_diffusion(test_set))
        test_its.append(self.global_iter)

        # Plot loss curve
        plt.title('Training and Test Loss Curve')
        plt.plot(losses, label='Train loss')
        plt.plot(test_its, test_losses, label='Test loss', marker='o')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.local_dir}/loss.png')
        plt.close()

    def test_diffusion(self, testset):
        """
        Evaluate the denoising performance of the diffusion model on a test dataset
        Args:
            testset: torch.utils.data.Dataset, test dataset
        Returns:
            float, average loss over the test dataset
        """
        # Set U-net to evaluation mode
        self.unet.eval()

        # Dataloader
        dataloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers)

        loss_total = 0

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                # Pass data to device and map to (-1, 1)
                x = x.to(self.device) * 2 - 1

                # Get timesteps
                t = torch.randint(0, self.T, (x.shape[0],)).long().to(self.device)

                # Experiment 3
                if self.transform_caption:
                    y = self.transform_caption(y, self.option)

                # Evaluate loss on test set
                loss = self.get_loss(x, t, y).to('cpu')

                # Add to global loss
                loss_total += loss.item()

        print(f'Test loss average: {loss_total / len(dataloader)}')
        return loss_total / len(dataloader)

    def sample(self, batch_size, prompt, return_img=False, save_plots=True):
        """
        Generates images from noise using the diffusion model
        Args:
            batch_size: int, number of images to sample
            prompt: str or list of str, text prompts guiding image generation. If using a list of str input, should be
                    of length batch_size
            return_img: bool, if True, returns the generated images as a tensor
            save_plots: bool, if True, saves the generated images as a png file
        Returns:
            torch.Tensor (generated images) or None
        """
        # Set U-Net to evaluation mode
        self.unet.eval()

        with torch.no_grad():
            # Get random noise
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)

            # Encode prompt
            last_hidden_states = self.encode_text(prompt)

            # If only one prompt is given, repeat the embedding to match batch size
            if isinstance(prompt, str):
                y = last_hidden_states.repeat(batch_size, 1, 1)
            elif isinstance(prompt, list):
                # Ensure the number of prompts matches the batch size
                if len(prompt) != batch_size:
                    raise ValueError(f"Expected {batch_size} prompts but received {len(prompt)}.")
                y = last_hidden_states  # Use the list of text embeddings directly
                prompt = prompt[0]  # Use the first prompt for directory naming
            else:
                raise TypeError("Prompt should be either a string or a list of strings.")

            # Sampling loop
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                # Get model prediction
                residual = self.unet(x, t.to(self.device), y + self.p_vec).sample

                # Update sample with step
                x = self.noise_scheduler.step(residual, t, x).prev_sample

            # Normalise to [0, 1] and past to cpu
            x = (x.detach().cpu().clip(-1, 1) + 1) / 2

        if save_plots:
            # Define directory for storing results
            plots_dir = os.path.join(self.local_dir, f'samples_{prompt}')
            os.makedirs(plots_dir, exist_ok=True)

            # Plot images and save to directory
            fig, ax = plt.subplots(1, 1)
            ax.set_title(f'{prompt}')
            ax.imshow(torchvision.utils.make_grid(x, nrow=10, pad_value=0.5).permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(f'{plots_dir}/' + 'grid_samples.png')
            print(f'Saved samples to {plots_dir}')
            plt.close()

        # Return image tensor
        if return_img:
            return x

    def save_checkpoint(self, filename):
        """
        Saves checkpoint of model weights
        Args:
            filename: str, name of file to write checkpoint
        Returns:
        """
        # Create dictionary with state dictionary of U-net
        model_states = {'model': self.unet.state_dict()}

        # Create dictionary with model_states and global iteration step count
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  }

        # Create directory to store checkpoints
        dir_path = os.path.join(self.local_dir, self.ckpt_dir)
        os.makedirs(dir_path, exist_ok=True)

        # Save checkpoints
        file_path = os.path.join(self.local_dir, self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        """
        Loads checkpoint of model weights
        Args:
            filename: str, name of file where checkpoint is stored
        Returns:
        """
        # Cet directory to load checkpoints
        file_path = os.path.join(self.local_dir, self.ckpt_dir, filename)

        # Load checkpoints
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            # Global iteration count
            self.global_iter = checkpoint['iter']
            # Load U-net state dictionary
            self.unet.load_state_dict(checkpoint['model_states']['model'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
