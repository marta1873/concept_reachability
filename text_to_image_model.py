import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
from torch.optim import Adam, RMSprop
from torch import nn
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from dataset import *
from conditional_unet import build_unet
from transformers import AutoTokenizer, T5EncoderModel


class Diffusion(nn.Module):
    def __init__(self, args):
        super(Diffusion, self).__init__()

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # scheduler
        if args.scheduler == 'ddpm':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        elif args.scheduler == 'ddim':
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=200, beta_schedule="squaredcos_cap_v2")
        elif args.scheduler == 'invddim':
            self.noise_scheduler = DDIMInverseScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

        # unet architecture
        self.unet = build_unet(size=args.u_net_size, num_channels=3).to(self.device)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        print('Tokenizer loaded')

        # text encoder
        self.text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-small").to(self.device)
        # freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print('Text encoder loaded')

        # optimizer
        self.optimizer_name = args.optim
        self.optim = None

        # traaining hyperparameters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.max_epochs = args.max_epoch
        self.T = args.T
        self.global_iter = 0

        # define directory to save files
        if args.experiment_num:
            start_point = f'experiment_{args.experiment_num}'
        else:
            start_point = ''

        if args.experiment_num == 2 or args.experiment_num == 3 or args.experiment_num == 4:
            self.local_dir = os.path.join(start_point,
                                          f'diffusion_b_{self.optimizer_name}_{self.lr}_{args.u_net_size}_{args.train_dir}',
                                          f'seed_{args.seed}')
        else:
            self.local_dir = os.path.join(start_point, f'diffusion_b_{self.optimizer_name}_{self.lr}_{args.u_net_size}',
                                          f'seed_{args.seed}')

        # name of direcotry to store checkpoints
        self.ckpt_dir = 'checkpoints'

        # vector to add in prompt space (set to 0 unless optimising concept vector or adding it for sampling)
        self.p_vec = torch.zeros(1, 10, 512, device=self.device, requires_grad=False, dtype=torch.float32)

    def encode_text(self, y):
        # pass to tokenizer
        inputs = self.tokenizer(y, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # pass to text encoder
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids.to(self.device),
                                        attention_mask=attention_mask.to(self.device))
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def predict_noise(self, x, t, prompt):
        # Shape of x:
        # bs, ch, w, h = x.shape

        # text embedding from prompt
        last_hidden_states = self.encode_text(prompt)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.unet(x, t, encoder_hidden_states=last_hidden_states + self.p_vec).sample  # (bs, 1, 28, 28)

    def get_loss(self, x, t, prompt):
        # add noise
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, t)

        # get model prediction
        noise_pred = self.predict_noise(noisy_x, t, prompt)  # model(x_noisy, t)

        # return loss
        return F.mse_loss(noise, noise_pred, reduction='sum').div(noisy_x.size(0))

    def train_diffusion(self, trainset, testset):
        # unet
        self.unet.train()

        # dataloader
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        # optimizer
        if self.optimizer_name == 'adam':
            self.optim = Adam(self.unet.parameters(), self.lr)
        elif self.optimizer_name == 'rmsprop':
            self.optim = RMSprop(self.unet.parameters(), self.lr)
        print(f'Using {self.optimizer_name} optimizer')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.98)

        losses = []
        test_losses = []
        test_its = []

        print('Begin training')

        # The training loop
        for epoch in range(self.max_epochs):
            print(epoch)
            for x, y in tqdm(dataloader):
                # Get some data and prepare the corrupted version
                x = x.to(self.device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))

                # get timesteps
                t = torch.randint(0, self.T, (x.shape[0],)).long().to(self.device)

                # Calculate the loss
                loss = self.get_loss(x, t, y)  # How close is the output to the noise

                # Backprop and update the params:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Store the loss for later
                losses.append(loss.item())

                # update global iteration count
                self.global_iter += 1

            print(f'Loss: {losses[-1]}')

            # update scheduler
            print(f'Learning rate: {scheduler.get_last_lr()}')
            scheduler.step()

            if epoch % 10 == 0:
                # save checkpoint
                self.save_checkpoint(str(self.global_iter), silent=False)

                # evaluate denoising ability on test set
                test_losses.append(self.test_diffusion(testset))
                test_its.append(self.global_iter)

            # Print out the average of the last 100 loss values to get an idea of progress:
            avg_loss = sum(losses[-100:]) / 100
            print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

        # final model
        self.save_checkpoint('last', silent=False)
        test_losses.append(self.test_diffusion(testset))
        test_its.append(self.global_iter)

        # View the loss curve
        plt.plot(losses, label='train loss')
        plt.plot(test_its, test_losses, label='test loss')
        plt.legend()
        plt.savefig(f'{self.local_dir}/loss.png')

    def test_diffusion(self, testset):
        # unet
        self.unet.eval()

        # dataloader
        dataloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers)

        # set loss to 0
        loss_total = 0

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                # Get some data and prepare the corrupted version
                x = x.to(self.device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))

                # get timesteps
                t = torch.randint(0, self.T, (x.shape[0],)).long().to(self.device)

                # evaluate loss on test set
                loss = self.get_loss(x, t, y).to('cpu')  # How close is the output to the noise

                # Store the loss for later
                loss_total += loss.item()

        print(f'Test loss average: {loss_total / len(dataloader)}')
        return loss_total / len(dataloader)

    def sample(self, batch_size, prompt, return_img=False):
        self.unet.eval()

        with torch.no_grad():
            # Get random noise
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)

            # prompt
            last_hidden_states = self.encode_text(prompt)

            # if only one prompt is given, make a copy of the prompt to match the batch size of samples
            if type(prompt) is not list:
                y = last_hidden_states.repeat(batch_size, 1, 1)
            # otherwise use list of prompts inputted
            else:
                y = last_hidden_states

            # Sampling loop
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                # Get model pred
                residual = self.unet(x, t.to(self.device), y + self.p_vec).sample

                # Update sample with step
                x = self.noise_scheduler.step(residual, t, x).prev_sample

            # return image tensor
            if return_img:
                return (x.detach().cpu().clip(-1, 1) + 1) / 2

        # create directory for storing results
        plots_dir = os.path.join(self.local_dir, f'samples_{prompt}')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # plot images and save to directory
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'{prompt}')
        ax.imshow(
            torchvision.utils.make_grid((x.detach().cpu().clip(-1, 1) + 1) / 2, nrow=10, pad_value=0.5).permute(1, 2,
                                                                                                                0))
        plt.axis('off')
        plt.savefig(f'{plots_dir}/' + 'grid_samples.png')
        print(f'Saved samples to {plots_dir}')
        plt.close()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'model': self.unet.state_dict(), }
        #optim_states = {'optim': self.optim.state_dict(), }
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  #'optim_states': optim_states
                  }

        # create directory to store checkpoints
        dir_path = os.path.join(self.local_dir, self.ckpt_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save checkpoints
        file_path = os.path.join(self.local_dir, self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        # get directory to load checkpoints
        file_path = os.path.join(self.local_dir, self.ckpt_dir, filename)

        # load checkpoints
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.global_iter = checkpoint['iter']
            self.unet.load_state_dict(checkpoint['model_states']['model'])
            #self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
