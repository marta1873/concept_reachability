import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision


# algorithm 1
def generate_data(prompts_plus, prompts_minus, diffusion_model, num_samples=200):

    # if more than one prompt plus/minus, randomly generate the order in which they are sampled
    n = np.random.randint(0, len(prompts_plus), size=num_samples)

    # get list with prompts plus & minus
    prompt_plus = [prompts_plus[i] for i in n]
    prompt_minus = [prompts_minus[i] for i in n]

    # sample images with prompt plus - these contain the target end point
    images_plus = diffusion_model.sample(batch_size=len(prompt_plus), prompt=prompt_plus, return_img=True)

    return images_plus, prompt_minus


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def add_vector(vector):
    # the hook signature
    def hook(module, input, output):
        output += vector
    return hook


def plot_losses_and_norm_evolution(losses, norms, plots_dir, filename, regularisation=None, space='h'):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].set_title('Loss')
    ax[0].plot(losses)

    ax[1].set_title('Norm')
    ax[1].plot(norms)

    if filename is None:
        plt.savefig(f'{plots_dir}/' + f'loss_norm_evolution_{space}_{regularisation}.png')
        plt.close()
    else:
        dir_path = os.path.join(plots_dir, filename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(f'{plots_dir}/' + f'{filename}/loss_norm_evolution_{space}_{regularisation}.png')
        plt.close()


def get_max(list_vals):
    return max(list_vals)


def get_average_last_n(list_vals, n):
    last_n = list_vals[-n:]
    return sum(last_n) / len(last_n)


def create_vector(diffusion_model):
    # create vector to add with hook in h space
    vec = torch.zeros(1, 128, 8, 8, device=diffusion_model.device, requires_grad=True, dtype=torch.float32)
    print(f'Norm of c_vec: {torch.norm(vec)}')
    return vec


# algorithm 2
def find_concept_vector(images, prompts, diffusion_model, lr=0.01, filename=None, regularisation=None, space='h'):
    # freeze unet parameters
    freeze_model(diffusion_model.unet)

    # register hook to add c_vec at relevant layer
    if space == 'h':
        # create concept vector in corresponding space
        vec = create_vector(diffusion_model)
        h = diffusion_model.unet.mid_block.register_forward_hook(add_vector(vec))
    else:
        diffusion_model.p_vec.requires_grad = True
        vec = diffusion_model.p_vec

    # optimizer
    optimizer = torch.optim.Adam([vec], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # lists to store results
    norms = []
    losses = []
    vector_dict = {}

    # Begin optimisation process
    print('Begin optimisation process')
    for i in range(50):
        print(f'Iteration: {i}')
        # get some data and prepare the corrupted version
        x = images.to(diffusion_model.device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))

        # get timesteps
        t = torch.randint(0, diffusion_model.T, (x.shape[0],)).long().to(diffusion_model.device)

        # calculate the MSE loss
        loss_diff = diffusion_model.get_loss(x, t, prompts)

        # get norm of concept vector
        norm = torch.linalg.vector_norm(vec)

        # regularisation
        if regularisation:
            loss = loss_diff + regularisation * norm
        else:
            loss = loss_diff

        # store loss
        print(f'Loss: {loss.item()}')
        losses.append(loss.item())

        # store norm
        norms.append(norm.cpu().detach().numpy())

        # calculate gradients of the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print results
        print(f'Norm of c_vec: {norm}')
        print(f'Mean of entries in c_vec: {torch.mean(vec)}')
        print(f'Variance of entries in c_vec: {torch.var(vec)}')

        # get checkpoints
        if i in [50, 500, 1000, 2500, 5000, 7500] and space ==' h':
            vector_dict[f'{i}'] = vec.detach().clone()

    # remove hook
    if space == 'h':
        h.remove()

    # get final results
    average_loss = get_average_last_n(losses, 10)

    # save results
    results_dict = {'final_loss': average_loss, 'initial_loss': losses[0], 'max_vector_entry': torch.max(torch.abs(vec)),
                    'norm_vector': norm, 'mean_vector_entry': torch.mean(vec), 'variance_vector_entry': torch.var(vec)}

    # create directory
    if not os.path.exists(f'{diffusion_model.local_dir}/{filename}/'):
        os.makedirs(f'{diffusion_model.local_dir}/{filename}/')
    torch.save(results_dict, f'{diffusion_model.local_dir}/{filename}/concept_vectors_{space}_{regularisation}.json')
    if space == 'h':
        torch.save(vector_dict, f'{diffusion_model.local_dir}/{filename}/vector_dict.json')

    # print results
    print(f'Average loss last 10 iterations: {average_loss}')
    print(f'Initial loss: {losses[0]}')
    print(f'Largest entry in vector: {torch.max(vec)}')

    # plot results
    plot_losses_and_norm_evolution(losses, norms, diffusion_model.local_dir, filename=filename,
                                   regularisation=regularisation, space=space)

    # save concept vector
    vec.requires_grad = False
    torch.save(vec, os.path.join(diffusion_model.local_dir, f'{filename}/concept_vectors_{space}_{regularisation}.pt'))

    return vec


def sample_with_concept_vector(diffusion_model, prompt, concept_vector, num_samples=100, test=None, regularisation=None,
                               space='h'):
    # set requires_grad to False
    concept_vector.requires_grad = False

    # add concept vector in space
    if space == 'h':
        # register hook to add c_vec at bottleneck layer
        h = diffusion_model.unet.mid_block.register_forward_hook(add_vector(concept_vector))
    else:
        # set p_vec at prompt space to concept vector
        diffusion_model.p_vec = concept_vector

    # sample with concept vector
    images = diffusion_model.sample(num_samples, prompt, return_img=True)

    # remove hook
    if space == 'h':
        h.remove()

    # create results path
    plots_dir = os.path.join(diffusion_model.local_dir, f'{test}', f'samples_{prompt}')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Show the results
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'{prompt}')
    ax.imshow(torchvision.utils.make_grid(images, nrow=10, pad_value=0.5).permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'{plots_dir}/' + f'grid_samples_concept_edit_{space}_{regularisation}.png')
    print(f'Saved samples to {plots_dir}')
    plt.close()
