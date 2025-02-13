import torch
import os
import matplotlib.pyplot as plt
import torchvision


def freeze_model(model):
    """
    Freezes all parameters of a PyTorch model by setting requires_grad = False
    Args:
        model: torch.nn.Module, PyTorch model
    Returns:
    """
    for param in model.parameters():
        param.requires_grad = False


def create_vector_h_space(diffusion_model):
    """
    Creates vector to add to bottleneck layer of U-net
    Args:
        diffusion_model: Diffusion instance
    Returns:
        vec: torch.Tensor, vector of dimension 1x128x8x8
    """
    # Create vector to add with hook in h-space
    vec = torch.zeros(1, 128, 8, 8, device=diffusion_model.device, requires_grad=True, dtype=torch.float32)
    print(f'Norm of c_vec: {torch.norm(vec)}')
    return vec


def add_vector(vector):
    """
    Creates a forward hook that adds a specified vector to the output of a layer during the forward pass
    Args:
        vector: torch.Tensor, vector to be added to the forward pass, must have same dimension as output
    Returns:
        hook: function, hook that can be registered to a PyTorch module's forward pass
    """
    def hook(module, input, output):
        output += vector
    return hook


def get_average_last_n(list_vals, n):
    """
    Calculates average of last n elements of a list or tensor
    Args:
        list_vals: torch.Tensor, list of tensors to be averaged
        n: int, number of elements to average
    Returns:
        float, average of last n elements
    """
    last_n = list_vals[-n:]
    return sum(last_n) / len(last_n)


def find_concept_vector(images, starting_prompts, diffusion_model, lr=0.02, save_dir=None, space='h'):
    """
    Optimisation of steering vector
    Args:
        images: torch.Tensor, collection of images containing target concepts
        starting_prompts: list of str, starting prompts
        diffusion_model: Diffusion, diffusion model instance with device configurations
        lr: float, learning rate for optimisation process
        save_dir: str, name used to save outputs
        space: str, if 'p', optimisation on the prompt space; if 'h' optimisation on the h-space
    Returns:
        vec: torch.Tensor, optimised steering vector
    """
    # Freeze U-net parameters
    freeze_model(diffusion_model.unet)

    # Create concept vector in space where steering is to be implemented
    if space == 'p':
        # Set requires_grad=True for weights on p_vec (depends on space)
        diffusion_model.p_vec.requires_grad = True
        vec = diffusion_model.p_vec
    elif space == 'h':
        # Register hook to add vector in h-space
        vec = create_vector_h_space(diffusion_model)
        h = diffusion_model.unet.mid_block.register_forward_hook(add_vector(vec))
    else:
        raise ValueError(f"Unsupported space '{space}'. Choose 'h' for latent space or 'p' for prompt space.")

    # Optimizer
    optimizer = torch.optim.Adam([vec], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # Tensor to store results
    losses = torch.zeros(5000)

    # Move data to GPU (mapped to (-1, 1))
    x = images.to(diffusion_model.device) * 2 - 1

    print('Begin optimisation process')
    for i in range(5000):
        # Get timesteps
        t = torch.randint(0, diffusion_model.T, (x.shape[0],)).long().to(diffusion_model.device)

        # Calculate the MSE loss
        loss = diffusion_model.get_loss(x, t, starting_prompts)

        # Record stats
        losses[i] = loss.item()

        # Calculate gradients and update vector
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print results
        if i % 100 == 0:
            print(f'Iteration: {i}, Loss: {loss.item()}, Norm of vector: {torch.linalg.norm(vec)}')

    # Remove hook
    if space == 'h':
        h.remove()

    # Get final loss (averaged over last 10 iterations)
    average_loss = get_average_last_n(losses, 10)

    # Print final results
    print(f'Average loss across final 10 iterations: {average_loss}')
    print(f'Initial loss: {losses[0]}')
    print(f'Largest entry in vector: {torch.max(vec)}')

    # Save final results
    results_dict = {'final_loss': average_loss.item(),
                    'initial_loss': losses[0].item(),
                    'max_vector_entry': torch.max(torch.abs(vec)).item(),
                    'norm_vector': torch.linalg.vector_norm(vec).item(),
                    'mean_vector_entry': torch.mean(vec).item(),
                    'variance_vector_entry': torch.var(vec).item()}

    # Create save directory
    os.makedirs(f'{diffusion_model.local_dir}/{save_dir}/', exist_ok=True)

    # Save results_dict
    torch.save(results_dict, f'{diffusion_model.local_dir}/{save_dir}/concept_vectors_{space}.json')

    # Save concept vector
    vec.requires_grad = False
    torch.save(vec, os.path.join(diffusion_model.local_dir, f'{save_dir}/concept_vectors_{space}.pt'))

    return vec


def sample_with_concept_vector(diffusion_model, prompt, concept_vector, num_samples=100, test=None, space='h'):
    """
    Samples images by adding steering vector in corresponding space
    Args:
        diffusion_model: Diffusion, diffusion model instance with device configurations
        prompt: str, starting prompt
        concept_vector: torch.Tensor, optimised vector
        num_samples: int, number of images to sample
        test: int, test number corresponding to the starting prompt and target concept combination
        space: str, either 'p' for steering on the prompt space or 'h' for steering on the h-space
    Returns:
        images: torch.Tensor, images sampled with steering
    """
    # Set requires_grad to False
    concept_vector.requires_grad = False

    # Add concept vector in space where steering is to be implemented
    if space == 'p':
        # Set diffusion_model.p_vec to concept vector
        diffusion_model.p_vec = concept_vector
    elif space == 'h':
        # Register hook to add vec at bottleneck layer of U-net
        h = diffusion_model.unet.mid_block.register_forward_hook(add_vector(concept_vector))
    else:
        raise ValueError(f"Unsupported space '{space}'. Choose 'h' for latent space or 'p' for prompt space.")

    # Sample with concept vector
    images = diffusion_model.sample(num_samples, prompt, return_img=True, save_plots=False)

    # Remove hook
    if space == 'h':
        h.remove()

    # Create save directory
    plots_dir = os.path.join(diffusion_model.local_dir, f'{test}', f'samples_{prompt}')
    os.makedirs(plots_dir, exist_ok=True)

    # Show the results
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'{prompt}')
    ax.imshow(torchvision.utils.make_grid(images, nrow=10, pad_value=0.5).permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'{plots_dir}/' + f'grid_samples_concept_edit_{space}.png')
    print(f'Saved samples to {plots_dir}')
    plt.close()

    return images
