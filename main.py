import argparse
import torch
import numpy as np
import json
from text_to_image_model import Diffusion
from dataset import ShapesDataset
from experiments import run_experiments, sample_prompts
from evaluate_images import evaluate_accuracy_batch, load_classifiers


def main(args):
    # Seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Diffusion model
    model = Diffusion(args)

    # Validate train_dir based on experiment number
    if args.experiment_num in [1, 3] and args.train_dir != 'complete_dataset':
        raise ValueError("For experiments 1 and 3, args.train_dir must be 'complete_dataset'.")

    if args.task == 'train':
        # Train and test datasets
        train_set = ShapesDataset(db_path=[f'data/{args.train_dir}'])
        test_set = ShapesDataset(db_path=['data/test_set'])
        # Train model
        model.train_diffusion(train_set, test_set)

    elif args.task == 'sample':
        # Load checkpoint
        model.load_checkpoint(args.ckpt_sample)

        # Sample input prompt
        if args.prompt:
            # Sample using prompt
            prompt = args.prompt
            imgs = model.sample(100, prompt, return_img=True)

            # Load pre-trained classifiers for evaluation
            model_list = load_classifiers()
            for classifier in model_list:
                classifier.eval()
            # Use classifiers to evaluate accuracy
            evaluate_accuracy_batch(imgs, model_list, ['colour', 's1', 's2'], prompt,
                                    save_directory=f'{model.local_dir}/samples_{prompt}')

        # If implementing experiments 1-4, sample the target concepts in a configuration dict
        else:
            # Load config_dict
            with open(args.config_dict_path, 'r') as f:
                config_dict = json.load(f)

            # Adjust parameters for experiment 3
            if args.experiment_num == 3:
                model.p_vec = torch.zeros(1, 10, 512, device=model.device, requires_grad=False, dtype=torch.float32)
                config_dict = config_dict[args.test]

            # Convert config_dict keys to int
            config_dict = {int(k): v for k, v in config_dict.items()}

            # Sample target concepts in config_dict
            sample_prompts(model, config_dict)

    elif args.task == 'steer':
        # Load checkpoint
        model.load_checkpoint(args.ckpt_sample)

        # Load config_dict
        with open(args.config_dict_path, 'r') as f:
            config_dict = json.load(f)

        # Select config dict for experiment 3
        if args.experiment_num == 3:
            config_dict = config_dict[args.test]

        # Convert config_dict keys to int
        config_dict = {int(k): v for k, v in config_dict.items()}

        # Implement steering
        for test in range(len(config_dict)):
            run_experiments(model, test, config_dict, space=args.space)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train', type=str, choices=['train', 'sample', 'steer'],
                        help='task to implement')
    parser.add_argument('--experiment_num', default=1, type=int, help='experiment number in results')
    parser.add_argument('--test', default=0, type=int, help='test index for experiment 3')
    parser.add_argument('--space', default='p', type=str, choices=['p', 'h'],
                        help='space where steering is implemented')
    parser.add_argument('--config_dict_path', default='', type=str,
                        help='path from which to load dictionary containing target and starting prompts')
    parser.add_argument('--train_dir', default='complete_dataset', type=str,
                        help='path to train set directory (within data directory)')
    parser.add_argument('--prompt', default=None, type=str,
                        help='prompt for sampling (for sample task)')
    parser.add_argument('--T', default=1000, type=int, help='number of noising steps')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training model')
    parser.add_argument('--max_epoch', default=70, type=int, help='number of training epochs')
    parser.add_argument('--ckpt_sample', default='last', help='checkpoint used for sampling and steering')
    parser.add_argument('--optim', default='adam', choices=['adam', 'rmsprop'], type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    args = parser.parse_args()
    main(args)
