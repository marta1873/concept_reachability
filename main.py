import argparse
import torch
import numpy as np
from text_to_image_model import Diffusion
from dataset import ShapesDataset
from experiments import run_experiments, sample_prompts
from experiment_1 import config_dict
from experiment_2 import config_dict_2_1
from experiment_3 import (config_dict_3_0, config_dict_3_1, config_dict_3_2, config_dict_3_3, config_dict_3_4,
                          config_dict_3_5, config_dict_3_6)
from experiment_4 import config_dict_4_1
from evaluate_images import evaluate_accuracy_batch, load_classifiers


def main(args):
    # Seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Diffusion model
    model = Diffusion(args)

    if args.task == 'train':
        # Train and test datasets
        train_set = ShapesDataset(db_path=[f'data/{args.train_dir}'])
        test_set = ShapesDataset(db_path=['data/test_set'])
        # Train model
        model.train_diffusion(train_set, test_set)

    elif args.task == 'sample':
        # Load checkpoint
        model.load_checkpoint(args.ckpt_sample)
        # Sample using prompt
        prompt = args.prompt
        imgs = model.sample(1, prompt, return_img=True)

        # Load pre-trained classifiers for evaluation
        model_list = load_classifiers()
        for classifier in model_list:
            classifier.eval()
        # Use classifiers to evaluate accuracy
        evaluate_accuracy_batch(imgs, model_list, ['colour', 's1', 's2'], prompt,
                                save_directory=f'{model.local_dir}/samples_{prompt}')

    elif args.task == 'experiment':
        # Baseline
        if args.experiment_num == 1:
            # Train model
            train_set = ShapesDataset(db_path=[f'data/complete_dataset'])
            test_set = ShapesDataset(db_path=['data/test_set'])
            model.train_diffusion(train_set, test_set)
            model.load_checkpoint(args.ckpt_sample)

            # Implement steering using prompts in config_dict
            for test in range(24):
                run_experiments(model, test, config_dict, space=args.space)

            # Sample target concepts in config_dict
            sample_prompts(model, config_dict)

        # Scarcity
        elif args.experiment_num == 2:
            # Train model
            train_set = ShapesDataset(db_path=[f'data/{args.train_dir}'])
            test_set = ShapesDataset(db_path=['data/test_set'])
            model.train_diffusion(train_set, test_set)
            model.load_checkpoint(args.ckpt_sample)

            # Implement steering using prompts in config_dict
            for test in range(17):
                run_experiments(model, test, config_dict_2_1, space=args.space)

            # Sample target concepts in config_dict
            sample_prompts(model, config_dict_2_1)

        # Underspecification
        elif args.experiment_num == 3:
            # Train model
            train_set = ShapesDataset(db_path=[f'data/complete_dataset'])
            test_set = ShapesDataset(db_path=['data/test_set'])
            model.train_diffusion(train_set, test_set)
            model.load_checkpoint(args.ckpt_sample)

            # Define config_dict according to test (level of specification explored)
            config_dicts_3 = [config_dict_3_0, config_dict_3_1, config_dict_3_2, config_dict_3_3, config_dict_3_4,
                              config_dict_3_5, config_dict_3_6]
            config_dict_3 = config_dicts_3[args.test]

            # Implement steering using prompts in config_dict
            for test in range(10):
                run_experiments(model, test, config_dict_3, space=args.space)

            # Sample target concepts in config_dict
            model.p_vec = torch.zeros(1, 10, 512, device=model.device, requires_grad=False, dtype=torch.float32)
            sample_prompts(model, config_dict_3)

        # Biases
        elif args.experiment_num == 4:
            # Train model
            train_set = ShapesDataset(db_path=[f'data/{args.train_dir}'])
            test_set = ShapesDataset(db_path=['data/test_set'])
            model.train_diffusion(train_set, test_set)
            model.load_checkpoint(args.ckpt_sample)

            # Implement steering using prompts in config_dict
            for test in range(18):
                run_experiments(model, test, config_dict_4_1, space=args.space)

            # Sample target concepts in config_dict
            sample_prompts(model, config_dict_4_1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='experiment', type=str, choices=['train', 'sample',
                        'experiment'], help='task to implement')
    parser.add_argument('--experiment_num', default=1, type=int, help='experiment number in results')
    parser.add_argument('--test', default=0, type=int, help='test index for experiment 3')
    parser.add_argument('--space', default='h', type=str, choices=['h', 'p'],
                        help='space where steering is implemented')

    parser.add_argument('--train_dir', default='complete_dataset', type=str, help='train set directory')
    parser.add_argument('--prompt', default='a blue circle behind a red triangle', type=str,
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
