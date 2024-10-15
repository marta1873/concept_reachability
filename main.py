import argparse
import torch
import numpy as np
from text_to_image_model import Diffusion
from dataset import ShapesDataset
from concept_directions import find_concept_vector, sample_with_concept_vector
from experiment_1 import run_experiments, config_dict, run_test
from experiment_2 import config_dict_2_1, sample_prompts
import os
from modify_concept_vector import sample_with_modified_vector
#from torchvision.utils import save_image


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    print(args)

    model = Diffusion(args)

    if args.task == 'train':
        train_set = ShapesDataset(db_path=[args.train_dir])
        test_set = ShapesDataset(db_path=['test_set'])

        model.train_diffusion(train_set, test_set)

    elif args.task == 'sample':
        model.load_checkpoint(args.ckpt_sample)
        prompt = args.prompt
        model.sample(100, prompt, return_img=False)
        #save_image(images[0], 'images_dir/image2.png')

    elif args.task == 'concept':
        model.load_checkpoint(args.ckpt_sample)

        prompts_plus = ['a red triangle behind a green square']
        prompts_minus = ['a red triangle behind a green square']

        run_test(model, 0, prompts_plus, prompts_minus, regularisation=None, base_model=None, space='h')

    elif args.task == 'experiment':
        if args.experiment_num == 1:
            model.load_checkpoint(args.ckpt_sample)

            #sample_prompts(model, config_dict)

            for test in range(19):
                run_experiments(model, test, config_dict, args.regularisation)

        elif args.experiment_num == 2:
            #train_set = ShapesDataset(db_path=[args.train_dir])
            #test_set = ShapesDataset(db_path=['test_set'])
            #model.train_diffusion(train_set, test_set)

            model.load_checkpoint(args.ckpt_sample)
            #sample_prompts(model, config_dict_2_1)

            if args.train_dir != 'complete_dataset':
                args.train_dir = 'complete_dataset'
                args.seed = 2
                base_model = Diffusion(args)
                base_model.load_checkpoint(args.ckpt_sample)
            else:
                base_model = None

            for test in range(17):
            #for test in [0, 1, 4, 6, 8, 12, 15]:
                run_experiments(model, test, config_dict_2_1, args.regularisation, base_model, space=args.space)
                #sample_with_modified_vector(model, test, config_dict_2_1, args.seed)


    elif args.task == 'sample_concept':
        model.load_checkpoint(args.ckpt_sample)
        c_vec1 = torch.load(os.path.join(model.local_dir, 'concept_vectors_3.pt'), map_location=model.device)
        c_vec2 = torch.load(os.path.join(model.local_dir, 'concept_vectors_5.pt'), map_location=model.device)
        #c_vec3 = torch.load(os.path.join(model.local_dir, 'concept_vectors_5.pt'), map_location=model.device)
        c_vec = c_vec1 + c_vec2
        prompt = args.prompt
        sample_with_concept_vector(model, prompt=prompt, concept_vector=c_vec, num_samples=100)

    elif args.task == 'test':

        test_set = ShapesDataset(db_path=['test_set'], noisy=False, exclude=args.dots_exclude)

        for seed in [0, 3, 4, 7]:
            print(f'Seed {seed}')
            args.seed = seed
            for ckpt in ['422', '4642', '8862', '13082', '17302', '21522', '25742', '29962', '34182', '38402', 'last']:
                print(f'Checkpoint {ckpt}')
                model.load_checkpoint(ckpt)
                model.test_diffusion(test_set)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='experiment', type=str, choices=['train', 'sample', 'concept', 'sample_concept', 'experiment', 'test'], help='what to do')
    parser.add_argument('--T', default=1000, type=int, help='number of noising steps')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--test', default=0, type=int, help='test index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--max_epoch', default=70, type=int, help='number of training epochs')
    parser.add_argument('--ckpt_sample', default='last', help='checkpoint used for sampling')
    parser.add_argument('--optim', default='adam', choices=['adam', 'sgd', 'rmsprop'], type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--train_dir', default='complete_dataset', type=str, help='train set directory')
    parser.add_argument('--u_net_size', default='normal', choices=['normal', 'small'], type=str, help='u_net size to use')
    parser.add_argument('--scheduler', default='ddpm', type=str, choices=['ddpm', 'ddim', 'invddim'], help='scheduler type')
    parser.add_argument('--prompt', default='', type=str, help='prompt for generation')
    parser.add_argument('--regularisation', default=None, type=float, help='constant for L1 regularisation')
    parser.add_argument('--experiment_num', default=2, type=int, help='experiment number in results')
    parser.add_argument('--space', default='p', type=str, choices=['h', 'p'], help='space where optimisation is done')
    args = parser.parse_args()

    main(args)