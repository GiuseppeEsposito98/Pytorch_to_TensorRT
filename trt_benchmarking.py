import os
import argparse
from tqdm import tqdm
from tensorrtConversion.common import *
import random


def main():

	ap = argparse.ArgumentParser(description="Benchmarking NN performance")
	ap.add_argument("--root", default = "./ConvertedNNs", help="Root model folder")
	ap.add_argument("--runs", default=100, help="Number of experimental runs")
	ap.add_argument("--samples", default=1000, help="Hardening techniques under test")
	ap.add_argument("--eval_mode", default='energy', help="Evaluation mode")
	args = ap.parse_args()

	sample_size = int(args.samples)
	n_runs = int(args.runs)
	
	root_modules_path = f'{args.root}'

	if args.eval_mode in ['energy']:
		for root, dirs, files in os.walk(root_modules_path):
			for file in files:
				if file.endswith('.plan'):
					root_module_file_path = os.path.join(root, file)
			
					qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream = setup(root_module_file_path)

					if args.eval_mode == 'energy':
						obs_npy = None
						vec_npy = None

						device='cuda'
						
						for run_idx in tqdm(range(n_runs), desc = 'Runs'):

								qnet_json = benchmark(qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream, n_runs, sample_size)

								file_path = os.path.join(root_modules_path, f"{file.split('.')[0]}_{run_idx}.json")
								
								save_stats(qnet_json, file_path)
	else:
		raise NotImplementedError(f'{args.eval_mode} evaluation mode that you are requesting has not been implemented yes')


if __name__ == '__main__':
	main() 
