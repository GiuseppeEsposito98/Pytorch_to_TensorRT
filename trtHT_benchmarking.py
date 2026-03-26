import os
import argparse
from tqdm import tqdm
from tensorrtConversion.common import *
import random


def main():

	ap = argparse.ArgumentParser(description="Benchmarking HT configuration performance")
	ap.add_argument("--root", default = "./ConvertedNNs", help="Root model folder")
	ap.add_argument("--map", help="Target map")
	ap.add_argument("--runs", default=100, help="Number of experimental runs")
	ap.add_argument("--ht", default=None, help="Hardening techniques under test")
	ap.add_argument("--samples", default=1000, help="Hardening techniques under test")
	ap.add_argument("--eval_mode", default=1000, help="Evaluation mode")
	args = ap.parse_args()

	obs_npy = None
	vec_npy = None

	method_name=''
	device='cuda'

	sample_size = int(args.samples)
	n_runs = int(args.runs)
	print(f'sample_size: {sample_size}')
	print(f'n_runs: {n_runs}')


	export_mode = 'base'
	mapUT = f'{args.map}'
	if not args.ht:
		HTs = ['base', 'FP-TMR', 'RP-TMR', 'Ranger', 'Model1', 'Model2', 'Model3', 'Model4', 'Selective TMR', 'Prediction FP-TMR', 'Prediction RP-TMR']
	else:
		HTs = [f'{args.ht}']
	for run_idx in tqdm(range(n_runs), desc = 'Run'):
		random.shuffle(HTs)
		for export_mode in tqdm(HTs, desc='HT configuration'):
			root_modules_path = f'{args.root}/{mapUT}/HT/{export_mode}'

			for file in [path for path in os.listdir(root_modules_path) if path.endswith('.plan')]:

				root_module_file_path = os.path.join(root_modules_path, f'{file}')

				# Qui inizializzi trt
				qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream = setup(root_module_file_path)

				qnet_json = benchmark(qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream, n_runs, sample_size)

				file_path = os.path.join(root_modules_path, f"{file.split('.')[0]}_{run_idx}.json")
				print(file_path)

				save_stats(qnet_json, file_path)
			


if __name__ == '__main__':
	main() 
