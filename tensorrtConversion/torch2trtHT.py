import os
import argparse
import torch
import onnx
import pickle
from pathlib import Path
from copy import deepcopy
import tensorrt as trt
from PTmodels.sb3net import SB3Net, TMRModule, FullTMRModule, pick_layer_by_idx_name
import numpy as np
from collections import OrderedDict
from tensorrtConversion.ConverterUtils import build_int8_engine_from_onnx, build_trt_engine
from tensorrtConversion.Calibration.calibrator import EntropyCalibrator
import json
import sys
from tensorrtConversion.torch2trt import iter_shape_leaves, make_inputs, export_to_onnx, iter_leaves, pick_layer_by_idx
from tqdm import tqdm
from tensorrtConversion.common import *

def main():

    ap = argparse.ArgumentParser(description="Hardening Techniques configurations converter from Pytortch to ONNX to TensorRT")
    ap.add_argument("--map", default = "blocks", help="Target map")
    ap.add_argument("--ht", default = "base", help="Target HT configuration")
    args = ap.parse_args()
    
    no_dynamic = True
    workspace_gb=None
    min_shape = None
    opt_shape = None
    max_shape = None
    
    ht = args.ht
    mapUT = args.map
    input_shapes = list()
    if mapUT == 'NH':
        input_shapes.append((3, 3,144,256))
        input_shapes.append((3,12))
    elif mapUT == 'blocks':
        input_shapes.append((1, 4,36,64))
        input_shapes.append((1,12))
    # default paths
    root_save_path = './'
    quant_onnx_path = 'NN.onnx'
    no_quant_onnx_path = 'last.onnx'
    quant_plan_path = 'NN.plan'
    no_quant_plan_path = 'last.plan'
    

    pickle_path = f'./PTmodels/{mapUT}/sb3net.p'
    shapes_path = f'./PTmodels/{mapUT}/embeddings_shape.json'

    with open(shapes_path, 'r') as f:
        shapes_dict = json.load(f)

    with open(pickle_path, 'rb') as f:
        model_arch = pickle.load(f)

    model = SB3Net(model_arch.cnn_extractor, model_arch.linear_extractor, model_arch.vec_extractor, model_arch.q_net)
    items = list(model._modules.items())[:-1]

    if not args.ht:
        HTs = ['base', 'FP-TMR', 'RP-TMR', 'Ranger', 'Model1', 'Model2', 'Model3', 'Model4', 'SelectiveTMR', 'PredictionFP-TMR', 'PredictionRP-TMR']
    else:
        HTs = [f'{args.ht}']
    
    for ht in tqdm(HTs, desc='HT configuration'):
        if mapUT == 'NH':
            last_layer, lyr_name = pick_layer_by_idx(model, 16)
            last_input_shape = shapes_dict["16"]
            cut_model = deepcopy(model)
            cut_model.q_net=torch.nn.Sequential(*list(cut_model.q_net.children())[:-1])
            if ht == 'Model1':
                for idx in [0,2,5]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [1,3,4,6]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Model2':
                for idx in [0,1,5,6]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [2,3]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Model3':
                for idx in [2,5]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [0,1,3]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Ranger':
                for idx in [0,1,2,3,4,5,6]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing=ht)
            elif ht == 'SelectiveTMR':
                for idx in [1]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'FP-TMR':
                for idx in [0,1,2,3,4,5,6]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'RP-TMR':
                for idx in [0,1,2,3,4,5,6]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'PredictionFP-TMR':
                cut_model = FullTMRModule(cut_model)
            elif ht == 'PredictionRP-TMR':
                cut_model = FullTMRModule(cut_model)

        elif mapUT == 'blocks':
            last_layer, lyr_name = pick_layer_by_idx(model, 14)
            last_input_shape = shapes_dict["14"]
            cut_model = deepcopy(model)
            cut_model.q_net = torch.nn.Sequential(*list(cut_model.q_net.children())[:-2])
            if ht == 'Model1':
                for idx in [3,4]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [0,1,2]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Model2':
                for idx in [0,2,3,4]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [1]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Model3':
                for idx in [1,3]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
                for idx in [0,4]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='Ranger')
            elif ht == 'Ranger':
                for idx in [0,1,2,3,4]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing=ht)
            elif ht == 'SelectiveTMR':
                for idx in [1]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'FP-TMR':
                for idx in [0,1,2,3,4,5]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'RP-TMR':
                for idx in [0,1,2,3,4,5]:
                    _wrap_layer_by_index_inplace(cut_model, lyr_idx = idx, replacing='FP-TMR')
            elif ht == 'PredictionFP-TMR':
                cut_model = FullTMRModule(cut_model)
            elif ht == 'PredictionRP-TMR':
                cut_model = FullTMRModule(cut_model)

        
        if 'Ranger' != ht:
            last_layer = TMRModule(last_layer)

        
        # print(cut_model)
        # sys.exit()

        root_save_path = f'ConvertedNNs/{mapUT}/HT/{ht}'
        root_quant_onnx_path = os.path.join(root_save_path, quant_onnx_path)
        root_no_quant_onnx_path = os.path.join(root_save_path, no_quant_onnx_path)

        Path(root_save_path).mkdir(parents=True, exist_ok=True)

        root_quant_plan_path = os.path.join(root_save_path, quant_plan_path)
        root_no_quant_plan_path = os.path.join(root_save_path, no_quant_plan_path)

        ######## TRTExec of the whole NN\{last layer} ################à
        
        export_to_onnx(
            model=cut_model,
            onnx_path=root_quant_onnx_path,
            input_shapes = input_shapes,
            dynamic=no_dynamic
        )
        print(f"[OK] ONNX salvato: {quant_onnx_path}")

        if 'FP-TMR' in ht:
            try:
                calibration_cache = os.path.join(root_save_path, 'calibration.cache')
                calib = EntropyCalibrator(training_data=None, cache_file=calibration_cache, inputs_shape=input_shapes)
                build_int8_engine_from_onnx(
                    onnx_path=root_quant_onnx_path,
                    plan_path=root_quant_plan_path,
                    calibrator=calib
                    )
            except RuntimeError as e:
                msg = f"Returned error: {e}"
                with open(os.path.join(root_save_path, 'log.txt'), 'w') as f:
                    f.write(msg)
        else:
            try:
                build_trt_engine(
                    onnx_path=root_quant_onnx_path,
                    plan_path=root_quant_plan_path
                    )
            except RuntimeError as e:
                msg = f"Returned error: {e}"
                with open(os.path.join(root_save_path, 'log.txt'), 'w') as f:
                    f.write(msg)

        
        ######## TRTExec of the last layer ################à

        

        export_to_onnx(
            model=last_layer,
            onnx_path=root_no_quant_plan_path,
            input_shapes = last_input_shape,
            dynamic=no_dynamic
        )
        print(f"[OK] ONNX salvato: {no_quant_plan_path}")

        export_to_onnx(
                model=last_layer,
                onnx_path=root_no_quant_onnx_path,
                input_shapes = last_input_shape,
                dynamic=no_dynamic
            )
        print(f"[OK] ONNX salvato: {root_no_quant_onnx_path}")
        build_trt_engine(
            onnx_path=root_no_quant_onnx_path,
            plan_path=root_no_quant_plan_path,
        )

if __name__ == "__main__":
    main()