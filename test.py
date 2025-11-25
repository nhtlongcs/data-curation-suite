
import os
import json
from huggingface_hub import snapshot_download
import torch
from typing import *
# patch to register Qwen2VL model 
# # https://github.com/QwenLM/Qwen3-VL/issues/43
# from transformers import AutoModelForCausalLM, Qwen2VLConfig, Qwen2VLForConditionalGeneration
# AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

# Patch: Register local Qwen2VLForCausalLM implementation before vLLM loads the model
import sys
import importlib.util

import os
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Register with transformers' AutoModelForCausalLM

def register():
    from transformers import AutoModelForCausalLM
    
    # Dynamically import Qwen2VLForCausalLM and Qwen2VLConfig from local models
    qwen2_vl_path = os.path.join(os.path.dirname(__file__), 'models', 'modeling_qwen2_vl_fast.py')
    spec = importlib.util.spec_from_file_location("modeling_qwen2_vl_fast", qwen2_vl_path)
    qwen2_vl_mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_qwen2_vl_fast"] = qwen2_vl_mod
    spec.loader.exec_module(qwen2_vl_mod)

    Qwen2VLForCausalLM = qwen2_vl_mod.Qwen2VLForCausalLM
    Qwen2VLConfig = qwen2_vl_mod.Qwen2VLConfig

    AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForCausalLM)
    
    tarsier_model_path = os.path.join(os.path.dirname(__file__), 'models', 'modeling_tarsier.py')
    spec_tarsier = importlib.util.spec_from_file_location("modeling_tarsier.py", tarsier_model_path)
    tarsier_mod = importlib.util.module_from_spec(spec_tarsier)
    sys.modules["tarsier_model"] = tarsier_mod
    spec_tarsier.loader.exec_module(tarsier_mod)

    TarsierForConditionalGeneration = tarsier_mod.TarsierForConditionalGeneration
    TarsierConfig = tarsier_mod.LlavaConfig

    AutoModelForCausalLM.register(config_class=TarsierConfig, model_class=TarsierForConditionalGeneration)
    # ModelRegistry.register_model("TarsierForConditionalGeneration", TarsierForConditionalGeneration)

# def load_model_and_processor(model_name_or_path, data_config):
#     print(Color.red(f"Load model and processor from: {model_name_or_path}"), flush=True)
#     if isinstance(data_config, str):
#         data_config = yaml.safe_load(open(data_config, 'r'))
#     processor = init_processor(model_name_or_path, data_config)
#     model_config = LlavaConfig.from_pretrained(
#         model_name_or_path,
#         trust_remote_code=True,
#     )
#     model = TarsierForConditionalGeneration.from_pretrained(
#         model_name_or_path,
#         config=model_config,
#         device_map='auto',
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True
#     )
#     model.eval()
#     return model, processor

# def file_to_base64(img_path):
#     with open(img_path, 'rb') as video_file:
#         video_b64_str = base64.b64encode(video_file.read()).decode()
#     return video_b64_str

# register()
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "omni-research/Tarsier2-Recap-7b"
model_dir = snapshot_download(model_id, allow_patterns=["config.json"])

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")