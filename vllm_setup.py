
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

# Register with transformers' AutoModelForCausalLM

# def register():
#     from vllm import ModelRegistry
#     from transformers import AutoModelForCausalLM
    
#     # Dynamically import Qwen2VLForCausalLM and Qwen2VLConfig from local models
#     qwen2_vl_path = os.path.join(os.path.dirname(__file__), 'models', 'modeling_qwen2_vl_fast.py')
#     spec = importlib.util.spec_from_file_location("modeling_qwen2_vl_fast", qwen2_vl_path)
#     qwen2_vl_mod = importlib.util.module_from_spec(spec)
#     sys.modules["modeling_qwen2_vl_fast"] = qwen2_vl_mod
#     spec.loader.exec_module(qwen2_vl_mod)

#     Qwen2VLForCausalLM = qwen2_vl_mod.Qwen2VLForCausalLM
#     Qwen2VLConfig = qwen2_vl_mod.Qwen2VLConfig

#     AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForCausalLM)
    
#     tarsier_model_path = os.path.join(os.path.dirname(__file__), 'models', 'modeling_tarsier.py')
#     spec_tarsier = importlib.util.spec_from_file_location("modeling_tarsier.py", tarsier_model_path)
#     tarsier_mod = importlib.util.module_from_spec(spec_tarsier)
#     sys.modules["tarsier_model"] = tarsier_mod
#     spec_tarsier.loader.exec_module(tarsier_mod)

#     TarsierForConditionalGeneration = tarsier_mod.TarsierForConditionalGeneration
#     TarsierConfig = tarsier_mod.LlavaConfig

#     AutoModelForCausalLM.register(config_class=TarsierConfig, model_class=TarsierForConditionalGeneration)
#     ModelRegistry.register_model("TarsierForConditionalGeneration", TarsierForConditionalGeneration)

# register()

# --- vLLM Plugin Registration for Tarsier ---

try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    vllm = None
    VLLM_AVAILABLE = False
from importlib.util import spec_from_file_location, module_from_spec


def register_tarsier_plugin():
    """
    Register the Tarsier model as a vLLM plugin, following vLLM's plugin system guidelines.
    This allows vLLM to discover and use the Tarsier model via its ModelRegistry.
    """
    if not VLLM_AVAILABLE:
        print("[WARN] vLLM is not installed. Skipping Tarsier plugin registration.")
        return
    try:
        from vllm import ModelRegistry
    except ImportError:
        print("[WARN] vLLM ModelRegistry not found. Skipping Tarsier plugin registration.")
        return

    # Dynamically import Tarsier model and config
    tarsier_path = os.path.join(os.path.dirname(__file__), 'minimal_tarsier', 'models', 'modeling_tarsier.py')
    spec = spec_from_file_location("modeling_tarsier", tarsier_path)
    tarsier_mod = module_from_spec(spec)
    sys.modules["modeling_tarsier"] = tarsier_mod
    spec.loader.exec_module(tarsier_mod)

    # These class names must match your implementation
    TarsierForConditionalGeneration = getattr(tarsier_mod, "TarsierForConditionalGeneration", None)
    TarsierConfig = getattr(tarsier_mod, "LlavaConfig", None)
    if TarsierForConditionalGeneration and TarsierConfig:
        ModelRegistry.register_model("TarsierForConditionalGeneration", TarsierForConditionalGeneration)
        # Optionally, register config if vLLM supports it
        # ModelRegistry.register_config("TarsierConfig", TarsierConfig)
        print("[INFO] Registered TarsierForConditionalGeneration with vLLM ModelRegistry.")
    else:
        print("[WARN] Tarsier model or config class not found. Plugin registration skipped.")

# Register the plugin at import time
register_tarsier_plugin()

# from .dataloader import get_dataloader
# from .config import ExperimentConfig

# TODO: add warmup

from typing import Any

def get_vllm_model() -> Any:
    # Patch: Ensure the config.json for the model uses the correct architecture name
    # vllm expects 'Qwen2VLForCausalLM' for Tarsier2-Recap-7b vision-language model
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Cannot create vLLM model.")
    model_id = "omni-research/Tarsier2-Recap-7b"
    # model_dir = snapshot_download(model_id, allow_patterns=["config.json"])

    llm = vllm.LLM(
        model=model_id,
        gpu_memory_utilization=0.9,
        dtype="half",
        trust_remote_code=True,
        disable_log_stats=True,
        swap_space=4,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=8192,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
    )
    return llm



# def vllm_call(model, questions: List[Dict], config: ExperimentConfig) -> List[Dict]:

#     model_name = config.MODEL_LIST[0]['MODEL_NAME']
#     if 'Qwen' in model_name:
#         no_think = True
#     else:
#         no_think = False

#     # For sorting, keep track of original order
#     original_order = {q['questionID']: idx for idx, q in enumerate(questions)}

#     # Group questions by subject, create a Dataloader for each subject
#     questions_by_subject = {}
#     for q in questions:
#         if 'topic' in q:
#             subject = q['topic']
#         else:
#             subject = q.get('subject', 'general')

#         if subject not in questions_by_subject:
#             questions_by_subject[subject] = []
#         questions_by_subject[subject].append(q)

#     tokenizer = model.get_tokenizer()
#     dataloaders = {
#         subject: get_dataloader(
#             questions_by_subject[subject], 
#             tokenizer, config, subject=subject,
#             use_python_code=config.Routers.Math.USE_PYTHON_CODE if subject in ['math', 'algebra'] else False,
#             no_think=no_think
#         )
#         for subject in questions_by_subject
#     }

#     results = []
#     remaining_questions = []

#     for subject, dataloader in dataloaders.items():
#         for batch in dataloader:
#             batch_questions = batch['prompt_template']
#             batch_questionIDs = batch['questionID']

#             if isinstance(batch_questionIDs, torch.Tensor):
#                 batch_questionIDs = batch_questionIDs.tolist()

#             # Run inference on the batch
#             responses = run_batch(
#                 model=model,  # Assuming single model for simplicity
#                 questions=batch_questions,
#                 router_config=ExperimentConfig.get_topic_router_config(subject)
#             )
#             # Collect results
#             for qid, response in zip(batch_questionIDs, responses):
#                 answer = response['answer']
#                 if answer is None or answer.strip() == "":
#                     remaining_questions.append(qid)
#                     continue
#                 res = {
#                     'questionID': qid,
#                     'answer': answer
#                 }
#                 if 'python_code' in response:
#                     res['python_code'] = response['python_code']
#                 if 'code_exec_error' in response:
#                     res['code_exec_error'] = response['code_exec_error']
#                 results.append(res)

#     # Handle remaining questions with code execution if needed
#     print("Checking for remaining questions needing code execution...Number of remaining questions:", len(remaining_questions))
#     if len(remaining_questions) > 0:
#         remaining_dataloader = get_dataloader(
#             [q for q in questions if q['questionID'] in remaining_questions],
#             tokenizer, config, subject='math', use_python_code=False,
#             no_think=no_think
#         )

#         for batch in remaining_dataloader:
#             batch_questions = batch['prompt_template']
#             batch_questionIDs = batch['questionID']
#             if isinstance(batch_questionIDs, torch.Tensor):
#                 batch_questionIDs = batch_questionIDs.tolist()
#             # Run inference on the batch
#             responses = run_batch(
#                 model=model,  # Assuming single model for simplicity
#                 questions=batch_questions,
#                 router_config=ExperimentConfig.get_topic_router_config(subject)
#             )
       
#             # Collect results
#             for qid, response in zip(batch_questionIDs, responses):
#                 answer = response['answer']

#                 res = {
#                     'questionID': qid,
#                     'answer': answer
#                 }
#                 if 'python_code' in response:
#                     res['python_code'] = response['python_code']
#                 if 'code_exec_error' in response:
#                     res['code_exec_error'] = response['code_exec_error']
#                 results.append(res)

#     assert len(results) == len(questions), "Some questions were not answered."

#     if config.SORT_QUESTIONS:
#         # restore original order
#         results = sorted(results, key=lambda x: original_order[x['questionID']])

#     return results



# def run_batch(model, questions: List[str], router_config) -> Dict[str, Any]:
#     # This function can be extended to support vision-language inference as in vllm's vision_language.py example:
#     # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
#     # For now, we keep text-only inference, but you can add image input handling as needed.
#     if getattr(router_config, 'USE_PYTHON_CODE', False):
#         max_new_tokens = router_config.MAX_NEW_TOKENS_PYTHON
#     else:
#         max_new_tokens = router_config.MAX_NEW_TOKENS

#     sampling_params = vllm.SamplingParams(
#         n=1,
#         top_p=0.9,
#         temperature=0,
#         seed=777,
#         skip_special_tokens=True,
#         max_tokens=max_new_tokens,
#     )

#     responses = model.generate(questions, sampling_params=sampling_params, use_tqdm=False)
#     responses = [response.outputs[0].text for response in responses]
#     responses = [
#         i.replace('<end_of_turn>', '').strip().replace('<think>', '').replace('</think>', '').strip()
#         for i in responses
#     ]

#     final_result = []
#     for i, response in enumerate(responses):
#         res = {
#             'answer': response
#         }
#         final_result.append(res)
#     return final_result

if __name__ == '__main__':
    # Simple test for Tarsier2-Recap-7b (text-only)
    model = get_vllm_model()
    # questions = [
    #     {'image_list': [], 'questionID': 1, 'question': 'Describe the scene in the video.'},
    # ]
    # For vision-language inference, refer to:
    # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
    # and https://github.com/bytedance/tarsier
