import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser

from PIL import Image
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: list[int] | None = None
    chat_template: str | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: SamplingParams | None = None


def load_tarsier2(question: str, video_path: str, sample_frames: int = 32) -> ModelRequestData:
    model_name = "omni-research/Tarsier2-Recap-7b"
    max_pixels_per_sample = 128 * 384 * 384
    # max_pixels_per_sample // num_images < image_processing_config['max_pixels']:
    #     image_processing_config['max_pixels'] = self.max_pixels_per_sample // num_images
    #     image_processing_config['min_pixels'] = min(image_processing_config['min_pixels'], image_processing_config['max_pixels'])

    engine_args = EngineArgs(
        model='/home/nhtlong/activity-description-generation/vllm/tarsier2_recap_patched_temp',
        trust_remote_code=True,
        max_model_len=32768,
        # max_model_len=8192,
        limit_mm_per_prompt={"image": sample_frames},
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        mm_processor_kwargs={"max_pixels": 128 * 384 * 384, "nframes": 32},
    )

    def load_video_to_image_seq(video_path: str) -> list[Image.Image]:
        import cv2

        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        images = []
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            images.append(pil_image)
            success, image = vidcap.read()
        vidcap.release()
        return images
    image_data = load_video_to_image_seq('/home/nhtlong/activity-description-generation/assets/BigBuckBunny.mp4')
    print("Loaded video frames:", len(image_data))
    # sampling frames from the video
    image_data = image_data[:: len(image_data) // sample_frames][:sample_frames]

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * sample_frames}"
        f"<|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )

def run_generate(
    question: str,
    seed: int | None = 123,
    tensor_parallel_size: int | None = 1,
):
    req_data = load_tarsier2(
        question=question,
        video_path='/home/nhtlong/activity-description-generation/assets/BigBuckBunny.mp4',
        sample_frames=96
    )

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    if tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = tensor_parallel_size
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
    )

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {"image": req_data.image_data},
        },
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


result = run_generate(question="Describe the video in detail.")
print(result)


# def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
#     model_name = "omni-research/Tarsier2-Recap-7b"

#     engine_args = EngineArgs(
#         model='/home/nhtlong/activity-description-generation/vllm/tarsier2_recap_patched_temp',
#         max_model_len=4096,
#         hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
#         limit_mm_per_prompt={modality: 1},
#     )

#     if modality == "image":
#         placeholder = "<|image_pad|>"
#     elif modality == "video":
#         placeholder = "<|video_pad|>"
        
#     image_inputs, video_inputs = process_vision_info(messages)

#     prompts = [
#         (
#             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
#             f"{question}<|im_end|>\n"
#             "<|im_start|>assistant\n"
#         )
#         for question in questions
#     ]

#     return ModelRequestData(
#         engine_args=engine_args,
#         prompts=prompts,
#     )
# run_tarsier2