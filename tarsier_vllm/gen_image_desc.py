#!/usr/bin/env python3
from dataclasses import asdict
from PIL import Image
import uuid as _uuid

from vllm import LLM, EngineArgs, SamplingParams

def make_engine_args(model_name: str):
    # Thêm hf_overrides để chỉ rõ kiến trúc Tarsier2 đã được đăng ký trong vLLM
    return EngineArgs(
        model=model_name,
        enforce_eager=True,

        max_model_len=4096,
        # Bắt buộc: nếu repo model dùng class tùy chỉnh, set kiến trúc tương ứng
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        # Giới hạn MM memory / prompt (tùy chỉnh theo tài nguyên)
        limit_mm_per_prompt={"image": 1},
        # Nếu cần, bật trust_remote_code (nếu vLLM hỗ trợ tham số này)
        # trust_remote_code=True,
        max_num_seqs=2,
        gpu_memory_utilization=0.8,
    )

def run_image_inference(model_name: str, image_path: str, question: str):
    engine_args = make_engine_args(model_name)
    # Chuyển sang dict để truyền thêm các tham số runtime (seed, mm cache)
    engine_args_dict = asdict(engine_args) | {"seed": 1234, "mm_processor_cache_gb": 4}

    # Khởi tạo LLM
    llm = LLM(**engine_args_dict)

    # Mẫu prompt: dùng placeholder phù hợp với Tarsier2
    # Lưu ý: token placeholder cần khớp với tokenization/model expectation
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Mẫu sampling params
    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

    # Load ảnh (PIL Image) và tạo uuid cho multi-modal data
    img = Image.open(image_path).convert("RGB")
    img_uuid = str(_uuid.uuid4())

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": img},          # key 'image' tương ứng với modality bạn dùng
        "multi_modal_uuids": {"image": img_uuid},
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # outputs là generator/list các response items
    for o in outputs:
        # Kiểm tra cấu trúc output theo phiên bản vLLM của bạn
        text = o.outputs[0].text.strip()
        print("=== Model output ===")
        print(text)
        print("====================")

if __name__ == "__main__":
    MODEL = "omni-research/Tarsier2-Recap-7b"  # hoặc tên model Tarsier2 khác
    IMAGE_PATH = "test.jpg"
    QUESTION = "tell me what's in this image."

    run_image_inference(MODEL, IMAGE_PATH, QUESTION)