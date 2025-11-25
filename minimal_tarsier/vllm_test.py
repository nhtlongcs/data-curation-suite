import argparse
import os
from typing import Optional
import json
import tempfile


def _lazy_imports():
	from vllm import LLM, SamplingParams
	from PIL import Image
	return LLM, SamplingParams, Image


def run(
	model: str = "omni-research/Tarsier2-Recap-7b",
	prompt: str = "Describe the image in detail.",
	image_path: Optional[str] = None,
	max_new_tokens: int = 256,
	temperature: float = 0.0,
	top_p: float = 1.0,
	tensor_parallel_size: int = 1,
	dtype: str = "auto",
):
	LLM, SamplingParams, Image = _lazy_imports()

	sampling_params = SamplingParams(
		max_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
	)
	from huggingface_hub import snapshot_download
	local_dir = tempfile.mkdtemp(prefix="tarsier2_recap_patched_", dir=".")
	repo_dir = snapshot_download(
        repo_id=model,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
	cfg_path = os.path.join(repo_dir, "config.json")
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	cfg["architectures"] = ["TarsierForConditionalGeneration"]
	with open(cfg_path, "w", encoding="utf-8") as f:
		json.dump(cfg, f)

	llm = LLM(
        model=repo_dir,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        task="generate",
        limit_mm_per_prompt={"image": 4},
    )

	if image_path:
		assert os.path.exists(image_path), f"image not found: {image_path}"
		image = Image.open(image_path).convert("RGB")
		inputs = [
			{
				"prompt": prompt,
				"multi_modal_data": {"image": [image]},
			}
		]
	else:
		inputs = [prompt]

	outputs = llm.generate(inputs, sampling_params)

	# vLLM returns a list of RequestOutput; each has .outputs (list of candidates)
	first_output = outputs[0]
	text = first_output.outputs[0].text if first_output.outputs else ""
	print(text)


def cli():
	parser = argparse.ArgumentParser(description="Run Tarsier on vLLM")
	parser.add_argument("--model", type=str, default="omni-research/Tarsier2-Recap-7b")
	parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
	parser.add_argument("--image", type=str, default=None, help="Optional image path")
	parser.add_argument("--max-new-tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--tp", dest="tensor_parallel_size", type=int, default=1)
	parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

	args = parser.parse_args()
	run(
		model=args.model,
		prompt=args.prompt,
		image_path=args.image,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		tensor_parallel_size=args.tensor_parallel_size,
		dtype=args.dtype,
	)


if __name__ == "__main__":
	cli()

