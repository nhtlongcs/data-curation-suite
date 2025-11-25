install:
	uv sync
	uv pip install "vllm>=0.8.5" --torch-backend=auto

	uv run python -c "import vllm; print('vllm version:', vllm.__version__)"
	uv run python -c "import torch; print('torch version:', torch.__version__)"
	uv run python -c "import transformers; print('transformers version:', transformers.__version__)"
	uv run python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"