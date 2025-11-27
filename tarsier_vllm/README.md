# Activity Description Generation

Generate descriptions for images and videos using Tarsier2 vision-language model with vLLM for high-performance inference.

## Quick Start


### Requirements

- Python >= 3.11
- CUDA 12.6+
- GPU with at least 16GB VRAM (for 7B model)

## 📖 Usage

### 1. Image Description Generation

Generate descriptions for single or multiple images.

**Input Format:**
- Single image path or list of image paths
- Images automatically converted to RGB
- Supports common formats: JPG, PNG, WebP, etc.

**Example:**

```python
from tarsier_vllm.gen_im import run_image_inference

MODEL = "omni-research/Tarsier2-Recap-7b"
IMAGE_PATH = "/path/to/image.jpg"
QUESTION = "Describe what you see in this image."

result = run_image_inference(MODEL, IMAGE_PATH, QUESTION)
```

**Batch Processing:**

```python
from tarsier_vllm.gen_im import run_batch_image_inference

image_configs = [
    {'image_path': 'image1.jpg', 'question': 'What is in this image?'},
    {'image_path': 'image2.jpg', 'question': 'Describe the scene.'},
]

results = run_batch_image_inference(MODEL, image_configs, batch_size=4)
```

### 2. Video Description Generation

Generate descriptions for videos with automatic chunking for long videos.

**Input Format:**
- Video path (MP4, AVI, MOV, etc.)
- Automatic frame extraction at minimum 2 fps
- Long videos automatically split into chunks

**Features:**
- ✅ Minimum 2 frames per second extraction
- ✅ Automatic video splitting for long videos
- ✅ Batch processing multiple videos
- ✅ Result aggregation from chunks

**Example - Single Video:**

```python
from tarsier_vllm.gen_video_enhance import run_video_inference

MODEL = "omni-research/Tarsier2-Recap-7b"
VIDEO_PATH = "/path/to/video.mp4"
QUESTION = "Describe the video content in detail."

result = run_video_inference(
    model_name=MODEL,
    video_path=VIDEO_PATH,
    question=QUESTION,
    max_frames=128,  # Max frames per chunk
    min_fps=2.0      # Minimum 2 frames/second
)

print(f"Video: {result['video_path']}")
print(f"Chunks: {result['num_chunks']}")
print(f"Description:\n{result['combined_text']}")
```

**Example - Batch Video Processing:**

```python
from tarsier_vllm.gen_video_enhance import run_batch_video_inference

video_configs = [
    {
        'video_path': '/path/to/video1.mp4',
        'question': 'Describe this video in detail.'
    },
    {
        'video_path': '/path/to/video2.mp4',
        'question': 'What activities are shown in this video?'
    },
    {
        'video_path': '/path/to/video3.mp4',
        'question': 'Summarize the main events.'
    },
]

batch_results = run_batch_video_inference(
    model_name=MODEL,
    video_configs=video_configs,
    batch_size=4,      # Process 4 videos/chunks in parallel
    max_frames=128,
    min_fps=2.0
)

# Access results
for result in batch_results:
    print(f"Video: {result['video_path']}")
    print(f"Chunks: {result['num_chunks']}")
    print(f"Answer: {result['combined_text']}\n")
```

**How Video Chunking Works:**

For a video with duration `D` seconds:
- Maximum chunk duration = `max_frames / min_fps` (e.g., 128 / 2 = 64 seconds)
- If `D <= 64s`: Single chunk, no splitting
- If `D > 64s`: Split into multiple chunks

Example:
- 180-second video with `max_frames=128`, `min_fps=2.0`
- Chunk duration = 64s
- Number of chunks = ceil(180 / 64) = 3 chunks
- Chunk 1: 0-64s, Chunk 2: 64-128s, Chunk 3: 128-180s

## 📊 Performance Tips

### Memory Optimization

1. **Reduce frame resolution**: `resize_to=(384, 216)` or lower
2. **Adjust max_frames**: Lower value = less memory per chunk
3. **Decrease batch_size**: Process fewer videos simultaneously
4. **Lower gpu_memory_utilization**: Set to 0.8-0.9 if OOM errors occur

### Speed Optimization

1. **Increase batch_size**: Process more items in parallel (if memory allows)
2. **Enable prefix caching**: Reuse KV cache for repeated prompts
3. **Enable chunked prefill**: Better throughput for long sequences
4. **Use FP8 quantization**: `kv_cache_dtype="fp8"` (experimental)


## 🛠️ Troubleshooting

### Out of Memory (OOM)

1. Reduce `max_frames`: Try 64 or 32
2. Lower `resize_to`: Use (384, 216) or (320, 180)
3. Decrease `batch_size`: Use 1 or 2
4. Lower `gpu_memory_utilization`: Set to 0.8

### Slow Processing

1. Increase `batch_size` if memory allows
2. Enable `enable_prefix_caching=True`
3. Use smaller `max_frames` for faster extraction
4. Consider using quantization

### Poor Quality Results

1. Increase `temperature` for more creative outputs
2. Use more frames: increase `max_frames`
3. Higher resolution: larger `resize_to`
4. Better prompts: be specific in questions

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference engine
- [Tarsier2](https://huggingface.co/omni-research/Tarsier2-Recap-7b) - Vision-language model
