from dataclasses import asdict
from PIL import Image
import uuid as _uuid
import av
from typing import List, Dict, Tuple
import math
import os.path as osp


from vllm import LLM, EngineArgs, SamplingParams

def make_engine_args(model_name: str, max_num_seqs: int = 4):
    return EngineArgs(
        model=model_name,
        # dtype="half", 
        # kv_cache_dtype="fp8",
        enable_chunked_prefill=True,
        enable_prefix_caching=True, 

        disable_log_stats=True,  # Monitor performance
        enforce_eager=True,
        max_model_len=32768,
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        # Giới hạn memory cho multimodal mỗi prompt (tùy chỉnh theo tài nguyên)
        limit_mm_per_prompt={"video": 1},
        max_num_seqs=max_num_seqs,
        mm_processor_kwargs={"max_pixels": 128 * 384 * 384},
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=16384,
    )

def get_video_info(video_path: str) -> Tuple[float, float, int]:
    """Get video FPS, duration, and total frames"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    total_frames = stream.frames
    duration = float(stream.duration * stream.time_base) if stream.duration else total_frames / fps
    container.close()
    return fps, duration, total_frames

def extract_frames_pyav(video_path: str, max_frames: int = 128, min_fps: float = 2.0, 
                        resize_to: Tuple[int, int] = (480, 270), 
                        start_time: float = None, end_time: float = None, save_preview: bool = True) -> List[Image.Image]:
    """
    Extract frames ensuring at least min_fps frames per second
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        min_fps: Minimum frames per second (default 2.0)
        resize_to: Resize dimensions (width, height)
        start_time: Start time in seconds (for video chunks)
        end_time: End time in seconds (for video chunks)
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    
    # Seek to start time if specified
    if start_time:
        container.seek(int(start_time * av.time_base))
    
    frames = []
    frame_times = []
    
    for frame in container.decode(video=0):
        frame_time = float(frame.time)
        
        # Skip frames before start_time
        if start_time and frame_time < start_time:
            continue
        
        # Stop if we've reached end_time
        if end_time and frame_time > end_time:
            break
            
        img = frame.to_image()  
        if resize_to:
            img = img.resize(resize_to, Image.LANCZOS)
        frames.append(img)
        frame_times.append(frame_time)

    container.close()
    
    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")
    
    # Calculate the duration of extracted segment
    duration = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 1.0
    
    # Calculate required frames for min_fps
    required_frames = max(int(duration * min_fps), 1)
    
    # Use the smaller of max_frames and required_frames
    target_frames = min(max_frames, required_frames)
    
    # Uniform sampling to get target_frames
    if len(frames) > target_frames:
        indices = [int(i * len(frames) / target_frames) for i in range(target_frames)]
        frames = [frames[i] for i in indices]
    
    if save_preview:
        preview_width = 4
        chunk_label = "full" if start_time is None and end_time is None else f"{(start_time or 0):.2f}_{(end_time or 0):.2f}"
        preview_filename = f"{osp.splitext(osp.basename(video_path))[0]}_{chunk_label}_preview.jpg"
        preview_height = (len(frames) + preview_width - 1) // preview_width
        preview_image = Image.new('RGB', (resize_to[0] * preview_width, resize_to[1] * preview_height))
        for idx, frame in enumerate(frames):
            x = (idx % preview_width) * resize_to[0]
            y = (idx // preview_width) * resize_to[1]
            preview_image.paste(frame, (x, y))
        
        preview_image.save(preview_filename)

    return frames

def split_video_chunks(video_path: str, max_frames: int = 128, min_fps: float = 2.0) -> List[Dict]:
    """
    Split video into chunks if it exceeds max_frames at min_fps
    
    Returns:
        List of dicts with 'start_time', 'end_time', 'chunk_id'
    """
    fps, duration, total_frames = get_video_info(video_path)
    
    # Calculate max duration per chunk at min_fps
    max_duration_per_chunk = max_frames / min_fps
    
    # If video fits in one chunk, return single chunk
    if duration <= max_duration_per_chunk:
        return [{'start_time': None, 'end_time': None, 'chunk_id': 0}]
    
    # Split into chunks
    num_chunks = math.ceil(duration / max_duration_per_chunk)
    chunks = []
    
    for i in range(num_chunks):
        start_time = i * max_duration_per_chunk
        end_time = min((i + 1) * max_duration_per_chunk, duration)
        chunks.append({
            'start_time': start_time,
            'end_time': end_time,
            'chunk_id': i
        })
    
    return chunks


def run_batch_video_inference(model_name: str, video_configs: List[Dict], batch_size: int = 4, 
                               max_frames: int = 128, min_fps: float = 2.0, 
                               resize_to: Tuple[int, int] = (480, 270)):
    """
    Run inference on multiple videos in batches
    
    Args:
        model_name: Model name/path
        video_configs: List of dicts with 'video_path' and 'question'
        batch_size: Number of videos to process in parallel
        max_frames: Maximum frames per chunk
        min_fps: Minimum frames per second
        resize_to: Resize dimensions
        
    Returns:
        List of dicts with 'video_path', 'results' (list of chunk results)
    """
    engine_args = make_engine_args(model_name, max_num_seqs=batch_size)
    engine_args_dict = asdict(engine_args) | {"seed": 1234, "mm_processor_cache_gb": 6}

    llm = LLM(**engine_args_dict)
    
    # Prepare all tasks (video + chunk combinations)
    all_tasks = []
    for config in video_configs:
        video_path = config['video_path']
        question = config['question']
        
        # Split video into chunks if needed
        chunks = split_video_chunks(video_path, max_frames=max_frames, min_fps=min_fps)
        
        print(f"Video: {video_path} -> {len(chunks)} chunk(s)", flush=True)
        
        for chunk in chunks:
            all_tasks.append({
                'video_path': video_path,
                'question': question,
                'chunk': chunk,
                'original_index': len(all_tasks)
            })
    
    # Process tasks in batches
    results_by_video = {config['video_path']: [] for config in video_configs}
    
    for batch_start in range(0, len(all_tasks), batch_size):
        batch_tasks = all_tasks[batch_start:batch_start + batch_size]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{math.ceil(len(all_tasks)/batch_size)}", flush=True)
        
        batch_inputs = []
        for task in batch_tasks:
            # Extract frames for this chunk
            frames = extract_frames_pyav(
                task['video_path'],
                max_frames=max_frames,
                min_fps=min_fps,
                resize_to=resize_to,
                start_time=task['chunk']['start_time'],
                end_time=task['chunk']['end_time']
            )
            
            chunk_id = task['chunk']['chunk_id']
            chunk_info = ""
            if task['chunk']['start_time'] is not None:
                chunk_info = f" (Chunk {chunk_id + 1}: {task['chunk']['start_time']:.1f}s - {task['chunk']['end_time']:.1f}s)"
            
            print(f"  - {task['video_path']}{chunk_info}: {len(frames)} frames", flush=True)
            
            # Build prompt
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|>"
                f"{task['question']}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"video": frames},
            })
        
        # Run inference on batch
        sampling_params = SamplingParams(temperature=0.2, max_tokens=256)
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        # Collect results
        for task, output in zip(batch_tasks, outputs):
            text = output.outputs[0].text.strip()
            results_by_video[task['video_path']].append({
                'chunk_id': task['chunk']['chunk_id'],
                'start_time': task['chunk']['start_time'],
                'end_time': task['chunk']['end_time'],
                'text': text
            })
    
    # Organize results by video
    final_results = []
    for config in video_configs:
        video_path = config['video_path']
        video_results = sorted(results_by_video[video_path], key=lambda x: x['chunk_id'])
        
        # Combine chunk results
        if len(video_results) == 1:
            combined_text = video_results[0]['text']
        else:
            combined_text = "\n\n".join([
                f"[Part {r['chunk_id'] + 1}: {r['start_time']:.1f}s - {r['end_time']:.1f}s]\n{r['text']}"
                for r in video_results
            ])
        
        final_results.append({
            'video_path': video_path,
            'question': config['question'],
            'num_chunks': len(video_results),
            'chunks': video_results,
            'combined_text': combined_text
        })
    
    return final_results

def run_single_video_inference(model_name: str, video_path: str, question: str, 
                               max_frames: int = 128, min_fps: float = 2.0):
    """
    Run inference on a single video (wrapper for backward compatibility)
    """
    results = run_batch_video_inference(
        model_name=model_name,
        video_configs=[{'video_path': video_path, 'question': question}],
        batch_size=1,
        max_frames=max_frames,
        min_fps=min_fps
    )
    
    return results[0]

if __name__ == "__main__":
    MODEL = "omni-research/Tarsier2-Recap-7b"
    
    video_configs = [
        {
            'video_path': "/home/nhtlong/activity-description-generation/data/ads1.mp4",
            'question': "Describe the video content in detail."
        },
        {
            'video_path': "/home/nhtlong/activity-description-generation/data/ads2.mp4",
            'question': "Describe the video content in detail."
        },
        {
            'video_path': "/home/nhtlong/activity-description-generation/data/lifelog1.mp4",
            'question': "Describe the video content in detail."
        },
        {
            'video_path': "/home/nhtlong/activity-description-generation/data/lifelog2.mp4",
            'question': "Describe the video content in detail."
        },
    ]
    
    batch_results = run_batch_video_inference(
        model_name=MODEL,
        video_configs=video_configs,
        batch_size=4,  # Process 4 videos/chunks in parallel
        max_frames=128,
        min_fps=2
    )
    
    print("\n=== Batch Results ===")
    for i, result in enumerate(batch_results):
        print(f"\n--- Video {i+1} ---")
        print(f"Path: {result['video_path']}")
        print(f"Question: {result['question']}")
        print(f"Chunks: {result['num_chunks']}")
        print(f"\nAnswer:\n{result['combined_text']}")
        print("-" * 80)

    # run_video_inference(MODEL, VIDEO_PATH, QUESTION)