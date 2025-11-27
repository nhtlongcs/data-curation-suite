from dataclasses import asdict
from PIL import Image
from typing import List, Dict, Tuple, Optional
import math
import os
import os.path as osp
import argparse
from pathlib import Path
from vllm import LLM, EngineArgs, SamplingParams

def make_engine_args(model_name: str, max_num_seqs: int = 4):
    return EngineArgs(
        model=model_name,
        # dtype="half", # dangerous: may cause instability 
        # kv_cache_dtype="fp8", # dangerous: may cause instability 
        # enable_chunked_prefill=True,
        # enable_prefix_caching=True, 

        disable_log_stats=True,  # Monitor performance
        # enforce_eager=True,
        max_model_len=32768,
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        # Giới hạn memory cho multimodal mỗi prompt (tùy chỉnh theo tài nguyên)
        limit_mm_per_prompt={"video": 1},
        max_num_seqs=max_num_seqs,
        mm_processor_kwargs={"max_pixels": 128 * 384 * 384},
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=16384,
    )

def _load_and_resize_image(path: str, resize_to: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resize_to:
        img = img.resize(resize_to, Image.LANCZOS)
    return img

def extract_images_from_list(list_path: str,
                             max_frames: int = 128,
                             sample_rate: int = 1,
                             resize_to: Tuple[int, int] = (480, 270),
                             start_index: Optional[int] = None,
                             end_index: Optional[int] = None,
                             save_preview: bool = True,
                             preview_dir: Optional[str] = './preview') -> List[Image.Image]:
    """
    Load images from a text file and sample like video frames.

    Args:
        list_path: Path to a text file containing image paths (one per line)
        max_frames: Maximum frames to return after sampling
        sample_rate: Keep every N-th image (>=1)
        resize_to: Resize dimensions (width, height)
        start_index: Optional start index (inclusive) within the image list
        end_index: Optional end index (exclusive) within the image list
    """
    if sample_rate < 1:
        sample_rate = 1

    with open(list_path, "r") as f:
        all_paths = [line.strip() for line in f if line.strip()]

    # apply slicing window if provided
    si = start_index or 0
    ei = end_index if end_index is not None else len(all_paths)
    si = max(0, si)
    ei = min(ei, len(all_paths))
    window_paths = all_paths[si:ei]

    # downsample by sample_rate (every N-th image)
    sampled_paths = window_paths[::sample_rate] if sample_rate > 1 else window_paths

    # if still too many, uniformly sample down to max_frames
    if len(sampled_paths) > max_frames:
        indices = [int(i * len(sampled_paths) / max_frames) for i in range(max_frames)]
        sampled_paths = [sampled_paths[i] for i in indices]

    frames: List[Image.Image] = []
    for p in sampled_paths:
        if not osp.exists(p):
            continue
        try:
            frames.append(_load_and_resize_image(p, resize_to))
        except Exception:
            # skip unreadable images
            continue

    if save_preview and frames:
        preview_width = 4
        # Determine preview filename, include start/end if provided (handle None)
        start_str = f"{si}" if start_index is not None else "start"
        end_str = f"{ei}" if end_index is not None else "end"
        preview_filename = f"{osp.splitext(osp.basename(list_path))[0]}_preview_{start_str}_{end_str}.jpg"
        if preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
            preview_filename = osp.join(preview_dir, preview_filename)
        preview_height = (len(frames) + preview_width - 1) // preview_width
        preview_image = Image.new('RGB', (resize_to[0] * preview_width, resize_to[1] * preview_height))
        for idx, frame in enumerate(frames):
            x = (idx % preview_width) * resize_to[0]
            y = (idx // preview_width) * resize_to[1]
            preview_image.paste(frame, (x, y))
        preview_image.save(preview_filename)

    return frames

def split_image_list_chunks(list_path: str,
                            max_frames: int = 128,
                            sample_rate: int = 1) -> List[Dict]:
    """
    Split an image list into chunks similar to video splitting.

    Strategy: ensure each chunk produces <= max_frames after applying sample_rate.
    That means chunk raw window size is max_frames * sample_rate.
    """
    with open(list_path, "r") as f:
        total_images = len([line for line in f if line.strip()])

    if total_images == 0:
        return [{'start_index': 0, 'end_index': 0, 'chunk_id': 0}]

    raw_chunk_size = max(1, max_frames * max(1, sample_rate))
    num_chunks = math.ceil(total_images / raw_chunk_size)
    chunks = []
    for i in range(num_chunks):
        start_index = i * raw_chunk_size
        end_index = min((i + 1) * raw_chunk_size, total_images)
        chunks.append({
            'start_index': start_index,
            'end_index': end_index,
            'chunk_id': i
        })
    return chunks


def run_batch_image_list_inference(model_name: str,
                                   list_configs: List[Dict],
                                   batch_size: int = 4,
                                   max_frames: int = 128,
                                   sample_rate: int = 1,
                                   resize_to: Tuple[int, int] = (480, 270),
                                   save_dir: Optional[str] = './tarsier_outputs') -> List[Dict]:
    """
    Run inference on multiple image-list files in batches.

    list_configs: [{'list_path': str, 'question': str}]
    """
    engine_args = make_engine_args(model_name, max_num_seqs=batch_size)
    engine_args_dict = asdict(engine_args) | {"seed": 1234, "mm_processor_cache_gb": 6}

    llm = LLM(**engine_args_dict)

    all_tasks = []
    for config in list_configs:
        list_path = config['list_path']
        question = config['question']
        chunks = split_image_list_chunks(list_path, max_frames=max_frames, sample_rate=sample_rate)
        print(f"Image list: {list_path} -> {len(chunks)} chunk(s)", flush=True)
        for chunk in chunks:
            # If output already exists, skip adding this chunk to the task list
            if save_dir:
                base = osp.splitext(osp.basename(list_path))[0]
                out_name = f"{base}_chunk_{chunk['chunk_id']}_{chunk['start_index']}_{chunk['end_index']}_line.txt"
                out_path = osp.join(save_dir, out_name)
                if osp.exists(out_path):
                    print(f"  - Skip existing output: {out_path}", flush=True)
                    continue

            all_tasks.append({
                'list_path': list_path,
                'question': question,
                'chunk': chunk,
                'original_index': len(all_tasks)
            })

    results_by_list = {config['list_path']: [] for config in list_configs}

    for batch_start in range(0, len(all_tasks), batch_size):
        batch_tasks = all_tasks[batch_start:batch_start + batch_size]
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{math.ceil(len(all_tasks)/batch_size)}", flush=True)

        batch_inputs = []
        for task in batch_tasks:

            frames = extract_images_from_list(
                task['list_path'],
                max_frames=max_frames,
                sample_rate=sample_rate,
                resize_to=resize_to,
                start_index=task['chunk']['start_index'],
                end_index=task['chunk']['end_index']
            )

            chunk_id = task['chunk']['chunk_id']
            chunk_info = f" (Chunk {chunk_id + 1}: idx {task['chunk']['start_index']} - {task['chunk']['end_index']})"
            print(f"  - {task['list_path']}{chunk_info}: {len(frames)} frames", flush=True)

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

        # If all tasks in this batch were skipped due to existing outputs
        if not batch_inputs:
            continue

        sampling_params = SamplingParams(temperature=0.2, max_tokens=256)
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for task, output in zip(batch_tasks, outputs):
            text = output.outputs[0].text.strip()
            # Store in-memory result
            results_by_list[task['list_path']].append({
                'chunk_id': task['chunk']['chunk_id'],
                'start_index': task['chunk']['start_index'],
                'end_index': task['chunk']['end_index'],
                'text': text
            })

            # Also save each chunk's text to a file: listname_chunk_idx_start_end_line.txt
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True) if save_dir else None
                    list_path = task['list_path']
                    base = osp.splitext(osp.basename(list_path))[0]
                    chunk_id = task['chunk']['chunk_id']
                    start_idx = task['chunk']['start_index']
                    end_idx = task['chunk']['end_index']
                    out_name = f"{base}_chunk_{chunk_id}_{start_idx}_{end_idx}_line.txt"
                    out_path = osp.join(save_dir, out_name)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text + "\n")
                except Exception as e:
                    print(f"Warning: Failed to write chunk text to file: {e}")
                    pass

    final_results = []
    for config in list_configs:
        list_path = config['list_path']
        list_results = sorted(results_by_list[list_path], key=lambda x: x['chunk_id'])
        if len(list_results) == 1:
            combined_text = list_results[0]['text']
        else:
            combined_text = "\n\n".join([
                f"[Part {r['chunk_id'] + 1}: idx {r['start_index']} - {r['end_index']}]\n{r['text']}"
                for r in list_results
            ])

        final_results.append({
            'list_path': list_path,
            'question': config['question'],
            'num_chunks': len(list_results),
            'chunks': list_results,
            'combined_text': combined_text
        })

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tarsier VLM on image lists")
    parser.add_argument("--model", type=str, default="omni-research/Tarsier2-Recap-7b")
    parser.add_argument("--question", type=str, default="Describe the content in detail.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=(480, 270))
    parser.add_argument("--dry-run", action="store_true", help="Do preprocessing only; do not load model")

    # image-list mode args
    parser.add_argument("--sample-rate", type=int, default=3, help="Keep every N-th image for image lists")
    args = parser.parse_args()
    image_lists = list(Path('deakin_txt_lists').glob('ID*.txt'))

    list_configs = [{
        'list_path': p,
        'question': args.question
    } for p in image_lists]
    # } for p in ['deakin_txt_lists/ID101_20220310.txt']]
    print(f"Found {len(list_configs)} image list(s) for processing.")
    if args.dry_run:
        for cfg in list_configs:
            chunks = split_image_list_chunks(cfg['list_path'], max_frames=args.max_frames, sample_rate=args.sample_rate)
            print(f"Image list: {cfg['list_path']} -> {len(chunks)} chunk(s)")
            for ch in chunks:
                frames = extract_images_from_list(
                    cfg['list_path'],
                    max_frames=args.max_frames,
                    sample_rate=args.sample_rate,
                    resize_to=tuple(args.resize),
                    start_index=ch['start_index'],
                    end_index=ch['end_index'],
                    save_preview=True,
                    preview_dir='preview'
                )
                print(f"  - Chunk {ch['chunk_id']+1}: idx {ch['start_index']} - {ch['end_index']} => {len(frames)} frames")
        print("Dry run complete.")
    else:
        results = run_batch_image_list_inference(
            model_name=args.model,
            list_configs=list_configs,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            sample_rate=args.sample_rate,
            resize_to=tuple(args.resize),
        )
        print("\n=== Image-List Batch Results ===")
        for i, result in enumerate(results):
            print(f"\n--- List {i+1} ---")
            print(f"Path: {result['list_path']}")
            print(f"Question: {result['question']}")
            print(f"Chunks: {result['num_chunks']}")
            print(f"\nAnswer:\n{result['combined_text']}")
            print("-" * 80)