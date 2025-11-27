## Data Curation Suite

Collection of ready-to-use, high-performance LLM/VLM workflows for data curation and annotation. Built to streamline multimedia understanding tasks with scalable batching, chunking, and preview generation.

### Overview
- **Modalities:** Images, image sequences (as lists), and videos treated uniformly via frame sampling.
- **Performance:** Batch execution, chunked prefill, prefix caching, and controlled GPU memory utilization.
- **For Curators/Annotators:** Produce consistent descriptions, summaries, and labels at scale with reproducible outputs written per chunk.

### Installation

Simple installation using `uv`, this will handle annoying dependencies (vllm, cuda, pytorch versions, etc).
```bash
## Install dependencies with uv (recommended)
uv sync
```

### Use Case 1: Video Activity Description Generation
Leverage a VLM to generate detailed activity descriptions for long videos by representing them as sampled frame sequences. This repository provides a reference pipeline using Tarsier2 (SOTA)

Check out the [detailed instructions](tarsier_vllm/README.md) for setup and usage.


### Use Case 2: JSON Generation (WIP)

Convert LLM descriptions into structured JSON format for easier integration with downstream applications. This workflow is under development and aims to provide a seamless way to organize and store generated metadata.

### Notes
- This repository focuses on reproducible, scalable generation for curators/annotators; it is not a training codebase.
- Ensure you have rights to process the media you run through the pipeline.
