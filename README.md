# PhaseServe: Efficient Phase-Disaggregated LLM Serving with Phase-Specialized Coordinated Scheduling

PhaseServe improves the performance of phase-disaggregated large language model (LLM) serving through **phase-specialized scheduling**.

It is built on top of [DistServe](https://github.com/LLMServe/DistServe) and extends its scheduling path for the two stages of LLM inference.

In particular, PhaseServe introduces:

- **length-aware scheduling for the context / prefill phase**, exposed through `--context-sched-policy sjf`, to better handle requests with known but highly heterogeneous prompt lengths;
- **attained-service-aware scheduling for the decoding phase**, exposed through `--decoding-sched-policy mlfq`, to better handle requests with unknown and dynamically evolving generation lengths.

The key idea is that the two phases of LLM inference have different scheduling characteristics. Prefill requests have known but heterogeneous costs, while decode requests have unknown and dynamically evolving generation lengths. PhaseServe applies different scheduling policies to the two phases to better match these characteristics.

## Relationship to DistServe

PhaseServe is implemented on top of DistServe. It preserves DistServe’s phase-disaggregated architecture, distributed execution framework, and KV-cache communication / memory-management stack, while extending the scheduler with phase-specialized policies for prefill and decoding.

## Supported Models

Inherited from DistServe:

- GPT-2 (`gpt2`, `gpt2-xl`, ...)
- OPT (`facebook/opt-1.3b`, `facebook/opt-6.7b`, ...)
- LLaMA2 (`meta-llama/Llama-2-7b`, `meta-llama/Llama-2-13b`, ...)

## Build && Install

PhaseServe follows the same build and installation pipeline as DistServe. Please refer to the original [DistServe](https://github.com/LLMServe/DistServe) setup for environment preparation, backend dependencies, and installation.

## Launching

### Launch Ray Cluster

PhaseServe relies on [Ray](https://ray.io) to implement distributed workers. If you do not launch a Ray runtime in advance, it can automatically initiate a cluster consisting of all GPUs on the current node. You may need to start the Ray runtime manually in advance if you want to use multiple nodes for inference.

### Run offline example

PhaseServe requires at least two GPUs to run. We provide an offline inference example in `examples/offline.py`.

### Run online example

To run online inference, launch the API server first, then launch the client example in `examples/online.py`.

## API Server Example

PhaseServe extends DistServe with two new scheduling options:

- `--context-sched-policy sjf`  
  Enables **Shortest-Job-First (SJF)** scheduling for the context / prefill phase, which is designed for requests with known but heterogeneous prompt lengths.

- `--decoding-sched-policy mlfq`  
  Enables **Multi-Level Feedback Queue (MLFQ)** scheduling for the decoding phase, which is designed for requests with unknown and dynamically evolving generation lengths.

A typical launch command for PhaseServe looks like:

```shell
python -m distserve.api_server.distserve_api_server \
    --host 127.0.0.1 \
    --port 8000 \
    --model facebook/opt-6.7b \
    --tokenizer facebook/opt-6.7b \
    --context-tensor-parallel-size 1 \
    --context-pipeline-parallel-size 1 \
    --decoding-tensor-parallel-size 1 \
    --decoding-pipeline-parallel-size 1 \
    --block-size 16 \
    --max-num-blocks-per-req 128 \
    --gpu-memory-utilization 0.95 \
    --swap-space 16 \
    --context-sched-policy sjf \
    --context-max-batch-size 32 \
    --context-max-tokens-per-batch 8192 \
    --decoding-sched-policy mlfq \
    --decoding-max-batch-size 32 \
    --decoding-max-tokens-per-batch 65536
```

For comparison, the default DistServe configuration uses:

* `--context-sched-policy fcfs`
* `--decoding-sched-policy fcfs`

Thus, switching from DistServe to PhaseServe mainly corresponds to replacing the default FCFS policies with:

* `sjf` for prefill
* `mlfq` for decoding

## Citation

If you use PhaseServe for your research, please cite our paper:

```bibtex
@article{phaseserve2026,
  title={PhaseServe: Efficient Phase-Disaggregated LLM Serving with Phase-Specialized Coordinated Scheduling},
  author={YOUR AUTHORS},
  journal={YOUR VENUE},
  year={2026}
}
```
