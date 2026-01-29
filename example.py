import os
import sys
import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.utils.profiler import ThreadSafeLayerProfiler


def main(model_path: str):
    path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    profiler = ThreadSafeLayerProfiler()
    profiler.register_model(llm.model_runner.model)

    with torch.no_grad():
        outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    profiler.print_stats(levels=[4])
    profiler.remove_hooks()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "~/huggingface/Qwen3-0.6B/"
    main(model_path=model_path)
