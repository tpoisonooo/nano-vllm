import os
import time
import sys
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

def bench(llm: LLM) -> int:
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(total_tokens)
    return total_tokens

def main(model_path: str):
    seed(0)
    path = os.path.expanduser(model_path)
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # warmup
    print("Warmup..")
    bench(llm)

    # loop
    print("Bench..")
    loop = 8
    total_tokens = 0
    start = time.time()
    for _ in range(loop):
        total_tokens += bench(llm)
    timecost = time.time() - start
    throughput = total_tokens / timecost
    print(
        f"Total: {total_tokens}tok, Time: {timecost:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "~/huggingface/Qwen3-0.6B/"
    main(model_path=model_path)
