# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper test: compare HF vs SGLang Jamba inference quality."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List

import requests
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class SimilarityResult:
    jaccard: float
    length_ratio: float
    is_nontrivial: bool


def _tokenize_for_similarity(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        word = "".join([c for c in raw if c.isalnum()])
        if word:
            tokens.append(word)
    return tokens


def compute_similarity(ref: str, cand: str) -> SimilarityResult:
    ref_tokens = _tokenize_for_similarity(ref)
    cand_tokens = _tokenize_for_similarity(cand)
    if not ref_tokens or not cand_tokens:
        return SimilarityResult(jaccard=0.0, length_ratio=0.0, is_nontrivial=False)
    ref_set, cand_set = set(ref_tokens), set(cand_tokens)
    jaccard = len(ref_set & cand_set) / max(1, len(ref_set | cand_set))
    length_ratio = len(cand_tokens) / max(1, len(ref_tokens))
    is_nontrivial = len(cand_tokens) >= 3
    return SimilarityResult(
        jaccard=jaccard, length_ratio=length_ratio, is_nontrivial=is_nontrivial
    )


def generate_hf(model_name: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = False
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def wait_for_health(base_url: str, timeout_s: int = 600) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = requests.get(f"{base_url}/health_generate", timeout=5)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError("SGLang server failed to become healthy in time.")


def launch_sglang_server(
    model: str,
    base_url: str,
    mem_fraction_static: float,
    disable_radix_cache: bool,
    device: str,
) -> subprocess.Popen:
    host, port = base_url.replace("http://", "").split(":")
    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--trust-remote-code",
        "--mem-fraction-static",
        str(mem_fraction_static),
    ]
    if disable_radix_cache:
        command.append("--disable-radix-cache")
    if device:
        command.extend(["--device", device])
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "0")
    return subprocess.Popen(command, env=env)


def generate_sglang(
    base_url: str, prompt: str, temperature: float, max_new_tokens: int, top_p: float
) -> str:
    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        },
    }
    resp = requests.post(f"{base_url}/generate", json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["text"].strip()


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai21labs/AI21-Jamba2-3B")
    parser.add_argument(
        "--prompt",
        default="Explain SGLang in one sentence with a concrete example.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:30099")
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--disable-radix-cache", action="store_true", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--temperature-seq", default="0.0,0.2,0.5,0.7,1.0")
    parser.add_argument("--top-p-seq", default="1.0,0.95,0.9,0.85,0.8")
    args = parser.parse_args(list(argv))

    temps = parse_float_list(args.temperature_seq)
    top_ps = parse_float_list(args.top_p_seq)
    if not temps:
        temps = [0.0]
    if not top_ps:
        top_ps = [1.0]

    print("Generating HF reference...")
    hf_text = generate_hf(args.model, args.prompt, args.max_new_tokens)
    print("HF output:", json.dumps(hf_text, ensure_ascii=True))

    proc = launch_sglang_server(
        model=args.model,
        base_url=args.base_url,
        mem_fraction_static=args.mem_fraction_static,
        disable_radix_cache=args.disable_radix_cache,
        device=args.device,
    )

    try:
        wait_for_health(args.base_url)
        best = None
        for attempt in range(args.max_attempts):
            temp = temps[min(attempt, len(temps) - 1)]
            top_p = top_ps[min(attempt, len(top_ps) - 1)]
            sgl_text = generate_sglang(
                base_url=args.base_url,
                prompt=args.prompt,
                temperature=temp,
                max_new_tokens=args.max_new_tokens,
                top_p=top_p,
            )
            sim = compute_similarity(hf_text, sgl_text)
            print(
                f"Attempt {attempt + 1}: temp={temp} top_p={top_p} "
                f"jaccard={sim.jaccard:.3f} length_ratio={sim.length_ratio:.2f} "
                f"nontrivial={sim.is_nontrivial}"
            )
            print("SGLang output:", json.dumps(sgl_text, ensure_ascii=True))
            if best is None or sim.jaccard > best[0].jaccard:
                best = (sim, sgl_text)
            if (
                sim.is_nontrivial
                and sim.jaccard >= args.similarity_threshold
                and 0.5 <= sim.length_ratio <= 2.0
            ):
                print("PASS: SGLang output meets similarity threshold.")
                return 0

        print("FAIL: No SGLang output met the similarity threshold.")
        if best is not None:
            sim, sgl_text = best
            print(
                f"Best attempt: jaccard={sim.jaccard:.3f} "
                f"length_ratio={sim.length_ratio:.2f} nontrivial={sim.is_nontrivial}"
            )
            print("Best SGLang output:", json.dumps(sgl_text, ensure_ascii=True))
        return 1
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
