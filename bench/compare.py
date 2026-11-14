#!/usr/bin/env python3
"""Head-to-head luatorch vs PyTorch (CPU) benchmark.

Run with the benchmark venv:
    .venv-bench/bin/python bench/compare.py
    .venv-bench/bin/python bench/compare.py --compile

Both sides use one thread and best-of-N timing.  The Lua side is run in a
subprocess; PyTorch runs in this process.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

try:
    import torch
except ImportError:
    sys.exit(
        "PyTorch is required for bench/compare.py. Create the benchmark env:\n"
        "  uv venv .venv-bench --python 3.12\n"
        "  uv pip install --python .venv-bench/bin/python torch "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "then run .venv-bench/bin/python bench/compare.py")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_of(iterations, fn):
    for _ in range(3):
        fn()
    best = float("inf")
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


def run_luatorch():
    env = dict(os.environ)
    lua_path = ";".join([
        os.path.join(ROOT, "lua", "?.lua"),
        os.path.join(ROOT, "lua", "?", "init.lua"),
        "",
    ])
    env["LUA_PATH"] = lua_path
    command = [os.environ.get("LUAJIT", "luajit"),
               os.path.join(ROOT, "bench", "luatorch_bench.lua")]
    output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
    result = {}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        name, value = line.split("\t", 1)
        result[name] = float(value)
    return result


def run_pytorch(compile_chain):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    n = 1_000_000
    a = torch.randn(n)
    b = torch.randn(n)
    c = torch.randn(n)

    results = {}
    results["add"] = best_of(30, lambda: a + b)
    results["add_inplace"] = best_of(30, lambda: a.add_(b))
    results["chain"] = best_of(30, lambda: a * b + c)
    results["chain_relu"] = best_of(30, lambda: (a * b + c).relu())
    results["sum"] = best_of(30, lambda: a.sum())
    results["exp"] = best_of(20, lambda: a.exp())

    x = torch.randn(512, 512)
    y = torch.randn(512, 512)
    results["matmul512"] = best_of(5, lambda: x @ y)

    import torch.nn as nn
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randn(128, 784)
    labels = torch.ones(128, dtype=torch.long)

    def mlp_forward():
        return model(batch)

    def mlp_forward_backward():
        loss = criterion(model(batch), labels)
        mlp_optimizer.zero_grad()
        loss.backward()

    def mlp_step():
        loss = criterion(model(batch), labels)
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()

    for _ in range(3):
        mlp_forward_backward()
        mlp_optimizer.step()

    results["mlp_forward"] = best_of(50, mlp_forward)
    results["mlp_forward_backward"] = best_of(50, mlp_forward_backward)
    results["mlp_adam_step"] = best_of(50, mlp_optimizer.step)
    results["mlp_train_step"] = best_of(50, mlp_step)

    train_x = torch.randn(1_000_000, requires_grad=True)
    train_y = torch.randn(1_000_000, requires_grad=True)
    train_z = torch.randn(1_000_000, requires_grad=True)

    def chain_train():
        train_x.grad = None
        train_y.grad = None
        train_z.grad = None
        output = (train_x * train_y + train_z).relu()
        output.sum().backward()

    results["chain_train"] = best_of(15, chain_train)

    reduce_a = torch.randn(1_000_000)
    reduce_b = torch.randn(1_000_000)
    reduce_c = torch.randn(1_000_000)
    results["mse_eager"] = best_of(
        50, lambda: ((reduce_a - reduce_b) ** 2).mean())
    results["reduce_chain_eager"] = best_of(
        50, lambda: (((reduce_a - reduce_b) ** 2 + reduce_c * 0.5).relu()).mean())

    train_prediction = torch.randn(1_000_000, requires_grad=True)
    train_target = torch.randn(1_000_000, requires_grad=True)

    def mse_train_eager():
        train_prediction.grad = None
        train_target.grad = None
        loss = ((train_prediction - train_target) ** 2).mean()
        loss.backward()

    results["mse_train_eager"] = best_of(30, mse_train_eager)

    results["logsumexp_eager"] = best_of(
        30, lambda: (reduce_a * reduce_b).logsumexp())
    logsumexp_p = torch.randn(1_000_000, requires_grad=True)
    logsumexp_q = torch.randn(1_000_000, requires_grad=True)

    def logsumexp_train_eager():
        logsumexp_p.grad = None
        logsumexp_q.grad = None
        loss = (logsumexp_p * logsumexp_q).logsumexp()
        loss.backward()

    results["logsumexp_train_eager"] = best_of(30, logsumexp_train_eager)

    log_softmax_input = torch.randn(4096, 256)
    results["log_softmax_eager"] = best_of(
        50, lambda: torch.log_softmax(log_softmax_input, dim=1))
    log_softmax_train_input = torch.randn(4096, 256, requires_grad=True)
    log_softmax_upstream = torch.randn(4096, 256)

    def log_softmax_train_eager():
        log_softmax_train_input.grad = None
        output = torch.log_softmax(log_softmax_train_input, dim=1)
        output.backward(log_softmax_upstream)

    results["log_softmax_train_eager"] = best_of(30, log_softmax_train_eager)

    cross_entropy_input = torch.randn(4096, 256, requires_grad=True)
    cross_entropy_labels = torch.ones(4096, dtype=torch.long)

    def cross_entropy_eager():
        cross_entropy_input.grad = None
        loss = torch.nn.functional.cross_entropy(
            cross_entropy_input, cross_entropy_labels)
        loss.backward()

    results["cross_entropy_eager"] = best_of(30, cross_entropy_eager)

    tiny = torch.randn(16)
    tiny_b = torch.randn(16)
    tiny_c = torch.randn(16)
    results["tiny_add"] = best_of(20000, lambda: tiny + tiny_b)
    results["tiny_chain"] = best_of(20000, lambda: (tiny * tiny_b + tiny_c).relu())

    if compile_chain:
        print("compiling (a*b+c).relu() ...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        compiled = torch.compile(lambda p, q, r: (p * q + r).relu())
        compiled(a, b, c)
        print("compile+warmup: %.1fs" % (time.perf_counter() - started),
              file=sys.stderr)
        results["chain_compiled"] = best_of(30, lambda: compiled(a, b, c))
    return results


def print_table(lua, torch_results, compiled):
    rows = [
        ("add (out-of-place)", "add", "add", None),
        ("add_ (in-place)", "add_inplace", "add_inplace", None),
        ("a*b+c", "chain", "chain", None),
        ("(a*b+c).relu()", "chain_relu", "chain_relu", None),
        ("(a*b+c).relu() fused, into", "chain_fused", "chain_relu", "chain_compiled"),
        ("(a*b+c).relu() fused, alloc", "chain_fused_alloc", "chain_relu", "chain_compiled"),
        ("sum", "sum", "sum", None),
        ("exp", "exp", "exp", None),
        ("matmul 512", "matmul512", "matmul512", None),
        ("tiny add (16)", "tiny_add", "tiny_add", None),
        ("tiny chain (16)", "tiny_chain", "tiny_chain", None),
        ("tiny chain fused (16)", "tiny_chain_fused", "tiny_chain", None),
        ("MLP forward 128x784", "mlp_forward", "mlp_forward", None),
        ("MLP forward+backward", "mlp_forward_backward", "mlp_forward_backward", None),
        ("MLP Adam step", "mlp_adam_step", "mlp_adam_step", None),
        ("MLP train step", "mlp_train_step", "mlp_train_step", None),
        ("MSE reduce (fused)", "mse_fused", "mse_eager", None),
        ("chain reduce (fused)", "reduce_chain_fused", "reduce_chain_eager", None),
        ("MSE loss train (fused)", "mse_train_fused", "mse_train_eager", None),
        ("logsumexp (fused)", "logsumexp_fused", "logsumexp_eager", None),
        ("logsumexp train (fused)", "logsumexp_train_fused", "logsumexp_train_eager", None),
        ("log_softmax (fused)", "log_softmax_fused", "log_softmax_eager", None),
        ("log_softmax train (fused)", "log_softmax_train_fused", "log_softmax_train_eager", None),
        ("cross_entropy 4096x256 (fused)", "cross_entropy_fused", "cross_entropy_eager", None),
        ("chain train (eager)", "chain_train", "chain_train", None),
        ("chain train (fused)", "chain_train_fused", "chain_train", None),
    ]

    header = f"{'workload':34s} {'luatorch':>10s} {'torch eager':>12s} {'speedup':>9s}"
    if compiled:
        header += f" {'torch.compile':>13s} {'speedup':>9s}"
    print(header)
    print("-" * len(header))
    for label, lua_key, torch_key, compiled_key in rows:
        lua_ms = lua.get(lua_key)
        torch_ms = torch_results.get(torch_key)
        if lua_ms is None or torch_ms is None:
            continue
        speedup = torch_ms / lua_ms if lua_ms > 0 else 0
        line = f"{label:34s} {lua_ms:9.3f}ms {torch_ms:10.3f}ms {speedup:8.2f}x"
        if compiled and compiled_key:
            compiled_ms = torch_results.get(compiled_key)
            if compiled_ms is not None:
                line += f" {compiled_ms:11.3f}ms {compiled_ms / lua_ms:8.2f}x"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true",
                        help="also benchmark torch.compile (slow first run)")
    args = parser.parse_args()

    print(f"luatorch vs PyTorch {torch.__version__}, 1 thread", file=sys.stderr)
    lua = run_luatorch()
    torch_results = run_pytorch(args.compile)
    print(f"PyTorch threads: {torch.get_num_threads()}", file=sys.stderr)
    print()
    print_table(lua, torch_results, args.compile)


if __name__ == "__main__":
    main()
