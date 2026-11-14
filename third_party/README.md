# Third-party components

`luatorch` itself is BSD-3-Clause. The following components are used at build or
run time.

## libtorch (downloaded, not committed)

- **What:** official PyTorch C++ distribution, CPU build, C++11 ABI, version 2.5.1
- **Used by:** `csrc/luatorch.cpp`, linked into `libluatorch.so`
- **License:** BSD-style (see the LICENSE inside the distribution)
- **How to fetch:** `scripts/fetch_libtorch.sh`
- **Location:** `third_party/libtorch/` (git-ignored)

## PyTorch source (reference only)

- **What:** shallow, blobless clone of <https://github.com/pytorch/pytorch>
- **Used by:** developers reading ATen signatures; not required to build
- **Location:** `third_party/pytorch-src/` (git-ignored)
- **License:** BSD-style

## luafun (vendored)

- **What:** Lua Fun, a high-performance functional library for LuaJIT
- **Upstream:** <https://github.com/rtsisyk/luafun>
- **License:** MIT/X11, Copyright (c) 2013-2017 Roman Tsisyk
- **Files:** `lua/fun.lua` (runtime copy), `third_party/fun.lua` (upstream copy),
  `third_party/luafun-COPYING.md`
- **Why:** replaces Moses with a lazy, JIT-friendly, single-file library

Moses is **not** a dependency.
