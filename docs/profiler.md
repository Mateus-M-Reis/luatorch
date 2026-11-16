# Profiling tensor methods

`torch.profiler` is an opt-in, Lua-level profiler for tensor method dispatch. It
records call counts and CPU time per method name without changing kernels or
dtypes.

```lua
torch.profiler.start()
local y = model:forward(x)
local loss = criterion:forward(y, target)
loss:backward()
torch.profiler.stop()

print(torch.profiler.format())
```

Output is sorted by total CPU time:

```text
profiler: tensor method calls (CPU time)
  add                1024 calls     12.431 ms total     0.012 ms avg ...
```

## API

| Function | Behavior |
| --- | --- |
| `torch.profiler.start()` | clear previous statistics and enable recording |
| `torch.profiler.stop()` | disable recording, keeping the statistics |
| `torch.profiler.reset()` | clear statistics without changing enabled state |
| `torch.profiler.report()` | sorted rows with `name`, `count`, `total`, `average`, `max` |
| `torch.profiler.format()` | human-readable report |

## Scope and limits

- Records methods called through a tensor instance (`t:add(x)`), including
  internal tensor-method calls made by other tensor methods.
- Does **not** record constructors such as `torch.zeros`, module functions such
  as `torch.matmul`, or static calls such as `Tensor.add(t, x)`.
- Uses `os.clock`, so it measures CPU time, not accelerator time.
- The wrapper is installed only while enabled; disabled dispatch keeps the
  normal metatable path.
