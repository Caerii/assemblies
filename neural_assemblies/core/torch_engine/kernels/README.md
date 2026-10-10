# The fused CUDA kernels

One compilation unit, cut by topic. `_fused_cuda.py` concatenates the `.cu` files **in name
order** (the numeric prefixes) into the source `torch.utils.cpp_extension.load_inline` compiles,
with `bindings.cpp` as the host-side declarations. The build is named by a hash of that source
(`_fused_cuda.build_name`), so a checkout with different kernels builds apart from this one, and
an unchanged hash proves an edit changed no byte the compiler sees: the files were cut from the
former `_CUDA_SRC` string with the hash unchanged (`na_fused_cuda_e178998e2fd0`).

| file | what it holds |
|---|---|
| `00_prelude.cu` | includes, `NA_STREAM` (every launch on torch's current stream, so graphs can capture it), the select's block constants |
| `01_hash_and_select.cu` | `fmix32`, the generated connectome's drive (`hashed_drive_kernel`), the k-WTA select keys and kernel |
| `02_store_fiber_kernels.cu` | the store fiber's potentiation corrections, exact in-degree, column masses, CSR and split-count forms |
| `03_hash_and_store_bindings.cu` | host bindings: `hashed_drive`, `hashed_indegree`, `column_mass`, `dev_correct`, `dev_correct_csr` |
| `04_relative_pricing.cu` | max-relative pricing kernels (column scaling, no clip) |
| `05_presence_and_scheduling.cu` | the connectome as a presence bitmask; the scheduled organ's price and key helpers |
| `06_organ_kernels.cu` | the DENSE ORGAN fiber: count-matrix drive (scalar and four-column) and write, templated on the count width |
| `06a_organ_packed_kernels.cu` | opt-in PACKED 4-bit organ counts (`count_dtype="int4"`): drive and write kernels and their host launchers; two counts a byte, rows of `RB` bytes (not `NB`: the prelude defines that name) |
| `07_present_kernels.cu` | the PRESENT-ONLY fiber: per-row lists of existing synapses; drive, write, train, probe |
| `08_store_and_presence_bindings.cu` | host bindings: exact and relative corrections and column masses, `hashed_presence` |
| `09_organ_drive_binding.cu` | host binding: `organ_drive` (dispatch on the count width, brain map, per-brain tables) |
| `10_fused_round_steps.cu` | fused elementwise steps of a round: `stim_add`, `charge` (same float operations as the torch chain) |
| `11_organ_write_binding.cu` | host binding: `organ_write` |
| `12_present_bindings.cu` | host bindings of the present-only fiber |
| `13_select_binding.cu` | host binding: `topk_select` |
| `bindings.cpp` | the declarations `load_inline` binds to Python |

The order is the compilation unit's dependency order (a binding follows the kernels it launches);
a new kernel goes into the file of its topic, and a new binding after the kernels it calls.
These files are part of every run's source archive (`research/runner.py` collects `.cu` and `.cpp`).
