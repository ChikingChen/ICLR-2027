# CODEX 变更说明

## 2026-05-27 记录默认三模型口径

### 变更文件

- `AGENTS.md`

### 变更目的

按用户要求记录后续默认不再关注 `Yi-9B-200K`：之后用户说“三个模型”时，默认只指 `Llama-3.1-8B`、`Qwen2.5-7B-Instruct-1M` 和 `GLM-4-9B-Chat-1M`，不包括 Yi。Yi 的本地目录说明仍保留，但新实验、新汇总和三模型命令默认不纳入 Yi，除非用户明确点名。

同时将 4k 全任务示例命令从四模型改为默认三模型，并把示例输出文件名改为 `ruler_results_4k_three_models.csv`。

## 2026-05-27 更新 FlashAttention RULER 分数和 timing 汇总

### 变更文件

- `RULER/benchmark_root/local_eval/FlashAttention/GLM_flashattention_ruler_scores.csv`
- `RULER/benchmark_root/local_eval/FlashAttention/Llama_flashattention_ruler_scores.csv`
- `RULER/benchmark_root/local_eval/FlashAttention/Qwen_flashattention_ruler_scores.csv`

### 变更目的

从 `RULER/benchmark_root/local_eval/` 中已有预测结果重新按 RULER 原生 synthetic metric 计算分数，并为 FlashAttention 运行结果单独整理 GLM、Llama 和 Qwen 三个模型的 4k、8k、16k 汇总表。原目录 `flashattention_ruler_scores_4k_8k_16k/` 已重命名为更通用的 `FlashAttention/`，便于后续继续加入其他 benchmark 或长度。

每个 CSV 包含 `length`、13 个 RULER 子任务分数、`overall` 平均分，并在 `overall` 后追加 `avg_prefill_attention_kernel_ms` 和 `avg_decode_attention_kernel_ms_per_token`。这两个 timing 字段来自 `ruler_results_4k_64k_all_models_token_ppl_timing_summary_by_model_and_length.csv`，表示对应 `model + length` 下 13 个任务第 0 条样本 attention profiler 的平均 prefill attention kernel 时间和平均 decode 每 token attention kernel 时间。

### 验证方式

已检查三张 CSV 均为 4 行（含表头）、17 列（`length` + 13 个任务 + `overall` + 2 个 attention timing 字段），数据行顺序均为 `4k`、`8k`、`16k`，新增 timing 字段均为非空数值。

## 2026-05-27 新增 Llama mask BOS RULER 分数汇总

### 变更文件

- `RULER/benchmark_root/local_eval/llama_mask_bos_ruler_scores/Llama_mask_bos_ruler_scores.csv`

### 变更目的

从 `RULER/benchmark_root/local_eval/Llama-3.1-8B-mask-bos/` 中已有预测结果重新按 RULER 原生 synthetic metric 计算分数，并为 Llama 遮住 BOS token 的运行结果单独整理汇总表。当前本地 mask BOS 结果只覆盖 `4096` 长度，因此 CSV 目前只有 `4k` 一行；列结构与 FlashAttention 汇总保持一致，包含 `length`、13 个 RULER 子任务分数和 `overall` 平均分。

### 验证方式

已检查 CSV 为 2 行（含表头）、15 列（`length` + 13 个任务 + `overall`），数据行是 `4k`，所有任务分数和 `overall` 均非空。

## 2026-05-26 新增 pooling attention CSV 汇总

### 变更文件

- `tools/compare_pooling_attention.py`
  - 新增 `pooling_attention_comparison.csv` 输出。
  - CSV 按 `layer + head + block` 展开，记录 block 内普通 token attention 列表、attention 之和、avg pooling 分数、max pooling 分数、差值和是否相等。
  - `metadata.json` 补充 CSV 统计口径说明。

- `tests/test_pooling_attention_compare.py`
  - 新增 CSV 行生成和文件写出测试，验证 `sum / avg / max` 三种口径。

### 变更目的

本次变更用于更直观地检查一个 pooling block 是否能代表其覆盖的普通 token：如果 block 内多个 token 被压成一个 pooling token，可直接在 CSV 中比较普通 token attention 之和与 max/avg pooling 分数的差异。

### 运行方式

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n model python -u tools/compare_pooling_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --query-generated-index 0 \
  --block-size 32 \
  --top-k-blocks 8 \
  --max-new-tokens 128 \
  --output-dir attention_dumps/pooling_token_compare/llama_niah_single_1_4k_sample0_block32 \
  --overwrite
```

重点查看：

```text
attention_dumps/pooling_token_compare/llama_niah_single_1_4k_sample0_block32/pooling_attention_comparison.csv
```

### 验证方式

```bash
conda run --no-capture-output -n model python -m unittest tests.test_pooling_attention_compare
conda run --no-capture-output -n model python -B -m py_compile tools/compare_pooling_attention.py tests/test_pooling_attention_compare.py
```

## 2026-05-26 接入 RULER 批量 BOS mask 评测链路

### 2026-05-26 追加修复

- 修复 `device_map="auto"` 场景下 Accelerate hook 让 `generate()` 参数校验看不到 `position_ids` 的问题。
- RULER 批量 masked BOS 生成现在不会把 `position_ids` 作为外部 `generate()` 参数传入，而是在 `prepare_inputs_for_generation` 内根据 `cache_position` 注入原始绝对位置，避免 `<|begin_of_text|>` 被遮住后后续 token 的位置整体左移。
- 新增回归测试覆盖 `generate()` 输入中移除 `position_ids` 和 masked BOS 生成准备阶段按 `cache_position` 写回位置。

### 变更文件

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 `--mask-bos-token` 参数，并在构造子进程命令时透传为 `pred/call_api.py --mask_bos_token`。

- `RULER/scripts/pred/call_api.py`
  - 新增 `--mask_bos_token` 参数。
  - 限定该参数只支持 `--server_type hf`。
  - 预测 jsonl 额外保留 `attention_mask_ablation` 字段，便于确认本次运行确实使用了 BOS mask。

- `RULER/scripts/pred/model_wrappers.py`
  - Hugging Face wrapper 支持保留 BOS token、把 position 0 的 `attention_mask` 置为 0，并显式传入原始 `position_ids`。
  - 该 mask 同步作用于普通生成、PPL 统计、prefill/decode timing、attention summary 和 attention kernel profiler 路径。

- `tests/test_run_parquet_parallel.py`、`tests/test_call_api_progress.py`、`tests/test_model_wrappers.py`
  - 新增 RULER runner 参数透传、非 HF 后端拒绝和 BOS mask helper 行为测试。

- `AGENTS.md`
  - 补充 RULER 4k 遮住 `<|begin_of_text|>` 的推荐运行命令和参数说明。

### 推荐命令

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B-mask-bos=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks all \
  --gpus 2 \
  --max-workers 1 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 0.5 \
  --log-batch-progress \
  --mask-bos-token \
  --overwrite-existing \
  --auto-evaluate \
  --timing-file ../benchmark_root/local_eval/ruler_timing_4k_llama_mask_bos.jsonl \
  --report-file ../benchmark_root/local_eval/ruler_results_4k_llama_mask_bos.csv
```

### 验证方式

```bash
conda run --no-capture-output -n model python -B -m unittest tests.test_run_parquet_parallel tests.test_call_api_progress tests.test_model_wrappers
```

## 2026-05-26 新增 BOS attention sink 汇总工具

### 变更文件

- `tools/summarize_bos_attention.py`
  - 新增完整 attention dump 汇总脚本。
  - 统计 `<|begin_of_text|>` 在所有 `generated token × layer × head` 组合中 attention 是否为 top-1。
  - 输出 BOS top-1 次数、比例、平均 attention、最小值、最大值和基础元数据。
  - 默认从 `prompt_tokens.jsonl` 按 token 文本定位 BOS，也支持 `--bos-position` 手动指定。

- `tests/test_attention_dump_tools.py`
  - 新增 BOS 汇总脚本单元测试，覆盖跨生成 token 统计、缺失 BOS 报错和 CLI 输出。

- `AGENTS.md`
  - 补充 `tools/summarize_bos_attention.py` 的用途、统计口径和典型命令。

### 变更目的

本次变更用于快速检查完整 attention dump 中 `<|begin_of_text|>` 是否作为 attention sink：脚本不重新跑模型、不修改 dump，只读取已有 `token_XXXX.npy` 文件并汇总 BOS 在所有生成 token、层和 head 上的 attention 排名与平均分配。

### 运行方式

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python tools/summarize_bos_attention.py \
  --dump-dir attention_dumps/llama_niah_single_1_4k_sample0
```

如果自动定位不到 BOS，可以手动指定：

```bash
  --bos-position 0
```

### 验证方式

```bash
conda run --no-capture-output -n model python -m unittest tests.test_attention_dump_tools -v
conda run --no-capture-output -n model python -B -m py_compile tools/summarize_bos_attention.py tests/test_attention_dump_tools.py
conda run --no-capture-output -n model python tools/summarize_bos_attention.py --dump-dir attention_dumps/llama_niah_single_1_4k_sample0
```

### 假设和限制

- “多少个头中注意力第一”按 `generated token × layer × head` 统计，不按静态 head 去重。
- top-1 判断使用完整 attention vector 的最大值；并列最大时 BOS 也计为第一。
- 脚本只打印终端汇总，不写新的 json/csv 文件。

## 2026-05-26 新增 Llama BOS token attention mask 实验开关

### 变更文件

- `tools/attention_mask_utils.py`
  - 新增共享 BOS attention mask ablation helper。
  - `--mask-bos-token` 语义固定为保留 `<|begin_of_text|>` token，但将 position 0 的 `attention_mask` 置为 0，并显式保留原始 `position_ids`。

- `tools/dump_llama_attention.py`
  - 新增 `--mask-bos-token` 参数。
  - 支持在完整 attention dump 实验中遮住 tokenizer 自动插入的 BOS token。
  - `metadata.json` 和 `summary.md` 新增生成文本预览和 `attention_mask_ablation` 元信息。
  - `prompt_tokens.jsonl` 新增 `attention_mask` 和 `masked` 字段，便于确认 position 0 是否被遮住。

- `tools/compare_pooling_attention.py`
  - 新增 `--mask-bos-token` 参数。
  - 支持在 pooling token 与细粒度 attention 对照实验中使用同一条 BOS mask 生成和 replay 路径。
  - `metadata.json`、`summary.md` 和 `tokens.jsonl` 同步记录遮蔽元信息。

- `tests/test_attention_dump_tools.py`
  - 新增 BOS mask helper、错误场景、token mask 字段和 metadata 字段测试。

- `tests/test_pooling_attention_compare.py`
  - 新增 pooling 工具 BOS mask helper 和 CLI 参数测试。

- `AGENTS.md`
  - 补充 `--mask-bos-token` 的用途、语义和输出字段说明。

### 变更目的

本次变更用于对比 Llama 生成时 `<|begin_of_text|>` 作为序列起点和潜在 attention sink 的影响。实验不会删除 BOS token，而是在生成和 KV cache replay 阶段禁止后续 token attend 到它，从而观察生成文本、完整 attention 分布和 pooling block 排名的变化。

### 运行方式

完整 attention dump 的遮蔽实验：

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n model python -u tools/dump_llama_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --output-dir attention_dumps/llama_niah_single_1_4k_sample0_mask_bos \
  --dtype float32 \
  --max-new-tokens 128 \
  --mask-bos-token \
  --overwrite
```

pooling token 对照的遮蔽实验：

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n model python -u tools/compare_pooling_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --query-generated-index 0 \
  --block-size 128 \
  --top-k-blocks 8 \
  --max-new-tokens 128 \
  --output-dir attention_dumps/pooling_token_compare/llama_niah_single_1_4k_sample0_mask_bos \
  --mask-bos-token \
  --overwrite
```

### 验证方式

```bash
conda run --no-capture-output -n model python -B -m unittest tests.test_attention_dump_tools tests.test_pooling_attention_compare
conda run --no-capture-output -n model python -B -m py_compile tools/attention_mask_utils.py tools/dump_llama_attention.py tools/compare_pooling_attention.py tests/test_attention_dump_tools.py tests/test_pooling_attention_compare.py
conda run --no-capture-output -n model python -B tools/dump_llama_attention.py --help
conda run --no-capture-output -n model python -B tools/compare_pooling_attention.py --help
```

### 假设和限制

- 上述单样本诊断工具的 `--mask-bos-token` 仍然只支持 batch size 1。
- 如果 tokenizer 编码后的第 0 个 token 不是 `bos_token_id`，脚本会直接报错，避免遮错 token。
- RULER 批量 benchmark runner 的 BOS mask 支持见本文件 2026-05-26 的批量评测链路记录。

## 2026-05-23 新增 pooling token 与细粒度 attention 对照工具

### 变更文件

- `tools/compare_pooling_attention.py`
  - 新增 Llama 单样本 attention 对照诊断脚本。
  - 先对一个 RULER 样例执行确定性生成，再 replay 指定生成 token 的完整 attention。
  - 按固定 block size 把 prompt token 切成 pooling token/block，同时输出 `max pooling` 和 `avg pooling` 的 block 级注意力分数。
  - 输出每个原始 prompt token 的 full attention 分数，支持保留每层每 head 的明细；也会保存压缩后的 `attention_detail.npz`。
  - 默认产物包括 `metadata.json`、`tokens.jsonl`、`pooling_tokens.jsonl`、`fine_tokens.jsonl`、`pooling_vs_fine_summary.jsonl`、`attention_detail.npz` 和 `summary.md`。

- `tests/test_pooling_attention_compare.py`
  - 新增不依赖 GPU 的单元测试，覆盖 block 切分、max/avg pooling 分数、细粒度 token 分数和输出文件格式。

- `AGENTS.md`
  - 新增 `tools/compare_pooling_attention.py` 的工具说明、典型运行命令、输出文件说明和限制。

- `CODEX_CHANGES.md`
  - 记录本次 pooling token attention 对照工具变更。

### 变更目的

本次变更用于检查 top-k block 选择中 pooling token 是否能代表其覆盖的多个原始 token。脚本会在同一个样本上同时给出 pooling token/block 的 `max`、`avg` 分数，以及 block 内每个细粒度 token 在 full attention 中的真实注意力分数，方便直接比较两种 pooling 方法的代表性。

### 运行方式

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python -u tools/compare_pooling_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --query-generated-index 0 \
  --block-size 128 \
  --top-k-blocks 8 \
  --max-new-tokens 128 \
  --output-dir attention_dumps/pooling_token_compare/llama_niah_single_1_4k_sample0 \
  --overwrite
```

如果只想保留汇总分数，并避免在 jsonl 中写入每层每 head 的长明细，可以增加：

```bash
  --omit-layer-head-details
```

### 验证方式

```bash
conda run --no-capture-output -n model python -B -m unittest tests.test_pooling_attention_compare
conda run --no-capture-output -n model python -B -m py_compile tools/compare_pooling_attention.py tests/test_pooling_attention_compare.py
conda run --no-capture-output -n model python -B tools/compare_pooling_attention.py --help
```

### 假设和限制

- 当前服务器 CUDA 仍不可用时，真实 Llama 跑数命令无法完成；脚本实现和单元测试不依赖 GPU。
- 该脚本只做诊断分析，不实现或替换真实稀疏 attention kernel。
- 默认主比较口径是跨 layer/head 平均后的 token attention；每层每 head 的完整数值保存在 jsonl 明细和 `attention_detail.npz` 中。

## 2026-05-21 新增 RULER prefill/decode timing 和首样本 attention profiler

### 变更文件

- `RULER/scripts/pred/model_wrappers.py`
  - 新增 Hugging Face 生成阶段 prefill/decode forward 耗时采集。
  - 新增 attention kernel profiler 汇总逻辑，按 profiler event 名称过滤 `flash_attn`、`scaled_dot_product`、`fmha`、`sdpa`、`attention` 等相关 CUDA 事件。
  - `HuggingFaceModel` 新增 `log_prefill_decode_timing` 和 `profile_attention_kernels` 开关；开启后强制使用直接模型路径，不走 pipeline。

- `RULER/scripts/pred/call_api.py`
  - 新增 `--log_prefill_decode_timing`、`--profile_attention_kernels` 和 `--attention_profile_sample_offset`。
  - 固定 `--batch_size 1`，传入非 `1` 会直接报错。
  - 新增 `<任务>.generation_timing.jsonl` sidecar：每条样本写 `sample_timing`，每个任务最多写一条固定第 0 行的 `attention_profile`。

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 runner 参数 `--log-prefill-decode-timing`、`--profile-attention-kernels` 和 `--attention-profile-sample-offset`。
  - 固定 `--batch-size 1`，传入非 `1` 会报错；构造子进程命令时始终传 `--batch_size 1`。
  - `--overwrite-existing` 现在会删除 `<任务>.generation_timing.jsonl`。

- `RULER/scripts/eval/collect_results.py`
  - 新增读取 `<任务>.generation_timing.jsonl` 并汇总到 detail、summary_by_model、summary_by_model_and_length 和 summary_by_task csv。
  - detail 新增 `sample_timing_records`、`prefill_forward_ms_total`、`decode_forward_ms_total`、`decode_forward_ms_per_token_avg`、`attention_profile_sample_line_no`、`attention_profile_sample_index`、`prefill_attention_kernel_ms`、`decode_attention_kernel_ms_total`、`decode_attention_kernel_ms_per_token_avg` 和 `attention_kernel_event_count` 等字段。

- `tests/test_model_wrappers.py`
  - 新增 generation timing 聚合和 attention profiler event 过滤测试。

- `tests/test_call_api_progress.py`
  - 新增 batch size 固定为 1、首条样本 profiler 选择、timing sidecar 记录构造测试。

- `tests/test_run_parquet_parallel.py`
  - 新增 runner 新参数透传、固定 batch size 校验和覆盖删除 timing sidecar 测试。

- `tests/test_collect_results.py`
  - 新增 generation timing sidecar 汇总测试。

- `AGENTS.md`
  - 同步新增参数、输出文件、固定 batch size 行为、summary 字段和限制说明。

- `CODEX_CHANGES.md`
  - 记录本次 prefill/decode timing 和 attention profiler 变更。

### 变更目的

本次变更用于让本地 RULER Hugging Face 推理在保留原有分数、PPL 和任务级耗时的基础上，进一步记录推理阶段内部耗时：普通 prefill/decode forward 耗时覆盖每条样本；严格 attention kernel profiler 固定每个 `model + length + task` 只采输入 jsonl 第 0 行，避免全量 profiler 带来过大开销。

### 主要函数和类

- `summarize_generation_timing`
  - 聚合 prefill/decode forward 耗时，并按生成 token 数计算 `decode_forward_ms_per_token_avg`。
- `summarize_attention_kernel_events`
  - 从 `torch.profiler` events 中过滤 attention 相关设备事件并汇总耗时。
- `ForwardTimingCollector`
  - 在 `generate()` 期间临时包裹 `model.forward`，按 prefill/decode 阶段记录耗时。
- `HuggingFaceModel.profile_attention_kernels_for_prompt`
  - 对单条 prompt 执行严格 profiler，分别返回 prefill attention kernel 时间和 decode attention kernel 时间。
- `validate_runtime_args`
  - 统一校验 `call_api.py` 参数，保证 batch size 和 profiler 采样口径固定。
- `build_generation_timing_record`、`build_attention_profile_record`
  - 构造 `<任务>.generation_timing.jsonl` 中的 `sample_timing` 和 `attention_profile` 记录。
- `aggregate_generation_timing`
  - 汇总 timing sidecar 到最终 csv 明细和汇总表。

### 运行方式

记录每个生成 token 的 PPL、每条样本 prefill/decode forward 耗时，并对每个任务第 0 行采样严格 attention kernel 时间：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-token-ppl \
  --log-prefill-decode-timing \
  --profile-attention-kernels \
  --overwrite-existing \
  --auto-evaluate
```

输出文件示例：

```text
RULER/benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred/niah_single_1.generation_timing.jsonl
RULER/benchmark_root/local_eval/ruler_results.csv
RULER/benchmark_root/local_eval/ruler_results_summary_by_model.csv
```

### 测试和验证

本次已经运行：

```bash
conda run --no-capture-output -n model python -B -m unittest tests.test_model_wrappers tests.test_call_api_progress tests.test_run_parquet_parallel tests.test_collect_results
conda run --no-capture-output -n model python -B -m unittest discover -s tests
conda run --no-capture-output -n model python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py \
  tools/count_ruler_samples.py
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-token-ppl \
  --log-prefill-decode-timing \
  --profile-attention-kernels \
  --dry-run
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 当前评测口径固定 `batch_size=1`；runner 和 `call_api.py` 都会拒绝非 `1` 的 batch size。
- 严格 attention profiler 固定只采输入 jsonl 第 0 行；`sample_line_no=0` 是行号，`sample_index` 是样本原始 `index` 字段。
- `prefill_forward_ms` 和 `decode_forward_ms_total` 是完整 forward 阶段耗时，不是纯 attention 时间。
- `prefill_attention_kernel_ms` 和 `decode_attention_kernel_ms_total` 是 profiler 按 kernel 名称过滤后的严格 attention 设备时间；如果 CUDA 不可用或 profiler 没有匹配事件，会写入 warning 或空值。
- `--profile-attention-kernels` 不能和会强制 eager attention 的 `--log-attention-scores` 同时使用。

## 2026-05-21 新增生成 token 级 PPL 明细记录

### 变更文件

- `RULER/scripts/pred/model_wrappers.py`
  - 扩展 Hugging Face 生成阶段 PPL 统计逻辑。
  - `compute_generation_ppl_stats()` 继续返回样本级 `generation_logprob_sum`、`generation_token_count`、`generation_nll` 和 `generation_ppl`，并可在 `include_token_details=True` 时额外返回每个生成 token 的 `position`、`token_id`、`token`、`logprob`、`nll` 和 `ppl`。
  - `HuggingFaceModel` 新增 `log_generation_token_ppl` 开关；开启后自动启用样本级 PPL，并在 batch 和 attention 路径中返回 token 级明细。

- `RULER/scripts/pred/call_api.py`
  - 新增 `--log_generation_token_ppl` 参数，仅支持 `--server_type hf`。
  - 新增 `build_generation_token_record()`，把模型返回的 token 级 PPL 明细写入 sidecar jsonl 记录。
  - 开启该参数后，每个任务会额外写出 `<任务>.generation_tokens.jsonl`，主预测 jsonl 仍保持原有评分兼容结构。

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 runner 参数 `--log-generation-token-ppl`，透传到 `pred/call_api.py`。
  - 该 runner 参数会自动启用样本级 `--log-generation-ppl`。
  - `--overwrite-existing` 现在会一并删除旧的 `<任务>.generation_tokens.jsonl`。

- `tests/test_model_wrappers.py`
  - 新增 token 级 PPL 明细测试，验证每个 token 的 logprob、NLL、PPL 和解码文本。

- `tests/test_call_api_progress.py`
  - 新增 `--log_generation_token_ppl` 参数存在性检查。
  - 新增 sidecar 记录构造测试，验证 `index`、`task`、`generation_token_count` 和 `tokens` 字段。

- `tests/test_run_parquet_parallel.py`
  - 新增 runner 新参数透传测试。
  - 覆盖 `--log-generation-token-ppl` 会自动启用样本级 PPL。
  - 覆盖覆盖重跑时删除 token 级 PPL sidecar 文件。

- `AGENTS.md`
  - 补充 token 级 PPL 明细文件、参数、输出口径和覆盖重跑清理范围。

- `CODEX_CHANGES.md`
  - 记录本次 token 级 PPL 明细能力。

### 变更目的

本次变更用于让本地 RULER Hugging Face 预测流程在模型运行时保留每个生成 token 的困惑度明细。主预测 jsonl 继续只保存样本级聚合 PPL，避免影响 RULER 原生评分；token 级明细独立写到 sidecar 文件，方便后续按任务、样本和 token 分析模型生成行为。

### 主要函数和类

- `compute_generation_ppl_stats`
  - 可选返回 token 级明细；每个 token 的 `nll=-logprob`，`ppl=exp(nll)`。
- `compute_batch_generation_ppl_stats`
  - 将 token 级明细开关应用到 batch 中每条样本。
- `HuggingFaceModel`
  - 新增 `log_generation_token_ppl` 配置，保证 token 明细和样本级 PPL 使用同一批 `generate(..., output_scores=True)` scores。
- `build_generation_token_record`
  - 构造 `<任务>.generation_tokens.jsonl` 的单行记录。
- `overwrite_files_for`
  - 覆盖重跑时把旧 token 级 PPL sidecar 纳入删除范围。

### 运行方式

单任务 dry-run 检查命令是否透传 token 级 PPL 参数：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-token-ppl \
  --dry-run
```

正式重跑并记录每个生成 token 的 PPL 时，应使用 `--overwrite-existing` 避免旧预测文件或旧 sidecar 混入：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-token-ppl \
  --overwrite-existing
```

输出文件示例：

```text
RULER/benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred/niah_single_1.generation_tokens.jsonl
```

### 测试和验证

本次需要运行的验证命令：

```bash
conda run --no-capture-output -n model python -B -m unittest tests/test_model_wrappers.py
conda run --no-capture-output -n model python -B -m unittest tests/test_call_api_progress.py
conda run --no-capture-output -n model python -B -m unittest tests/test_run_parquet_parallel.py
conda run --no-capture-output -n model python -B -m unittest discover -s tests
conda run --no-capture-output -n model python -B -m py_compile \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/run_parquet_parallel.py
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-token-ppl \
  --dry-run
git diff --check
git -C RULER diff --check
```

### 假设和限制

- token 级 PPL 只支持 `--server_type hf`，因为它依赖 Hugging Face `generate(..., output_scores=True)` 返回的逐步 scores。
- 记录的是模型对自己生成答案 token 的困惑度，不是参考答案困惑度，也不是 prompt 困惑度。
- token 明细基于生成阶段 token ids 计算；如果后续 `stop_words` 裁剪了文本，sidecar 仍记录模型原始生成 token 的 PPL。
- 开启 token 级明细会显著增加输出文件体积；全量四模型多长度运行前应确认磁盘空间。

## 2026-05-21 将 RULER 统一汇总输出迁移为 CSV

### 变更文件

- `RULER/scripts/eval/collect_results.py`
  - 将统一汇总输出从单个 xlsx 工作簿改为一组 csv 文件。
  - 主输出文件保存 `detail` 明细表，其他表按同名前缀写出 `summary_by_model`、`summary_by_model_and_length`、`summary_by_task` 和 `run_info` csv。
  - 新增 `.csv` 后缀校验，并在命令行入口访问数据目录前提前执行，避免继续生成 `.xlsx` 汇总文件。

- `RULER/scripts/run_parquet_parallel.py`
  - 将自动汇总默认文件名改为 `ruler_results.csv`。
  - 更新 `--auto-evaluate` 和 `--report-file` 的命令行说明，使 runner workflow 指向 csv 汇总。
  - 在 runner 配置阶段拒绝非 `.csv` 的 `--report-file`，避免推理结束后才发现汇总路径格式错误。

- `tests/test_collect_results.py`
  - 将汇总输出测试从 xlsx zip/XML 检查改为 csv 文件内容检查。
  - 覆盖 `.xlsx` 输出路径应被拒绝的行为。
  - 覆盖 `.xlsx` 输出路径会先于数据目录发现被拒绝。

- `tests/test_run_parquet_parallel.py`
  - 覆盖 runner 默认自动汇总文件名为 `ruler_results.csv`。
  - 覆盖 runner 会提前拒绝 `.xlsx` 汇总路径。

- `AGENTS.md`
  - 将当前 RULER workflow、命令示例和输出说明更新为 csv 多文件汇总。
  - 保留旧不完整 `.xlsx` 产物的历史事实，并明确当前新 workflow 应使用 csv。

- `CODEX_CHANGES.md`
  - 记录本次 csv 迁移。

### 变更目的

本次变更用于满足 RULER workflow 最终汇总文件不再使用 `.xlsx` 格式的要求。新的汇总输出只使用 Python 标准库 `csv`，适合脚本读取，也避免依赖 `openpyxl`、`xlsxwriter` 或自定义 xlsx XML 写出逻辑。

### 主要函数和类

- `csv_file_for_sheet`
  - 根据主 csv 路径和表名生成对应的 sidecar csv 路径。
- `validate_csv_output_file`
  - 校验 `--output-file` 必须使用 `.csv` 后缀。
- `write_csv`
  - 将 `collect_results()` 返回的五张表写出为一组 csv 文件。
- `build_config`
  - 继续构造 runner 配置，但默认 `report_file` 现在指向 `ruler_results.csv`。

### 运行方式

手动生成统一 csv 汇总：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python eval/collect_results.py \
  --output-root ../benchmark_root/local_eval \
  --data-root ../benchmark_root/parquet_data/synthetic \
  --benchmark synthetic \
  --seq-lengths 4096 \
  --tasks all \
  --timing-file ../benchmark_root/local_eval/ruler_timing.jsonl \
  --output-file ../benchmark_root/local_eval/ruler_results_4k_all_models.csv
```

自动预测后汇总时继续使用 runner 的 `--auto-evaluate`，默认会写出 `output-root/ruler_results.csv` 和同名前缀的 summary/run_info csv 文件。

### 测试和验证

本次需要运行的验证命令：

```bash
conda run --no-capture-output -n model python -B -m unittest tests.test_collect_results tests.test_run_parquet_parallel
conda run --no-capture-output -n model python -B -m unittest discover -s tests
conda run --no-capture-output -n model python -B -m py_compile RULER/scripts/eval/collect_results.py RULER/scripts/run_parquet_parallel.py
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --auto-evaluate \
  --dry-run
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 旧的 `.xlsx` 汇总产物不会自动删除或迁移，后续新汇总应显式输出到 `.csv`。
- `--output-file` 和 `--report-file` 参数名保持不变，但参数值现在必须是 `.csv` 主输出路径。
- 每次汇总会写出五个 csv 文件；读取 summary 时需要使用同名前缀的 sidecar 文件。

## 2026-05-20 新增 RULER 20 条固定随机子数据集划分脚本

### 变更文件

- `tools/slice_ruler_dataset.py`
  - 新增 RULER jsonl 子数据集划分脚本。
  - 默认从 `RULER/benchmark_root/parquet_data/synthetic` 读取 4k 到 64k 的 13 个 synthetic 任务。
  - 默认每个任务长度用固定随机种子 `0` 抽取 20 条，写入 `RULER/benchmark_root/parquet_data_20/synthetic`。
  - 写出 `subset_report.json`，记录源文件、目标文件、源行数、写出行数、随机种子、抽中的原始行号和样本 `index`。

- `tests/test_slice_ruler_dataset.py`
  - 新增子数据集划分脚本单元测试。
  - 覆盖固定随机抽样、输出顺序、源文件不变、runner 兼容目录结构、划分报告和样本数不足报错。

- `CODEX_CHANGES.md`
  - 记录本次脚本、测试、运行方式和限制。

### 变更目的

本次变更用于在不改动原始 RULER 输入数据的前提下，快速复制出一个小样本评测集。新子数据集保留 runner 需要的目录结构，因此后续只需要把 `run_parquet_parallel.py` 的 `--data-root` 指向新目录即可用四个本地模型运行。

### 主要函数和类

- `parse_lengths`
  - 解析长度参数，默认 `all` 表示 `4096,8192,16384,32768,65536`。
- `parse_task_csv`
  - 解析任务参数，默认 `all` 表示 13 个 synthetic 任务。
- `slice_jsonl_file`
  - 从单个 jsonl 文件中固定随机抽样，并按原文件行顺序复制到目标文件。
- `slice_dataset`
  - 按长度和任务矩阵批量复制子数据集，并写出 `subset_report.json`。
- `main`
  - 命令行入口。

### 运行方式

生成默认 20 条子数据集：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python -B tools/slice_ruler_dataset.py
```

子数据集输出位置：

```text
/data/czy/ICLR-2027/RULER/benchmark_root/parquet_data_20/synthetic
```

使用四个模型运行该子数据集：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --model Qwen2.5-7B-Instruct-1M=../../models/Qwen2.5-7B-Instruct-1M \
  --model Yi-9B-200K=../../models/Yi-9B-200K \
  --model GLM-4-9B-Chat-1M=../../models/GLM-4-9B-Chat-1M \
  --seq-lengths 4096,8192,16384,32768,65536 \
  --tasks all \
  --gpus 0,2,3,5 \
  --max-workers 4 \
  --data-root ../benchmark_root/parquet_data_20/synthetic \
  --output-root ../benchmark_root/local_eval_20 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 10 \
  --log-batch-progress \
  --auto-evaluate \
  --report-file ../benchmark_root/local_eval_20/ruler_results_4k_64k_20samples_all_models.xlsx \
  --skip-existing
```

### 测试和验证

本次需要运行的验证命令：

```bash
conda run --no-capture-output -n model python -B -m unittest tests/test_slice_ruler_dataset.py
conda run --no-capture-output -n model python -B -m unittest discover -s tests
conda run --no-capture-output -n model python -B -m py_compile tools/slice_ruler_dataset.py RULER/scripts/run_parquet_parallel.py
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --data-root ../benchmark_root/parquet_data_20/synthetic \
  --output-root ../benchmark_root/local_eval_20 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --dry-run
```

### 假设和限制

- 默认固定随机种子为 `0`，同一源数据下重复运行会抽到同一批样本。
- 默认覆盖 4k、8k、16k、32k、64k，不包含 128k 及以上长度。
- 脚本只复制已有 jsonl 行，不重新生成 prompt，不读取原始 parquet。
- 运行四模型命令前仍需确认 GPU 和 CUDA 可用。

## 2026-05-20 更新当前 model 环境和 RULER 命令说明

### 变更文件

- `AGENTS.md`
  - 将当前环境状态从旧的 `base`/`model_download` 说明更新为当前主用 conda 环境 `model`。
  - 记录 `model` 环境 Python、`torch`、`transformers`、`pyarrow` 和 `pandas` 版本。
  - 记录当前 `nvidia-smi` 仍失败、`torch.cuda.is_available()` 为 `False`，正式 GPU 推理前仍需修复 GPU driver 或设备节点。
  - 记录 `RULER/benchmark_root/parquet_data/synthetic/` 已经存在，覆盖 117 个 `validation.jsonl` 输入文件。
  - 记录 `RULER/benchmark_root/local_eval/` 已经有四模型 4k 输出目录、日志、部分预测、timing 和 `ruler_results_4k_all_models.xlsx`，但结果不完整，复用前需要检查。
  - 将推荐命令统一改为 `conda run --no-capture-output -n model ...`。
  - 将四模型 4k runner 示例更新为 GPU `0,2,3,5`、`--max-workers 4`、`--auto-evaluate` 和 `ruler_results_4k_all_models.xlsx`。
  - 补充 `tools/count_ruler_samples.py` 和 `ruler_sample_counts_4k_64k.csv` 的说明和运行命令。

- `CODEX_CHANGES.md`
  - 记录本次 `AGENTS.md` 环境和命令更新。

### 变更目的

本次变更用于让协作说明和当前服务器实际状态一致，避免后续继续使用 `<ruler-env>`、`dl-a800`、`ruler-glm44` 或旧 GPU 列表命令。当前文档明确说明：脚本运行默认使用 `model` 环境，输入 jsonl 已经转换完成，但 GPU driver/CUDA 可用性仍是正式推理前的阻塞。

### 主要函数和类

本次只修改文档，没有新增或修改函数、类。

### 运行方式

当前四模型 4k 续跑和自动汇总命令已更新为：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --model Qwen2.5-7B-Instruct-1M=../../models/Qwen2.5-7B-Instruct-1M \
  --model Yi-9B-200K=../../models/Yi-9B-200K \
  --model GLM-4-9B-Chat-1M=../../models/GLM-4-9B-Chat-1M \
  --seq-lengths 4096 \
  --tasks all \
  --gpus 0,2,3,5 \
  --max-workers 4 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 10 \
  --log-batch-progress \
  --auto-evaluate \
  --report-file ../benchmark_root/local_eval/ruler_results_4k_all_models.xlsx \
  --skip-existing
```

### 测试和验证

本次环境和文档检查使用的命令：

```bash
hostname
conda env list
conda run --no-capture-output -n model python -c "import sys, torch, transformers, pyarrow, pandas, nltk, yaml; print(sys.executable); print(sys.version.split()[0]); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('transformers', transformers.__version__); print('pyarrow', pyarrow.__version__); print('pandas', pandas.__version__)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
find RULER/benchmark_root/parquet_data/synthetic -path '*/data/*/validation.jsonl' | wc -l
find RULER/benchmark_root/local_eval -path '*/synthetic/4096/pred/*.jsonl' | wc -l
find RULER/benchmark_root/local_eval -path '*/synthetic/4096/logs/*.log' | wc -l
```

### 假设和限制

- `model` 环境可以用于脚本检查、数据转换、统计和 dry-run；由于当前 CUDA 不可用，不能据此确认 GPU 推理已经可运行。
- 当前 `local_eval` 下已有的 4k 输出不完整，文档中不把它描述为最终有效测评结果。
- 旧环境名仅保留在“旧命令需要替换”的提醒中，不再作为推荐命令出现。

## 2026-05-20 新增 RULER 样本数统计脚本

### 变更文件

- `tools/count_ruler_samples.py`
  - 新增 RULER 转换后 jsonl 输入样本数统计脚本。
  - 默认统计 `4096`、`8192`、`16384`、`32768`、`65536` 五个长度。
  - 默认覆盖 13 个 synthetic 任务，并读取 `validation.jsonl`。
  - 支持 `table`、`csv` 和 `json` 三种输出格式。

- `tests/test_count_ruler_samples.py`
  - 新增样本数统计脚本的单元测试。
  - 覆盖长度解析、缺失文件标记、表格输出和 CSV 输出。

- `ruler_sample_counts_4k_64k.csv`
  - 新增一次当前工作区 4k 到 64k 样本数统计输出。
  - 每行对应一个任务，每列对应一个长度，最后一列为任务总样本数。

- `CODEX_CHANGES.md`
  - 记录本次脚本、测试和统计输出变更。

### 变更目的

本次变更用于快速查看当前 RULER 4k 到 64k 转换后输入数据中，每个任务在每个长度下有多少条测试样本。脚本读取 runner 实际使用的 jsonl 输入目录，避免重新读取原始 parquet 或依赖 `pyarrow`。

### 主要函数和类

- `parse_length_value`
  - 将 `4k`、`64k`、`4096` 等长度写法转换为 token 数。
- `parse_lengths`
  - 解析逗号分隔长度列表。
- `parse_tasks`
  - 解析任务列表，支持 `all`。
- `collect_counts`
  - 收集每个任务在各长度上的 jsonl 非空行数。
- `format_table`
  - 输出终端可读表格。
- `format_csv`
  - 输出 CSV 文本。
- `format_json`
  - 输出 JSON 文本。
- `main`
  - 命令行入口。

### 运行方式

默认以终端表格展示当前 4k 到 64k 样本数：

```bash
cd /data/czy/ICLR-2027
python -B tools/count_ruler_samples.py
```

生成 CSV 文件：

```bash
cd /data/czy/ICLR-2027
python -B tools/count_ruler_samples.py --format csv > ruler_sample_counts_4k_64k.csv
```

指定长度、任务或输出格式：

```bash
python -B tools/count_ruler_samples.py \
  --lengths 4k,8k,64k \
  --tasks niah_single_1,qa_2 \
  --format json
```

### 测试和验证

本次运行的验证命令：

```bash
python -B -m unittest tests.test_count_ruler_samples
conda run --no-capture-output -n model python -B -m unittest tests.test_count_ruler_samples
python -B -m py_compile tools/count_ruler_samples.py
python -B tools/count_ruler_samples.py
python -B tools/count_ruler_samples.py --format json --tasks niah_single_1 --lengths 4k,64k
python -B tools/count_ruler_samples.py --format csv > ruler_sample_counts_4k_64k.csv
git diff --check
git -C RULER diff --check
git status --short
git -C RULER status --short
```

新增单测在 `base` 和 `model` 环境中均通过。当前 `base` 环境运行全量测试：

```bash
python -B -m unittest discover -s tests
```

会因环境缺少 `numpy`、`yaml` 和 `torch` 在既有测试上失败，与本次新增脚本无关。

### 假设和限制

- 脚本统计的是 `RULER/benchmark_root/parquet_data/synthetic/` 下转换后的 jsonl 输入，不直接统计原始 parquet。
- 缺失的 jsonl 文件在输出中显示为 `MISSING`，不会被当成 0。
- 样本数按非空行统计，不校验每行 JSON 内容是否合法。

## 2026-05-19 记录新服务器状态并删除旧文档入口

### 变更文件

- `AGENTS.md`
  - 按当前新服务器状态重写协作说明。
  - 将工作区路径更新为 `/data/czy/ICLR-2027`。
  - 记录旧的 `dl-a800` 和 `ruler-glm44` conda 环境当前不可见。
  - 记录当前可见环境为 `base` 和 `model_download`，且当前都缺少 `torch` 等 RULER 推理依赖。
  - 记录当前 `nvidia-smi` 无法和 NVIDIA driver 通信，GPU 状态需要重新确认。
  - 将模型目录更新为 `Llama-3.1-8B`、`Qwen2.5-7B-Instruct-1M`、`GLM-4-9B-Chat-1M` 和 `Yi-9B-200K`。
  - 记录 `RULER/benchmark_root/` 当前不存在，转换后的 jsonl、预测输出、评分输出和 timing 文件都需要重新生成。
  - 将命令示例改为新路径和占位环境 `<ruler-env>`，避免继续使用旧服务器路径和旧环境名。

- `README.md`
  - 按用户要求删除根目录 README，后续由用户重新整理。

- `RULER/docker/Dockerfile`
  - 按用户要求删除旧 Docker 模板。

- `RULER/docker/requirements.txt`
  - 按用户要求删除旧 Docker 依赖清单。

- `CODEX_CHANGES.md`
  - 记录本次环境梳理、文档更新和删除操作。

### 变更目的

本次变更用于把仓库说明从旧服务器迁移到当前新服务器状态，避免后续继续复制不可用的 `dl-a800`、`ruler-glm44`、`/home/test05/czyprojects` 和旧模型目录命令。同时删除不再需要的根目录 README 和 RULER 旧 Docker 模板。

### 主要函数和类

本次只修改文档和删除文档/模板文件，没有新增或修改函数、类。

### 运行方式

当前新服务器尚未配置完整 RULER 推理环境。正式运行前需要先创建或指定新的 `<ruler-env>`，安装 `torch`、`transformers`、`pyarrow`、`pandas`、`nltk` 等依赖，并确认 GPU driver 和 CUDA 可用。

配置完成后，基础检查命令为：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
conda run -n <ruler-env> python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

重新生成 RULER jsonl 输入：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python RULER/scripts/data/prepare_parquet.py
```

runner dry-run 示例：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run -n <ruler-env> python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --dry-run
```

### 测试和验证

文档和文件删除变更，验证命令：

```bash
git diff --check
git -C RULER diff --check
```

本次还做了只读环境检查：

```bash
git status --short
git -C RULER status --short
find models -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
conda env list
python --version
conda run -n model_download python --version
python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
conda run -n model_download python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
find benchmark/RULER-llama3-1M -mindepth 1 -maxdepth 1 -type d ! -name '.cache' | wc -l
find benchmark/RULER-llama3-1M -name 'validation-*.parquet' | wc -l
find RULER -path '*benchmark_root*' -print
find . -name '*.jsonl' | wc -l
```

检查结论：

- 当前可见模型目录为 `GLM-4-9B-Chat-1M`、`Llama-3.1-8B`、`Qwen2.5-7B-Instruct-1M`、`Yi-9B-200K`。
- 当前 conda 环境只有 `base` 和 `model_download`。
- `base` 与 `model_download` 均缺少 `torch`。
- 当前 `nvidia-smi` 无法和 NVIDIA driver 通信。
- 当前没有 `RULER/benchmark_root/`，仓库内 `.jsonl` 数量为 0。
- 原始 parquet benchmark 仍存在，117 个任务长度目录，118 个 parquet 分片。

### 假设和限制

- “删除 README.md”按根目录 `README.md` 理解；`RULER/README.md` 作为上游 RULER 说明保留。
- 当前没有尝试安装依赖、下载模型或运行推理，避免在未确认环境策略前改变服务器环境。
- `RULER/docker/` 删除后，后续若需要容器化，应重新按当前模型和依赖版本设计 Dockerfile，而不是恢复旧模板。
- 由于当前 RULER 运行环境未配置，本次没有运行 Python 单元测试或 py_compile。

## 2026-05-13 新增根目录 README 使用说明

### 变更文件

- `README.md`
  - 新增根目录项目说明文档。
  - 说明根目录、`RULER/`、`models/`、`benchmark/`、`attention_dumps/`、`tools/` 和 `tests/` 的用途。
  - 写入用户给定的四模型 4k 覆盖重跑自动测评命令。
  - 写入用户给定的 `tools/inspect_attention_dump.py` attention 查看命令。
  - 补充 `RULER/scripts/eval/collect_results.py` 单独评测命令，说明如何在已有预测后手动生成 `ruler_4k_results.xlsx`。
  - 补充 RULER 原生 `eval/evaluate.py` 单目录评分命令。
  - 解释主要命令参数、输出文件、xlsx sheet、attention dump 文件和常见检查命令。

- `CODEX_CHANGES.md`
  - 记录本次 README 新增和内容补充。

### 变更目的

本次变更用于把当前本地 RULER 测评工作区整理成一个可直接阅读的入口文档。后续使用者可以从 `README.md` 了解每个目录和关键文件的作用，复制四模型覆盖重跑命令、attention 查看命令，或在预测完成后单独调用评测汇总脚本。

### 主要函数和类

本次只新增文档，没有新增或修改函数、类。

### 运行方式

四模型 4k 覆盖重跑并自动测评的命令已写入 `README.md`。单独生成统一测评表的命令也已写入：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/collect_results.py \
  --output-root /home/test05/czyprojects/RULER/benchmark_root/local_eval \
  --data-root /home/test05/czyprojects/RULER/benchmark_root/parquet_data/synthetic \
  --benchmark synthetic \
  --seq-lengths 4096 \
  --tasks all \
  --timing-file /home/test05/czyprojects/RULER/benchmark_root/local_eval/ruler_4k_timing.jsonl \
  --output-file /home/test05/czyprojects/RULER/benchmark_root/local_eval/ruler_4k_results.xlsx
```

### 测试和验证

文档-only 变更，验证命令：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- README 中的四模型命令会覆盖对应任务的旧预测和日志。
- 用户给定的四模型命令不包含 `--log-generation-ppl`；README 中已注明如果需要在预测 jsonl 和 xlsx 中包含 PPL，需要额外加入该参数。
- 单独评测命令只读取已有预测结果，不重新加载模型，不重新推理。

## 2026-05-13 覆盖重跑和自动测评文档补充

### 变更文件

- `AGENTS.md`
  - 在正式运行预测流程中新增“覆盖重跑和自动测评汇总”小节。
  - 补充 `--overwrite-existing`、`--log-generation-ppl` 和 `--auto-evaluate` 的组合命令。
  - 明确 `--overwrite-existing` 和 `--skip-existing` 互斥，且覆盖重跑只删除当前任务预测、attention 摘要和日志。
  - 明确 `--auto-evaluate` 会调用 `RULER/scripts/eval/collect_results.py`，默认输出 `RULER/benchmark_root/local_eval/ruler_results.xlsx`。
  - 将“运行评分”标题改为“运行评分和统一测评汇总”，让 RULER 原生评分与统一 xlsx 汇总都在同一节中查找。

- `CODEX_CHANGES.md`
  - 新增本次文档补充记录。

### 变更目的

本次变更用于把“覆盖已有预测重新生成”和“跑完后自动测评汇总”两个新流程集中写入协作说明，避免只在参数列表中分散出现。后续维护者可以直接复制命令完成覆盖重跑、生成阶段 PPL 记录和统一 xlsx 测评表生成。

### 主要函数和类

本次只修改文档，没有新增或修改函数、类。

### 运行方式

覆盖已有预测并自动测评：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --log-generation-ppl \
  --overwrite-existing \
  --auto-evaluate
```

### 测试和验证

文档-only 变更，验证命令：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次只补充说明，不改变 runner、预测或汇总代码。
- 统一测评表仍由 `RULER/scripts/eval/collect_results.py` 生成。
- 覆盖重跑仍不删除 `summary.csv`、`submission.csv` 或 `ruler_results.xlsx`。

## 2026-05-13 attention 查看工具支持按权重排序

### 变更文件

- `tools/inspect_attention_dump.py`
  - 新增 `sort_attention_rows()`，可按 `position` 或 `attention` 排序，并支持降序。
  - 新增命令行参数 `--sort-by` 和 `--descending`。
  - 默认仍按 token position 输出，保持旧命令行为不变。

- `tests/test_attention_dump_tools.py`
  - 新增按 attention 降序排序的测试。
  - 验证排序不会原地修改原始 rows。

- `AGENTS.md`
  - 补充按 attention 从大到小查看表格的命令示例。

- `CODEX_CHANGES.md`
  - 记录本次功能和文档变更。

### 变更目的

本次变更用于让 `tools/inspect_attention_dump.py` 在保留原有表格格式的基础上，支持把 attention 权重从大到小输出，便于快速找到指定生成 token、层和 head 最关注的上下文 token。

### 主要函数和类

- `sort_attention_rows()`
  - 根据 `sort_by` 和 `descending` 返回新的排序列表。
  - `sort_by="position"` 保持位置顺序。
  - `sort_by="attention"` 按 attention 权重排序。

- `build_parser()`
  - 增加 `--sort-by` 和 `--descending` 参数。

- `main()`
  - 在计算 `attention_sum` 后对表格行排序，再调用 `format_table()` 输出。

### 运行方式

按 attention 从大到小查看第 0 个生成 token、第 0 层、第 0 个 head：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0 \
  --sort-by attention \
  --descending
```

不传 `--sort-by attention --descending` 时，仍然按 position 从小到大输出。

### 测试和验证

先确认新增测试会失败：

```bash
conda run -n dl-a800 python -B -m unittest tests.test_attention_dump_tools.AttentionDumpToolsTest.test_inspector_can_sort_rows_by_attention_descending
```

实现后验证：

```bash
conda run -n dl-a800 python -B -m unittest tests.test_attention_dump_tools.AttentionDumpToolsTest.test_inspector_can_sort_rows_by_attention_descending
conda run -n dl-a800 python -B -m unittest tests.test_attention_dump_tools
PYTHONPYCACHEPREFIX=/tmp/codex-pycache conda run -n dl-a800 python -m py_compile tools/inspect_attention_dump.py
conda run -n dl-a800 python tools/inspect_attention_dump.py --help
conda run -n dl-a800 python tools/inspect_attention_dump.py --dump-dir attention_dumps/llama_niah_single_1_sample_0 --generated-token 0 --layer 0 --head 0 --sort-by attention --descending
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次只改变查看工具的输出顺序，不重新生成 attention dump。
- `attention_sum` 仍基于完整 rows 计算，不受输出排序影响。
- 排序只影响终端表格展示，不修改 `.npy`、`prompt_tokens.jsonl` 或 `generated_tokens.jsonl`。

## 2026-05-13 RULER 统一结果汇总和生成阶段 PPL

### 变更文件

- `RULER/scripts/pred/model_wrappers.py`
  - 新增生成阶段 PPL 统计辅助函数。
  - `HuggingFaceModel` 支持 `log_generation_ppl`，开启后基于 `generate(..., output_scores=True)` 的 token scores 计算生成答案 token 的 logprob、NLL 和 PPL。

- `RULER/scripts/pred/call_api.py`
  - 新增 `--log_generation_ppl` 参数，仅支持 `--server_type hf`。
  - 新增 `build_prediction_record()`，写预测 jsonl 时保留 `generation_logprob_sum`、`generation_token_count`、`generation_nll` 和 `generation_ppl`。

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 `--log-generation-ppl`，透传给 `call_api.py`。
  - 新增 `--timing-file`、`--auto-evaluate` 和 `--report-file`。
  - 每个子任务结束后追加写入结构化 timing jsonl，记录模型、长度、任务、GPU、开始时间、结束时间、耗时、预测行数、样本数和退出码。
  - `--auto-evaluate` 会在调度结束后调用 `RULER/scripts/eval/collect_results.py` 生成统一 xlsx。

- `RULER/scripts/eval/collect_results.py`
  - 新增统一汇总脚本。
  - 读取预测 jsonl、输入 jsonl、runner timing jsonl 和 RULER synthetic metric，输出 `ruler_results.xlsx`。
  - xlsx 包含 `detail`、`summary_by_model`、`summary_by_model_and_length`、`summary_by_task` 和 `run_info`。
  - 使用 Python 标准库 `zipfile` 和 XML 生成最小 xlsx，不依赖 `openpyxl` 或 `xlsxwriter`。

- `tests/test_model_wrappers.py`
  - 新增生成阶段 PPL 纯函数测试和空生成测试。

- `tests/test_call_api_progress.py`
  - 新增 `--log_generation_ppl` 参数存在性检查。
  - 新增预测记录保留 PPL 字段测试。

- `tests/test_run_parquet_parallel.py`
  - 新增 runner PPL 参数透传、timing jsonl 写入和自动汇总命令构造测试。

- `tests/test_collect_results.py`
  - 新增统一汇总脚本测试，覆盖明细唯一键、缺失状态、模型级时间汇总、`summary_by_model_and_length` sheet、PPL 聚合、GPU 和文件路径字段。

- `tests/__init__.py`
  - 新增空包标记，保证 `python -m unittest tests.test_*` 形式稳定导入。

- `AGENTS.md`
  - 补充生成阶段 PPL、统一汇总脚本、runner 新参数、xlsx sheet 和验证命令说明。

- `CODEX_CHANGES.md`
  - 记录本次变更。

### 变更目的

本次变更用于让 RULER 本地评测在预测阶段记录生成答案 token 的 PPL，并在评测结束后或手动执行时，把四个模型、多个长度和多个任务的准确率、PPL、耗时与完成状态统一汇总到一个 xlsx 文件中。明细表中每一行唯一对应 `model + length + task`。

### 主要函数和类

- `compute_generation_ppl_stats()`
  - 基于单条样本的生成 scores 和实际生成 token ids 计算 `generation_logprob_sum`、`generation_token_count`、`generation_nll` 和 `generation_ppl`。

- `compute_batch_generation_ppl_stats()`
  - 对 batch 中每条样本分别调用生成阶段 PPL 统计逻辑。

- `build_prediction_record()`
  - 把模型返回结果转换为预测 jsonl 记录，并保留可选生成阶段 PPL 字段。

- `write_timing_record()`
  - 追加写入单个 runner 子任务的结构化耗时记录。

- `build_collect_results_command()` 和 `run_collect_results()`
  - 构造并执行统一汇总脚本命令。

- `collect_results()`
  - 生成按 sheet 分组的汇总数据。

- `write_xlsx()`
  - 使用标准库写出包含多个 sheet 的 xlsx 文件。

### 运行方式

预测时同时记录生成阶段 PPL，并在结束后自动汇总：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-generation-ppl \
  --auto-evaluate
```

手动生成统一汇总：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/collect_results.py \
  --output-root ../benchmark_root/local_eval \
  --data-root ../benchmark_root/parquet_data/synthetic \
  --benchmark synthetic \
  --seq-lengths all \
  --tasks all \
  --timing-file ../benchmark_root/local_eval/ruler_timing.jsonl \
  --output-file ../benchmark_root/local_eval/ruler_results.xlsx
```

### 测试和验证

```bash
conda run -n dl-a800 python -B -m unittest tests.test_model_wrappers tests.test_call_api_progress
conda run -n dl-a800 python -B -m unittest tests.test_collect_results
conda run -n dl-a800 python -B -m unittest tests.test_run_parquet_parallel
conda run -n dl-a800 python -B -m unittest discover -s tests
conda run -n dl-a800 python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py
git diff --check
git -C RULER diff --check
```

### 假设和限制

- PPL 口径是模型对自己生成答案 token 的困惑度，不是参考答案困惑度，也不是 prompt 困惑度。
- 当前 `--log_generation_ppl` 只支持 `--server_type hf`。
- PPL 统计基于生成阶段返回的 token scores，对 stop words 后处理裁剪出的文本不重新计算。
- `collect_results.py` 只内置 synthetic 的 13 个任务顺序；其他 benchmark 需要后续扩展。
- xlsx 是标准库生成的最小工作簿，适合 Excel/LibreOffice 打开和脚本检查，但不包含格式化样式。

## 2026-05-13 runner 覆盖重跑已有预测

### 变更文件

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 `--overwrite-existing` 参数。
  - 覆盖重跑时，在启动单个任务前删除对应任务已有的预测 jsonl、attention 摘要文件和日志文件。
  - 新增互斥校验：`--overwrite-existing` 不能和 `--skip-existing` 同时使用。

- `tests/test_run_parquet_parallel.py`
  - 新增 `--skip-existing` 和 `--overwrite-existing` 互斥测试。
  - 新增覆盖重跑清理旧预测、旧日志和 attention 输出的测试。
  - 确认无关任务预测文件不会被误删。

- `AGENTS.md`
  - 在 runner 参数说明中补充 `--overwrite-existing`。
  - 说明已有结果需要从头重跑时应使用 `--overwrite-existing`，不要使用 `--skip-existing`。

- `CODEX_CHANGES.md`
  - 记录本次覆盖重跑参数、测试和文档变更。

### 变更目的

本次变更用于解决已有预测文件存在时，`pred/call_api.py` 只会按已有 `index` 断点续跑、不会覆盖重跑的问题。现在可以通过 runner 的 `--overwrite-existing` 明确删除旧结果后从头生成。

### 主要函数和类

- `RunnerConfig`
  - 新增 `overwrite_existing` 字段。

- `overwrite_files_for()`
  - 返回覆盖重跑时需要清理的预测、attention 和日志路径。

- `delete_existing_outputs()`
  - 删除单个任务旧输出，并返回实际删除的路径。

- `launch_job()`
  - 在真实启动任务前执行覆盖清理；`--dry-run` 不会删除文件。

### 运行方式

重跑已经生成过的任务时使用：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --overwrite-existing
```

### 测试和验证

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_run_parquet_parallel.py'
conda run -n dl-a800 python -B -m py_compile RULER/scripts/run_parquet_parallel.py
git diff --check
git -C RULER diff --check
```

### 假设和限制

- `--overwrite-existing` 只删除当前任务对应的预测、attention 和日志，不删除 `summary.csv` 或 `submission.csv`。
- 重跑预测后，需要重新运行 `RULER/scripts/eval/evaluate.py` 才能更新评分汇总。
- 该参数由 runner 在启动任务前清理旧文件，不修改 `RULER/scripts/pred/call_api.py` 的断点续跑逻辑。

## 2026-05-13 Llama 单样本 attention 工具文档同步

### 变更文件

- `AGENTS.md`
  - 新增 `tools/` 顶层目录说明，记录 `dump_llama_attention.py` 和 `inspect_attention_dump.py` 的用途。
  - 在推荐流程中新增“Llama 单样本完整 attention 导出”小节。
  - 补充导出命令、查看命令、输出目录结构、`.npy` shape 含义和磁盘空间限制。

- `CODEX_CHANGES.md`
  - 新增本次文档同步记录。
  - 将原有 “Llama 单样本完整 Attention 导出工具” 记录补充为包含 `AGENTS.md` 文档说明。

### 变更目的

本次变更把 Llama 单样本完整 attention 工具写入仓库协作规范，方便后续会话或维护者直接从 `AGENTS.md` 找到运行命令、输出位置和查看方式。

### 主要函数和类

本次只修改文档，没有新增或修改函数、类。

### 运行方式

导出一个 Llama 样本的完整 attention：

```bash
cd /home/test05/czyprojects
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u tools/dump_llama_attention.py \
  --model-path models/Meta-Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --output-dir attention_dumps/llama_niah_single_1_sample_0 \
  --dtype float32 \
  --max-new-tokens 128 \
  --overwrite
```

查看一个生成 token 的指定层和 head：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0
```

### 测试和验证

文档-only 变更，验证命令：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次只同步文档，没有运行真实 Llama 推理。
- `tools/dump_llama_attention.py` 仍主要面向本地 `Meta-Llama-3.1-8B` 单样本诊断。
- 完整 attention 文件可能较大，长上下文或较多生成 token 时需要提前确认磁盘空间。

## 2026-05-13 coding 前默认使用子代理说明

### 变更文件

- `AGENTS.md`
  - 在 `Agent 专用说明` 中新增规则：以后每次开始 coding 前，默认先判断并使用子代理做可并行的检查、定位、测试设计或小范围实现。
  - 说明可以不使用子代理的例外情况：任务非常小、用户明确要求不要使用子代理，或当前环境没有可用子代理能力。

- `CODEX_CHANGES.md`
  - 记录本次文档变更。

### 变更目的

本次变更用于把“coding 前默认使用子代理”的协作偏好写入仓库规范，方便后续会话延续同样的工作方式。

### 主要函数和类

本次只修改文档，没有新增或修改函数、类。

### 运行方式

本次没有新增可运行代码。

### 测试和验证

文档-only 变更，验证命令：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 子代理使用仍需要遵守当前运行环境和工具权限。
- 如果用户明确要求不要使用子代理，用户指令优先。

## 2026-05-13 Llama 单样本完整 Attention 导出工具

### 变更文件

- `tools/dump_llama_attention.py`
  - 新增独立诊断脚本，不修改现有 RULER 推理脚本。
  - 读取一个 RULER jsonl 样本，使用本地 `Meta-Llama-3.1-8B` 生成回答。
  - 使用 eager attention 和 KV cache replay 逐个保存每个生成 token 的完整 attention。
  - 每个 `token_XXXX.npy` 的 shape 为 `[num_layers, num_heads, key_length]`，其中每个 `[layer, head, :]` 的和应接近 1。

- `tools/inspect_attention_dump.py`
  - 新增查看工具，可打印某个 `generated-token/layer/head` 的完整 attention 表格。
  - 表格列为 `position | source | token_id | token_text | attention`。

- `tests/test_attention_dump_tools.py`
  - 新增导出元数据、token jsonl、summary、attention shape/sum 和查看表格测试。

- `AGENTS.md`
  - 新增 `tools/` 目录说明和 Llama 单样本完整 attention 导出流程。
  - 记录导出命令、查看命令、输出文件结构和 `.npy` shape 含义。

- `CODEX_CHANGES.md`
  - 记录本次独立 attention dump 工具的变更、运行方式、验证方式和限制。

### 变更目的

本次变更用于满足“只简单了解一个 Llama 模型在一个任务里的完整 attention”的需求。新工具不接入 `call_api.py` 或 `run_parquet_parallel.py`，只作为一次性诊断脚本使用；运行时通过 `CUDA_VISIBLE_DEVICES=0` 保证一张卡只加载一个模型。

### 主要函数和类

- `dump_generated_attention()`
  - 先用 prompt 建 KV cache，再逐个 replay 生成 token，并保存每一步的完整 attention。

- `stack_step_attentions()`
  - 将模型返回的逐层 attention 堆叠为 `[layer, head, key_position]`。

- `summarize_attention_array()`
  - 记录 attention shape 和每个 `[layer, head]` 的归一化误差。

- `build_attention_rows()`
  - 将某个 `.npy` 文件中的一条 attention 向量与 prompt/generated token 文本对齐。

- `format_table()`
  - 输出固定列宽的完整 attention 表格。

### 运行方式

导出一个样本的完整 attention：

```bash
cd /home/test05/czyprojects
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u tools/dump_llama_attention.py \
  --model-path models/Meta-Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --output-dir attention_dumps/llama_niah_single_1_sample_0 \
  --dtype float32 \
  --max-new-tokens 128 \
  --overwrite
```

查看第 0 个生成 token、第 0 层、第 0 个 head 的完整分布：

```bash
conda run -n dl-a800 python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0
```

输出目录结构：

```text
attention_dumps/llama_niah_single_1_sample_0/
  metadata.json
  prompt_tokens.jsonl
  generated_tokens.jsonl
  token_0000.npy
  token_0001.npy
  ...
  summary.md
```

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/codex-pycache conda run -n dl-a800 python -m py_compile tools/dump_llama_attention.py tools/inspect_attention_dump.py
conda run -n dl-a800 python tools/dump_llama_attention.py --help
conda run -n dl-a800 python tools/inspect_attention_dump.py --help
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本工具只针对本地 Llama 诊断场景设计，不保证 GLM、Qwen、Yi 的特殊模型代码都能直接复用。
- 完整 attention 文件可能很大；4k prompt 下每个生成 token 的 float32 attention 约十几 MB。
- `token_XXXX.npy` 表示第 `XXXX` 个生成 token 作为 query 时，对 prompt 和已生成上下文的 attention，而不是修改 RULER benchmark 预测结果。
- 本次没有运行真实模型推理，只验证了工具的数据格式和查看逻辑。

## 2026-05-13 runner 按模型指定 Python 环境

### 变更文件

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 `--model-python NAME=PYTHON` 参数，可重复传入多个模型别名到 Python 解释器的映射。
  - 构造 `pred/call_api.py` 子进程命令时，优先使用当前模型别名对应的专属 Python；未配置的模型继续使用全局 `--python`。
  - 增加模型别名校验：如果 `--model-python` 引用的别名没有出现在 `--model` 中，则抛出 `ValueError`。

- `tests/test_run_parquet_parallel.py`
  - 新增 GLM 专属 Python 覆盖测试，确认 `glm-4-9b` 使用 `/home/test05/miniconda3/envs/ruler-glm44/bin/python`。
  - 新增默认 Python 回退测试，确认其他模型继续使用全局 `--python`。
  - 新增未知模型别名报错测试，避免配置拼写错误被静默忽略。

- `AGENTS.md`
  - 在 `RULER/scripts/run_parquet_parallel.py` 参数说明中补充 `--model-python NAME=PYTHON`，说明它可重复传入，并且只覆盖对应模型别名的 `pred/call_api.py` 子进程 Python。
  - 更新四模型 4k 全任务运行示例，显式使用 `/home/test05/miniconda3/envs/dl-a800/bin/python` 作为全局默认 Python，并为 `glm-4-9b` 指定 `/home/test05/miniconda3/envs/ruler-glm44/bin/python`。
  - 补充当前环境约定：非 GLM 三个模型使用 `dl-a800`，GLM 使用 `ruler-glm44`；runner 可由 `dl-a800` 启动，GLM 子进程使用 `ruler-glm44` 的 Python。

- `CODEX_CHANGES.md`
  - 新增本记录，说明 runner、测试和文档同步变更。

### 变更目的

本次变更用于让一个 `run_parquet_parallel.py` 进程同时调度不同 conda 环境中的模型。全局 `--python` 仍作为默认 Python，`--model-python` 可为特定模型别名覆盖子进程 Python。这样四模型并行测评时，Meta-Llama、Qwen、Yi 继续使用 `dl-a800`，GLM 单独使用 `ruler-glm44`。

### 主要函数和类

- `RunnerConfig`
  - 新增 `model_python` 字段，保存模型别名到 Python 解释器的映射。

- `parse_model_python_specs()`
  - 解析 `--model-python NAME=PYTHON` 参数，并校验模型别名必须来自 `--model`。

- `build_call_api_command()`
  - 根据当前 `Job` 的模型别名选择专属 Python 或默认 `config.python`。

- `build_parser()`
  - 新增 `--model-python` 命令行参数。

### 运行方式

四模型示例：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --model Qwen3-8B=../../models/Qwen3-8B \
  --model Yi-9B-200K=../../models/Yi-9B-200K \
  --model glm-4-9b=../../models/glm-4-9b \
  --python /home/test05/miniconda3/envs/dl-a800/bin/python \
  --model-python glm-4-9b=/home/test05/miniconda3/envs/ruler-glm44/bin/python \
  --seq-lengths 4096 \
  --tasks all \
  --gpus 0,1,2,3,4,5,6,7 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 10 \
  --log-batch-progress \
  --skip-existing
```

### 测试和验证

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_run_parquet_parallel.py'
conda run -n dl-a800 python -B -m py_compile RULER/scripts/run_parquet_parallel.py
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 当前文档假设 GLM 模型别名为 `glm-4-9b`，且 GLM 环境 Python 位于 `/home/test05/miniconda3/envs/ruler-glm44/bin/python`。
- `--model-python` 只选择启动子进程的 Python，不自动创建或激活 conda 环境；对应 Python 路径必须已经存在且依赖完整。
- 如果后续改用不同 GLM 别名，需要同步修改 `--model-python` 左侧的别名。

## 2026-05-13 Hugging Face 注意力摘要输出

### 变更文件

- `RULER/scripts/pred/model_wrappers.py`
  - 新增 `summarize_attention_layers()`，把模型返回的逐层 attention 张量压缩成每层 Top-K token 分数。
  - 新增注意力调试模式：开启后 Hugging Face wrapper 不走 pipeline，而是直接加载模型并使用 `attn_implementation="eager"`，保证 `output_attentions=True` 有机会返回注意力权重。
  - 对每条样本记录首个生成 token 对上下文 token 的逐层注意力分布，避免直接保存 prefill 阶段完整 `seq_len x seq_len` 注意力矩阵。

- `RULER/scripts/pred/call_api.py`
  - 新增 `--log_attention_scores` 和 `--attention_top_k` 参数。
  - 开启后在预测目录写入 `<task>.attention.md` 和 `<task>.attention.jsonl`。
  - Markdown 文件使用表格展示每层归一化和、Top token 位置、分数和 token 文本，便于直接查看。

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 `--log-attention-scores` 和 `--attention-top-k` 参数，并透传给 `pred/call_api.py`。

- `tests/test_model_wrappers.py`
  - 增加注意力层摘要排序、token 解码和归一化和测试。

- `tests/test_call_api_progress.py`
  - 增加注意力参数和 Markdown 输出格式测试。

- `tests/test_run_parquet_parallel.py`
  - 增加 runner 透传注意力参数的测试。

- `CODEX_CHANGES.md`
  - 记录本次注意力摘要功能、运行方式、验证方式和限制。

### 变更目的

本次变更用于在本地 Hugging Face 模型运行 RULER 任务时，按样本输出逐层注意力权重摘要。默认不启用，因此普通 benchmark 推理路径不变；只有显式传入注意力开关时才会额外计算并写出注意力文件。

### 主要函数和类

- `summarize_attention_layers()`
  - 对每层 attention 在 head 维度取平均，重新归一化后保留 Top-K token。

- `decode_attention_token()`
  - 将 token id 解码成适合写入 Markdown 表格的短文本。

- `HuggingFaceModel._process_one_with_attention()`
  - 单条样本生成预测，并附带注意力摘要。

- `HuggingFaceModel._summarize_first_generated_token_attention()`
  - 先用 prompt 建 KV cache，再只对首个生成 token 做一次 `output_attentions=True` 的前向计算，降低注意力调试的显存开销。

- `format_attention_markdown()`
  - 将注意力摘要格式化为 Markdown 表格。

### 运行方式

单任务直接查看注意力：

```bash
cd /home/test05/czyprojects/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u pred/call_api.py \
  --data_dir ../benchmark_root/parquet_data/synthetic/4096/data \
  --save_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic \
  --task niah_single_1 \
  --server_type hf \
  --model_name_or_path ../../models/Meta-Llama-3.1-8B \
  --temperature 0.0 \
  --top_k 32 \
  --top_p 1.0 \
  --batch_size 1 \
  --log_attention_scores \
  --attention_top_k 8 \
  --log_batch_progress
```

并行 runner 中开启注意力摘要：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run --no-capture-output -n dl-a800 python -u run_parquet_parallel.py \
  --model llama=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-attention-scores \
  --attention-top-k 8 \
  --log-batch-progress
```

输出文件：

- `RULER/benchmark_root/local_eval/<模型名>/synthetic/<长度>/pred/<任务>.attention.md`
- `RULER/benchmark_root/local_eval/<模型名>/synthetic/<长度>/pred/<任务>.attention.jsonl`

如果需要看更多 token 的分数，可以调大 `--attention_top_k` 或 `--attention-top-k`；值越大，输出越长。

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/codex-pycache conda run -n dl-a800 python -m py_compile RULER/scripts/pred/model_wrappers.py RULER/scripts/pred/call_api.py RULER/scripts/run_parquet_parallel.py
cd /home/test05/czyprojects/RULER/scripts && conda run -n dl-a800 python -B run_parquet_parallel.py --model llama=../../models/Meta-Llama-3.1-8B --seq-lengths 4096 --tasks niah_single_1 --gpus 0 --server-type hf --batch-size 1 --log-attention-scores --attention-top-k 8 --log-batch-progress --dry-run
git diff --check
git -C /home/test05/czyprojects/RULER diff --check
```

### 假设和限制

- 注意力摘要目前只支持 `--server_type hf`，不支持 vLLM、OpenAI、Gemini、Mamba 等后端。
- 记录的是首个生成 token 的逐层注意力分布，不直接保存 prompt prefill 阶段完整注意力矩阵；后者是 `seq_len x seq_len`，在 4k、1M 长上下文下非常容易造成显存和磁盘爆炸。
- 开启注意力摘要后会强制使用 eager attention，并额外做一次前向计算，因此速度会明显慢于正常 benchmark。
- GLM 仍受当前 Transformers/cache 兼容问题影响；如需查看 GLM 注意力，优先使用单独的 GLM 兼容环境。

## 2026-05-13 AGENTS 当前文件状态校准

### 变更文件

- `AGENTS.md`
  - 将模型目录写成当前真实路径，例如 `models/Meta-Llama-3.1-8B/`，避免只有裸目录名造成误解。
  - 将 `call_api.py`、`run_parquet_parallel.py`、`prepare_parquet.py`、`evaluate.py` 等说明统一改成完整仓库相对路径。
  - 按当前文件系统补充已转换 jsonl 数量：9 个长度、每个 13 个任务、共 117 个 `validation.jsonl`。
  - 按当前 `local_eval/` 文件补充 4k 预测状态：三个模型各 13 个任务，`glm-4-9b` 有 3 个任务。
  - 明确当前未发现 `summary*.csv` 评分文件。
  - 修正 `__pycache__/` 和 `.pyc` 说明，从“不要保留”改为“不要提交；新产生时清理或确认被忽略”，避免和当前已有本地缓存文件冲突。

- `CODEX_CHANGES.md`
  - 记录本次按当前文件系统校准 `AGENTS.md` 的变更。

### 变更目的

本次变更用于把 `AGENTS.md` 中容易被理解为过期或不存在的裸文件名、裸目录名改成当前真实路径，并同步当前实际生成数据和预测结果状态。这样后续阅读文档时，可以直接按路径定位文件，不需要猜测某个脚本或目录是在顶层还是在 `RULER/scripts/` 下。

### 主要函数和类

本次只修改文档，没有新增或修改 Python 函数、类或运行时代码。

### 运行方式

本次没有新增运行入口。文档中保留的主入口仍是：

```bash
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --dry-run
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

### 验证方式

本次文档变更应运行：

```bash
git diff --check
git -C RULER diff --check
git status --short
git -C RULER status --short
```

### 假设和限制

- 本次没有修改执行代码，因此不需要重新跑模型推理或评分。
- 文档中的“当前状态”基于 2026-05-13 文件系统检查；如果后续删除模型、数据或预测结果，需要再次同步。
- 评分文件 `summary*.csv` 是 `RULER/scripts/eval/evaluate.py` 的预期输出；当前未发现并不表示评分脚本不存在。

## 2026-05-13 本地 RULER 项目说明文档重写

### 变更文件

- `AGENTS.md`
  - 重写项目总览，明确当前仓库是本地 RULER benchmark 测评工作区。
  - 补充从原始 parquet 数据到 RULER jsonl、预测 jsonl、日志、评分 csv 的完整数据流。
  - 按顶层目录和 `RULER/` 关键脚本解释文件夹来源、用途和生成关系。
  - 修正 `RULER/` 的 git 状态说明：当前它由外层 git 仓库跟踪，不是独立 `.git` 子仓库，但仍建议同时运行 `git status --short` 和 `git -C RULER status --short`。
  - 补充当前本地可见模型、已转换长度、任务集合、已有预测和缺失评分文件的状态说明。
  - 新增 `prepare_parquet.py`、`run_parquet_parallel.py`、`call_api.py`、`evaluate.py` 的常用命令和参数解释。
  - 新增任务含义、长度目录含义、已有数据和结果检查命令。
  - 更新测试规范，移除当前工作区不存在的旧顶层 runner 示例，保留当前 RULER parquet 流程的验证命令。

- `CODEX_CHANGES.md`
  - 记录本次文档重写的内容、运行方式、验证方式、假设和限制。

### 变更目的

本次变更用于让 `AGENTS.md` 不只是协作规则，还能作为当前项目的入口说明。阅读该文件后，应能理解哪些目录是下载来的模型和 benchmark，哪些目录是转换或预测生成的，如何启动测评，命令参数分别控制什么，以及测评结果应该到哪里查看。

### 主要函数和类

本次只修改文档，没有新增或修改 Python 函数、类或运行时代码。

### 运行方式

文档中整理的主流程包括：

```bash
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress \
  --dry-run
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

### 验证方式

本次文档变更应运行：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次没有修改执行代码，因此不需要重新跑模型推理。
- 文档中的本地资产状态基于 2026-05-13 当前工作区检查结果；后续新增模型、数据或评分文件后需要同步更新。
- `AGENTS.md` 使用当前实际存在的 RULER parquet 工作流，不再把缺失的旧顶层 runner 当作主要入口。
- 评分仍要求预测 jsonl 已经存在；`run_parquet_parallel.py` 只负责预测，不自动调用 `eval/evaluate.py`。

## 2026-05-12 call_api 有限重试与失败释放

### 变更文件

- `RULER/scripts/pred/call_api.py`
  - 新增 `--max_retries` 参数，默认每个 batch 最多重试 3 次。
  - 新增 `[BATCH_FAILED]` 失败日志，包含任务名、batch 序号、样本范围、重试次数和异常类型。
  - 将原来的无限重试改为有限重试；超过次数后把错误传回主线程，让 `call_api.py` 以非零状态退出。

- `RULER/scripts/run_parquet_parallel.py`
  - 扩展 `is_batch_progress_line()`，让 runner 终端同步回显 `[BATCH_FAILED]`。

- `tests/test_call_api_progress.py`
  - 增加 `process_batch_with_retries()` 的瞬时失败恢复和超过重试次数抛错测试。
  - 确认 `--max_retries` 和 `[BATCH_FAILED]` 保留在 `call_api.py` 中。

- `tests/test_run_parquet_parallel.py`
  - 增加 runner 识别 `[BATCH_FAILED]` 的测试。

- `CODEX_CHANGES.md`
  - 记录本次有限重试、失败释放和验证方式。

### 变更目的

本次变更用于解决 GLM 或其他模型在 `llm.process_batch()` 内持续报错时，`call_api.py` 无限重试导致子进程不退出、GPU worker 无法释放的问题。现在单个 batch 连续失败超过 `--max_retries` 后，错误会从推理线程传回主线程，进程退出码变为非零，`run_parquet_parallel.py` 随后会打印 `[FAILED]` 并把对应 GPU 放回可调度队列。

### 主要函数和类

- `format_batch_failure_message()`
  - 生成稳定的 `[BATCH_FAILED]` 日志文本。

- `process_batch_with_retries()`
  - 对 `llm.process_batch()` 做有限重试，失败时打印 traceback 和 batch 元信息，超过次数后抛错。

- `raise_worker_errors()`
  - 在主线程检查推理线程记录的异常，并重新抛出，确保子进程非零退出。

- `is_batch_progress_line()`
  - 新增对 `[BATCH_FAILED]` 的识别。

### 运行方式

单独运行 `call_api.py` 时可以显式设置重试次数：

```bash
cd /home/test05/czyprojects/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u pred/call_api.py \
  --data_dir ../benchmark_root/parquet_data/synthetic/4096/data \
  --save_dir ../benchmark_root/local_eval/glm-4-9b/synthetic/4096/pred \
  --benchmark synthetic \
  --task niah_single_1 \
  --server_type hf \
  --model_name_or_path ../../models/glm-4-9b \
  --temperature 0.0 \
  --top_k 32 \
  --top_p 1.0 \
  --batch_size 1 \
  --max_retries 3 \
  --log_batch_progress
```

通过 runner 运行时，失败 batch 会同时写入日志并回显到终端；当前 runner 尚未新增透传 `--max_retries` 的命令行参数，因此使用 `call_api.py` 默认值 3。

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/codex-pycache conda run -n dl-a800 python -m py_compile RULER/scripts/pred/call_api.py RULER/scripts/run_parquet_parallel.py
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次变更解决的是“失败后释放 GPU worker”的问题，不修复 GLM 与当前 Transformers cache 格式不兼容的根因。
- 如果模型调用不是抛异常，而是在 CUDA 内部永久卡死，则仍需要后续在 runner 层增加任务级 timeout 和强制 kill。
- runner 暂时使用 `call_api.py` 的 `--max_retries` 默认值；如需按命令行调整，需要后续给 `run_parquet_parallel.py` 增加透传参数。

## 2026-05-12 GLM wrapper 兼容与 batch 进度回显修复

### 变更文件

- `RULER/scripts/pred/model_wrappers.py`
  - 新增 `ensure_num_hidden_layers_alias()`。
  - 对只提供 `num_layers`、没有 `num_hidden_layers` 的模型配置补充兼容别名。
  - 解决 `glm-4-9b` 在新版 Transformers 生成阶段触发 `ChatGLMConfig object has no attribute 'num_hidden_layers'` 的问题。

- `RULER/scripts/run_parquet_parallel.py`
  - 修改 `is_batch_progress_line()`，从行首匹配改为行内包含匹配。
  - 解决 `tqdm` 进度条前缀和 `[BATCH_START]` 写在同一行时，runner 终端不回显 batch 进度的问题。

- `tests/test_model_wrappers.py`
  - 新增 `ensure_num_hidden_layers_alias()` 的单元测试。
  - 覆盖缺少 `num_hidden_layers` 时补别名，以及已有 `num_hidden_layers` 时不覆盖原值。

- `tests/test_run_parquet_parallel.py`
  - 增加带 `tqdm` 前缀的 `[BATCH_START]` 行识别测试。

- `CODEX_CHANGES.md`
  - 记录本次 GLM wrapper 兼容和 runner 终端回显修复。

### 变更目的

本次变更用于解决两个实际运行问题。第一，`glm-4-9b` 的本地配置使用 `num_layers`，而当前 Transformers 的生成缓存逻辑会访问 `num_hidden_layers`，导致 `call_api.py` 子进程在第一条样本上无限重试。第二，`tqdm` 可能把进度条文本和 `[BATCH_START]` 输出写在同一行，runner 之前只按行首匹配，因此命令行上看不到 batch 进度。

### 主要函数和类

- `ensure_num_hidden_layers_alias(config)`
  - 如果模型配置没有 `num_hidden_layers` 但有 `num_layers`，则补充 `config.num_hidden_layers = config.num_layers`。

- `is_batch_progress_line(line)`
  - 现在只要一行中包含 `[BATCH_START]` 或 `[BATCH_DONE]`，就会被 runner 识别为需要回显的 batch 进度行。

### 运行方式

如果已有旧 runner 或旧 `call_api.py` 子进程正在运行，需要先停止旧进程，然后重新启动。重新运行时可以继续使用：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --model Qwen3-8B=../../models/Qwen3-8B \
  --model glm-4-9b=../../models/glm-4-9b \
  --model Yi-9B-200K=../../models/Yi-9B-200K \
  --seq-lengths 4096 \
  --tasks all \
  --gpus 0,1,2,3,4,5,6,7 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 10 \
  --log-batch-progress
```

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_run_parquet_parallel.py'
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_model_wrappers.py'
conda run -n dl-a800 python -B -m py_compile RULER/scripts/run_parquet_parallel.py RULER/scripts/pred/model_wrappers.py
```

### 假设和限制

- 该修复解决的是 `glm-4-9b` 配置字段兼容问题，不保证所有 ChatGLM 类模型的 prompt 模板和停止词都已经最优。
- 已经在运行中的 Python 子进程不会自动应用本次代码变更，必须重启对应进程。
- 如果后续仍出现 OOM，需要继续降低 `--batch-size` 或减少并发任务；本次修复不改变显存需求。

## 2026-05-12 协作文档同步

### 变更文件

- `AGENTS.md`
  - 更新项目结构说明，补充 `RULER/`、`prepare_parquet.py`、`run_parquet_parallel.py`、`call_api.py` 和 `eval/evaluate.py` 的职责。
  - 更新构建、测试与开发命令，补充 parquet 转换、并行预测 dry-run、多模型并行预测和 RULER 原生评分命令。
  - 明确预测 jsonl、运行日志和评分结果的默认输出路径。
  - 更新测试规范，补充 RULER parquet 转换、并行 runner 和 `call_api.py` 相关变更应运行的验证命令。
  - 补充 `RULER/` 是嵌套 git 仓库的说明，要求同时检查外层仓库和嵌套仓库状态。
  - 强调不要提交 `__pycache__/`、`.pyc`、模型权重、原始 benchmark parquet、预测输出、评分输出或运行日志。

- `CODEX_CHANGES.md`
  - 记录本次 `AGENTS.md` 文档同步的内容、目的、运行方式、验证方式、假设和限制。

### 变更目的

本次变更用于把协作说明同步到当前真实工作流。现在仓库不仅包含旧版 `run_ruler_4k.py`，还包含基于 RULER 原生脚本的 parquet 转换、并行预测和评分流程。更新后的 `AGENTS.md` 可以让后续协作时直接看到如何生成 jsonl、如何用一张 GPU 跑一个任务、预测结果在哪里、以及如何用 `eval/evaluate.py` 生成最终分数。

### 主要函数和类

本次只修改文档，没有新增或修改 Python 函数、类或运行时代码。

### 运行方式

文档中新增或确认的核心命令包括：

```bash
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model llama=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1000 \
  --log-batch-progress \
  --dry-run
```

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/llama/synthetic/4096/pred \
  --benchmark synthetic
```

### 验证方式

本次文档变更应运行：

```bash
git diff --check
git -C RULER diff --check
```

### 假设和限制

- 本次没有修改执行代码，因此不需要重新跑模型推理。
- 文档中的示例模型别名 `llama`、`qwen` 和模型路径只是示例，实际运行时应与本地 `models/` 目录一致。
- RULER 原生评分仍要求预测 jsonl 已经生成；`run_parquet_parallel.py` 本身只负责预测，不自动运行 `eval/evaluate.py`。
- `eval/evaluate.py` 会把 `summary.csv` 或 `summary-<task>.csv` 写回传入的 `pred/` 目录。

## 2026-05-12 parquet 数据并行预测 runner

### 变更文件

- `RULER/scripts/run_parquet_parallel.py`
  - 新增 parquet-jsonl 数据预测调度脚本。
  - 支持一个命令重复传入多个 `--model NAME=PATH`，并展开为模型、长度和任务的任务矩阵。
  - 支持 `--seq-lengths all` 自动发现已经转换好的长度目录，也支持显式长度列表。
  - 支持 `--tasks all` 或显式任务列表，任务名沿用 RULER `synthetic.yaml`。
  - 通过 `CUDA_VISIBLE_DEVICES=<gpu>` 为每个 `pred/call_api.py` 子进程只暴露一张 GPU。
  - 使用动态调度：某张 GPU 上的子进程结束后，立即从队列里取下一个任务补上。
  - 预测输出写入 `RULER/benchmark_root/local_eval/<模型名>/synthetic/<长度>/pred/<任务>.jsonl`。
  - 子进程日志写入 `RULER/benchmark_root/local_eval/<模型名>/synthetic/<长度>/logs/<任务>.log`。
  - 子进程输出中的 `[BATCH_START]` 和 `[BATCH_DONE]` 会同步回显到 runner 终端，同时保留在日志文件中。

- `RULER/scripts/pred/call_api.py`
  - 新增可选参数 `--log_batch_progress`。
  - 参数开启后，在子进程日志里输出 `[BATCH_START]` 和 `[BATCH_DONE]`，包含任务名、batch 序号、batch 大小、样本 index 范围和耗时。
  - 默认不输出这些 batch 日志，因此不改变原有命令的默认行为。

- `tests/test_run_parquet_parallel.py`
  - 新增标准库 `unittest` 测试。
  - 覆盖模型参数解析、长度发现、任务展开、任务矩阵顺序、batch 进度行识别和 `call_api.py` 命令构造。

- `tests/test_call_api_progress.py`
  - 新增轻量静态回归测试。
  - 确认 `call_api.py` 保留 `--log_batch_progress`、`[BATCH_START]` 和 `[BATCH_DONE]`。

- `CODEX_CHANGES.md`
  - 记录本次新增 runner、进度日志开关、运行方式、验证方式、假设和限制。

### 变更目的

本次变更用于把已经通过 `prepare_parquet.py` 转好的本地 parquet-jsonl 数据直接送入 RULER 原生预测流程，同时避免一个 Hugging Face 本地模型默认占用 8 张 GPU。新的 runner 会把每个任务作为独立子进程启动，并把每个子进程限制在一张 GPU 上，从而可以在 8 张 GPU 上并行跑多个模型或多个任务。

### 主要函数和类

- `ModelSpec`
  - 表示一个模型别名和模型路径。

- `Job`
  - 表示一个具体的模型、长度和任务组合。

- `RunnerConfig`
  - 保存构造 `pred/call_api.py` 命令需要的路径、采样参数和日志开关。

- `parse_model_specs()`
  - 解析重复传入的 `--model NAME=PATH` 参数。

- `resolve_seq_lengths()` 和 `resolve_tasks()`
  - 解析 `all` 或逗号分隔列表，并展开长度和任务。

- `build_jobs()`
  - 按长度、任务、模型顺序展开任务矩阵，使多个模型尽早交错进入队列。

- `build_call_api_command()`
  - 构造单个任务调用 `pred/call_api.py` 的命令。

- `is_batch_progress_line()` 和 `stream_child_output()`
  - 识别子进程 batch 进度行，并把这些行同步回显到 runner 终端。

- `run_scheduler()`
  - 维护待运行队列、运行中子进程和空闲 GPU 队列，负责动态补任务。

- `launch_job()`
  - 设置 `CUDA_VISIBLE_DEVICES` 和 `PYTHONUNBUFFERED`，启动单个 `call_api.py` 子进程。

### 运行方式

只做 dry-run，验证命令展开是否正确：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model llama=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1000 \
  --log-batch-progress \
  --dry-run
```

一个命令同时调度四个模型，使用 8 张 GPU 动态补任务：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model llama=../../models/Meta-Llama-3.1-8B \
  --model qwen=../../models/Qwen3-8B \
  --model yi=../../models/Yi-9B-200K \
  --model other=../../models/Your-Other-Model \
  --seq-lengths all \
  --tasks all \
  --gpus 0,1,2,3,4,5,6,7 \
  --server-type hf \
  --batch-size 1000 \
  --poll-interval 10 \
  --log-batch-progress
```

运行过程中，runner 终端会显示 `[QUEUE]`、`[START]`、`[RUNNING]`、`[BATCH]`、`[DONE]`、`[FAILED]` 和 `[SUMMARY]`。每个任务的完整子进程输出仍会写入对应日志文件。

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_run_parquet_parallel.py'
conda run -n dl-a800 python -B -m unittest discover -s tests -p 'test_call_api_progress.py'
conda run -n dl-a800 python -B -m unittest discover -s tests
conda run -n dl-a800 python -B -m py_compile RULER/scripts/run_parquet_parallel.py RULER/scripts/pred/call_api.py
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python -B run_parquet_parallel.py \
  --model llama=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1000 \
  --log-batch-progress \
  --dry-run
```

### 假设和限制

- runner 默认从 `RULER/benchmark_root/parquet_data/synthetic/<长度>/data` 读取已经转换好的 jsonl。
- runner 不负责生成 parquet-jsonl 数据；数据生成仍然由 `RULER/scripts/data/prepare_parquet.py` 完成。
- runner 不负责评分；预测完成后仍然需要按 RULER 原生流程调用 `eval/evaluate.py`。
- `--server-type hf` 仍然使用上游 `pred/model_wrappers.py` 中的 `HuggingFaceModel`。
- 单卡限制依赖 `CUDA_VISIBLE_DEVICES`，即每个子进程只看到一张物理 GPU。
- `call_api.py` 仍保留上游 `nemo` 依赖；如果当前 Python 环境缺少 `nemo`，真实预测会在 import 阶段失败。
- `--batch-size 1000` 只是把参数传给 `call_api.py`，实际能否跑取决于模型、上下文长度和单张 GPU 显存。

## 2026-05-12 RULER parquet 转换脚本

### 变更文件

- `RULER/scripts/data/prepare_parquet.py`
  - 新增本地 parquet 到 RULER jsonl 的离线转换脚本。
  - 默认扫描 `benchmark/RULER-llama3-1M/` 下所有 `<任务>_<长度>` 目录。
  - 默认写入 `RULER/benchmark_root/parquet_data/synthetic/<长度>/data/<任务>/validation.jsonl`。
  - 将 parquet 字段 `answers` 转成 RULER 后续流程需要的 `outputs`，忽略已有 `predictions`。
  - 支持 `--tasks`、`--lengths`、`--max_samples`、`--skip_existing`、`--subset`、`--report_file` 等可选参数。

- `tests/test_prepare_parquet.py`
  - 新增标准库 `unittest` 测试。
  - 覆盖任务目录名解析、长度过滤解析、字段归一化和 jsonl 写出逻辑。

- `CODEX_CHANGES.md`
  - 记录本次新增脚本、测试、运行方式、验证方式、假设和限制。

### 变更目的

本次变更用于复用已经下载好的 `benchmark/RULER-llama3-1M/` parquet 数据，不再通过 RULER 原版 `data/prepare.py` 重新生成 synthetic 数据。转换后的 jsonl 保持原版 `pred/call_api.py` 和 `eval/evaluate.py` 的输入格式，从而可以直接用 RULER 文件夹中的预测和评分流程跑本地模型。

### 主要函数

- `parse_dataset_dir_name()`
  - 从 `niah_multikey_1_128k` 这类目录名解析出任务名、长度后缀和数字长度。

- `parse_length_value()`、`canonical_length_suffix()`、`parse_length_filter()`
  - 负责 `4k`、`128k`、`1M`、`4096` 等长度写法和数字长度之间的转换。

- `read_parquet_records()`
  - 使用 `pyarrow.parquet` 读取 parquet 分片。

- `normalize_record()` 和 `write_jsonl()`
  - 将单条 parquet 记录转换成 RULER jsonl 记录，并按 `index` 排序写出。

- `discover_dataset_dirs()` 和 `convert_dataset_dir()`
  - 自动发现需要转换的任务长度目录，并写出对应 jsonl。

### 运行方式

全量转换：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

只转换一个 4k 任务的一条样本：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py \
  --tasks niah_single_1 \
  --lengths 4k \
  --max_samples 1
```

转换后的数据示例路径：

```text
RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl
```

### 验证方式

已运行：

```bash
conda run -n dl-a800 python -m unittest tests.test_prepare_parquet
conda run -n dl-a800 python -m py_compile RULER/scripts/data/prepare_parquet.py
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py \
  --save_dir /tmp/ruler_parquet_test \
  --report_file /tmp/ruler_parquet_test_report.json \
  --tasks niah_single_1 \
  --lengths 4k \
  --max_samples 1
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py \
  --save_dir /tmp/ruler_parquet_test_long \
  --report_file /tmp/ruler_parquet_test_long_report.json \
  --tasks niah_single_1 \
  --lengths 128k,1M \
  --max_samples 1
```

### 假设和限制

- parquet 字段固定为 `index`、`input`、`answers`、`length`、`predictions`。
- parquet 中的 `input` 已经是完整 Completion-style prompt，不再经过 `template.py`。
- `dl-a800` 环境中安装了 `pyarrow`。
- 本次没有修改 `RULER/scripts/run.sh`，后续测评时需要手动把 `call_api.py --data_dir` 指向对应长度的 `data` 目录，或再单独添加 `run.sh` 接入开关。

## 变更文件

- `print_this_is.py`
  - 新增一个简单 Python 脚本。
  - 运行后输出固定文本 `this is`。

- `tests/test_print_this_is.py`
  - 新增标准库 `unittest` 测试。
  - 验证 `print_this_is.py` 退出码为 `0`，标准输出精确为 `this is\n`，标准错误为空。

- `CODEX_CHANGES.md`
  - 记录新增脚本和测试的目的、运行方式、验证方式、假设和限制。

- `AGENTS.md`
  - 将仓库协作规范全文翻译为中文。
  - 明确要求本项目中的代码注释、文档字符串、解释文档、变更说明和面向开发者的说明文件必须使用中文。
  - 说明命令、路径、Python API 名称、模型名、benchmark 名称和必要英文专有名词可以保留原文。

- `run_ruler_4k.py`
  - 这是本地 Hugging Face 自回归语言模型的 RULER 4k 基准测试脚本。
  - 支持从 `models/` 自动发现本地模型，并从 `benchmark/RULER-llama3-1M/*_4k` 自动发现 4k 任务。
  - 为主要辅助函数和 `Example` 数据类补充了中文文档字符串。
  - 为非显然逻辑补充了中文注释，包括 parquet 读取兜底、Llama-3 提示词转换、分词器 padding 兜底、Transformers dtype 兼容、断点续跑、Hugging Face 模块缓存设置等。
  - 清理了批量输入张量移动设备时的重复分支。
  - 将运行时提示、错误信息和命令行帮助中的解释性文本改为中文。

- `CODEX_CHANGES.md`
  - 用中文记录本次代码变更的目的、主要函数、运行方式、验证方式、假设和限制。

## 变更目的

`print_this_is.py` 用于满足一个最小命令行输出需求：运行 Python 文件后输出 `this is`。

`run_ruler_4k.py` 用于把 `models/` 下的本地模型跑在 `benchmark/RULER-llama3-1M/` 下的 RULER 4k parquet 任务上。脚本可以运行单个模型、多个指定模型，或全部本地模型，并把逐任务预测和汇总分数写入 `outputs/ruler_4k/`。

脚本默认采用本地优先策略：读取本地模型权重和本地 parquet 基准文件，`local_files_only=True` 默认开启。

## 主要函数和类

- `print_this_is.main()`
  - 打印固定文本 `this is`。

- `PrintThisIsTest`
  - 使用标准库 `unittest` 运行脚本并检查输出。

- `Example`
  - 表示单条已经归一化后的基准样本。

- `read_parquet_records()`
  - 读取基准 parquet 文件。
  - 优先使用 `pyarrow`，如果不可用再尝试 `pandas` 或 `datasets`。

- `load_examples()`
  - 读取单个任务下的所有 parquet 分片。
  - 按 `index` 排序，并支持通过 `--max-samples` 截断样本数。

- `parse_llama3_segments()`、`plain_from_llama3_prompt()`、`chat_from_llama3_prompt()`、`build_prompt()`
  - 负责提示词预处理。
  - `plain` 会去掉 Llama-3 对话外壳 token，同时保留任务内容和少样本示例。
  - `chat` 会把 Llama-3 格式的多轮内容转换成当前分词器的对话模板。
  - `as_is` 会原样使用 parquet 中的提示词。

- `sample_score()` 和 `score_predictions()`
  - 计算 RULER 风格的大小写不敏感字符串匹配分数。

- `discover_models()` 和 `discover_tasks()`
  - 把命令行里的模型和任务选择解析成本地目录。

- `load_tokenizer()` 和 `load_model()`
  - 加载本地 Hugging Face 分词器和模型。
  - 支持 dtype、device map、remote-code trust 策略和 local-files-only 行为配置。

- `generate_batch()`
  - 对提示词做分词。
  - 调用 `model.generate()`。
  - 只解码新增生成的 token。

- `run_model()`
  - 为单个模型运行所有指定 RULER 任务，并写入 jsonl 预测文件。

- `write_summary()`
  - 为单个模型写入 `summary.json` 和 `summary.csv`。

## 运行方式

运行简单输出脚本：

```bash
cd /home/test05/czyprojects
python3 print_this_is.py
```

只检查模型和任务发现，不启动推理：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python run_ruler_4k.py --dry-run --model all --tasks all
```

最小冒烟测试：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python run_ruler_4k.py \
  --model Meta-Llama-3.1-8B \
  --tasks niah_single_1_4k \
  --max-samples 1 \
  --max-new-tokens 8 \
  --overwrite
```

运行全部本地模型和全部 RULER 4k 任务：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python run_ruler_4k.py \
  --model all \
  --tasks all \
  --batch-size 1 \
  --continue-on-error
```

## 验证方式

验证简单输出脚本：

```bash
python3 -m unittest tests.test_print_this_is
```

语法检查：

```bash
python3 -m py_compile run_ruler_4k.py
```

空运行检查：

```bash
conda run -n dl-a800 python run_ruler_4k.py --dry-run --model all --tasks all
```

提示词转换检查：

```bash
conda run -n dl-a800 python -c "from pathlib import Path; from run_ruler_4k import load_examples, build_prompt; root=Path('benchmark/RULER-llama3-1M'); print([(p.name, build_prompt(load_examples(p, 1)[0].input, object(), 'plain').count('<|')) for p in sorted(root.glob('*_4k'))])"
```

预期结果：每个任务输出的计数都应为 `0`，表示被检查样本中的 Llama-3 控制 token 标记已经在 `plain` 模式下被去掉。

## 假设和限制

- `print_this_is.py` 只负责输出固定文本，不解析命令行参数。
- 基准数据已经以 parquet 文件形式存在于 `benchmark/RULER-llama3-1M/` 下。
- 运行环境至少安装一个 parquet 读取器；当前脚本首选 `pyarrow`。
- `dl-a800` 环境应包含 `torch`、`transformers`、`accelerate`、`safetensors`、`sentencepiece`、`pyarrow` 和 `tiktoken`。
- 完整 8B/9B 推理需要 CUDA 可见的 GPU 和足够显存；如果 CUDA 不可见，脚本会打印警告。
- 当前评分逻辑是轻量级字符串匹配，符合这批 RULER 任务的基本评估方式，但没有完整复刻上游 RULER 的所有评估工具。
- `outputs/ruler_4k/` 下的生成结果属于运行产物，除非明确要求，否则不应提交。
