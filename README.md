# 本地 RULER 长上下文测评工作区

本仓库用于在本地复用已经下载好的 RULER benchmark parquet 数据，调用本地 Hugging Face 模型生成预测，再计算 RULER 原生准确率、生成阶段困惑度和运行耗时。当前主流程不重新生成 RULER 数据，而是使用 `benchmark/RULER-llama3-1M/` 中已有数据。

核心流程：

```text
benchmark/RULER-llama3-1M/
  -> RULER/scripts/data/prepare_parquet.py
  -> RULER/benchmark_root/parquet_data/synthetic/<长度>/data/<任务>/validation.jsonl
  -> RULER/scripts/run_parquet_parallel.py
  -> RULER/benchmark_root/local_eval/<模型>/synthetic/<长度>/pred/<任务>.jsonl
  -> RULER/scripts/eval/collect_results.py
  -> RULER/benchmark_root/local_eval/ruler_results.xlsx
```

## 目录和文件

### 根目录文件

- `README.md`
  - 当前文件，说明项目目录、测评命令、参数含义和输出位置。
- `AGENTS.md`
  - 给协作 agent 和维护者看的完整工作规范。
  - 记录更细的目录来源、命令参数、测试要求和不要提交的文件类型。
- `CODEX_CHANGES.md`
  - Codex 每次文件变更后的中文变更记录。
  - 只要修改仓库文件，就需要同步更新。
- `.gitignore`
  - Git 忽略规则。
- `.git/`
  - 当前仓库的 Git 元数据，不手动修改。
- `.agents/`、`.codex`
  - 本地 agent 工具运行相关目录或标记文件，一般不需要手动操作。

### 数据、模型和输出目录

- `models/`
  - 本地模型权重目录。
  - 当前可见模型包括：
    - `models/Meta-Llama-3.1-8B/`
    - `models/Qwen3-8B/`
    - `models/Yi-9B-200K/`
    - `models/glm-4-9b/`
  - 这些通常是大文件，不要提交模型权重。

- `benchmark/RULER-llama3-1M/`
  - 已下载好的 RULER parquet benchmark 数据。
  - 子目录形如 `niah_single_1_4k/`、`qa_2_128k/`、`cwe_1M/`。
  - 每个任务长度目录中通常有 `validation-*.parquet`。
  - 不要提交原始 parquet 或 Hugging Face 缓存。

- `RULER/`
  - 本地 RULER 项目副本和本仓库适配脚本。
  - 主要代码都在 `RULER/scripts/`。

- `RULER/benchmark_root/parquet_data/synthetic/`
  - parquet 转换后的 RULER jsonl 输入。
  - 结构为 `<长度>/data/<任务>/validation.jsonl`。

- `RULER/benchmark_root/local_eval/`
  - 本地评测输出根目录。
  - 标准结构：
    - `local_eval/<模型>/synthetic/<长度>/pred/<任务>.jsonl`
    - `local_eval/<模型>/synthetic/<长度>/logs/<任务>.log`
  - 统一汇总默认输出：
    - `local_eval/ruler_timing.jsonl`
    - `local_eval/ruler_results.xlsx`

- `attention_dumps/`
  - 单样本完整 attention 导出结果。
  - 例如 `attention_dumps/llama_niah_single_1_sample_0/`。
  - 文件可能很大，通常作为本地诊断输出，不提交。

### 脚本和测试目录

- `RULER/scripts/data/prepare_parquet.py`
  - 把 `benchmark/RULER-llama3-1M/` 的 parquet 转成 RULER 预测脚本可读的 jsonl。

- `RULER/scripts/run_parquet_parallel.py`
  - 并行调度器。
  - 按模型、长度、任务展开任务矩阵。
  - 每个子进程通过 `CUDA_VISIBLE_DEVICES=<gpu>` 只看到一张 GPU。
  - 支持覆盖重跑、生成阶段 PPL、timing 记录和自动生成统一测评表。

- `RULER/scripts/pred/call_api.py`
  - 单任务预测入口。
  - 读取 `validation.jsonl`，加载模型，写出带 `pred` 字段的预测 jsonl。
  - 本地支持 `--log_generation_ppl`，会在预测记录中写入：
    - `generation_logprob_sum`
    - `generation_token_count`
    - `generation_nll`
    - `generation_ppl`

- `RULER/scripts/pred/model_wrappers.py`
  - 本地 Hugging Face、Mamba 等模型 wrapper。
  - Hugging Face wrapper 中的生成阶段 PPL 基于 `generate(..., output_scores=True)` 返回的 token scores。

- `RULER/scripts/eval/evaluate.py`
  - RULER 原生评分脚本。
  - 对单个 `pred/` 目录生成 `summary.csv`、`summary-<任务>.csv` 和 `submission.csv`。

- `RULER/scripts/eval/collect_results.py`
  - 本地统一汇总脚本。
  - 跨模型、长度和任务汇总准确率、PPL、耗时和完成状态。
  - 输出一个 xlsx，包含：
    - `detail`
    - `summary_by_model`
    - `summary_by_model_and_length`
    - `summary_by_task`
    - `run_info`

- `tools/dump_llama_attention.py`
  - 导出一个 Llama 样本生成过程中的完整 attention。

- `tools/inspect_attention_dump.py`
  - 查看某个生成 token、某层、某个 head 的完整 attention 表格。
  - 支持按 position 或 attention 权重排序。

- `tests/`
  - 本地回归测试。
  - 覆盖 parquet 转换、runner 命令构造、PPL 字段、统一汇总和 attention 工具。

## 环境检查

RULER 主流程默认使用 `dl-a800` conda 环境。GLM 当前单独使用 `ruler-glm44`。

检查基础依赖：

```bash
conda run -n dl-a800 python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
```

如果要跑 GLM，确认 GLM 环境也能导入需要的依赖：

```bash
conda run -n ruler-glm44 python -c "import torch, transformers; print('ok')"
```

## 数据转换

如果 `RULER/benchmark_root/parquet_data/synthetic/<长度>/data/<任务>/validation.jsonl` 已经存在，通常不需要重复转换。

转换全部数据：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

只转换一个小样本用于检查：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py \
  --tasks niah_single_1 \
  --lengths 4k \
  --max_samples 5 \
  --skip_existing
```

常用参数：

- `--parquet_data_dir`：原始 parquet 根目录，默认 `benchmark/RULER-llama3-1M`。
- `--save_dir`：转换后 jsonl 保存根目录，默认 `RULER/benchmark_root/parquet_data/synthetic`。
- `--subset`：数据 split，默认 `validation`。
- `--tasks`：任务过滤，例如 `niah_single_1,qa_2`。
- `--lengths`：长度过滤，例如 `4k,128k,1M` 或 `4096,131072`。
- `--max_samples`：每个任务长度最多转换多少条，适合 smoke test。
- `--skip_existing`：目标文件存在时跳过，避免覆盖。

## 并行评测命令

下面这条是你给定的四模型 4k 全任务覆盖重跑命令。它会删除对应任务旧预测并从头跑，写 timing 文件，并在全部任务结束后自动生成统一 xlsx 测评表。

运行前进入脚本目录：

```bash
cd /home/test05/czyprojects/RULER/scripts
```

正式命令：

```bash
conda run --no-capture-output -n dl-a800 python -B run_parquet_parallel.py \
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
  --overwrite-existing \
  --auto-evaluate \
  --timing-file /home/test05/czyprojects/RULER/benchmark_root/local_eval/ruler_4k_timing.jsonl \
  --report-file /home/test05/czyprojects/RULER/benchmark_root/local_eval/ruler_4k_results.xlsx
```

这条命令会自动测评并生成 `ruler_4k_results.xlsx`。如果还希望在预测 jsonl 和 xlsx 中包含生成阶段 PPL，需要在命令中额外加入：

```bash
  --log-generation-ppl \
```

### 这条命令会产生什么

- 每个任务预测：
  - `RULER/benchmark_root/local_eval/<模型>/synthetic/4096/pred/<任务>.jsonl`
- 每个任务日志：
  - `RULER/benchmark_root/local_eval/<模型>/synthetic/4096/logs/<任务>.log`
- 任务耗时记录：
  - `RULER/benchmark_root/local_eval/ruler_4k_timing.jsonl`
- 统一测评表：
  - `RULER/benchmark_root/local_eval/ruler_4k_results.xlsx`

### run_parquet_parallel.py 参数说明

- `--model NAME=PATH`
  - 指定模型别名和本地模型路径。
  - 可以重复传入多个模型。
  - `NAME` 会成为输出目录名。
  - 示例：`--model Qwen3-8B=../../models/Qwen3-8B`。

- `--python PYTHON`
  - 默认子进程 Python。
  - 上面命令中默认使用 `dl-a800` 环境的 Python。

- `--model-python NAME=PYTHON`
  - 给某个模型单独指定 Python。
  - 当前 GLM 使用 `ruler-glm44`，其他模型继续使用 `dl-a800`。
  - 适合不同模型依赖不完全一致的情况。

- `--seq-lengths`
  - 要评测的上下文长度。
  - 可以写 `4096`、`4096,8192`，也可以写 `all`。
  - `all` 会自动发现 `--data-root` 下的数字长度目录。

- `--tasks`
  - 要评测的任务。
  - 可以写 `niah_single_1,qa_2`，也可以写 `all`。
  - `all` 对应 13 个 synthetic 任务。

- `--gpus`
  - 物理 GPU 编号列表。
  - runner 会把每个子进程绑定到一张 GPU。
  - 示例：`0,1,2,3,4,5,6,7`。

- `--max-workers`
  - 最大并发子进程数。
  - 不传时默认等于 GPU 数量。
  - 如果想 8 张卡只跑 4 个并发任务，可设为 `--max-workers 4`。

- `--server-type hf`
  - 使用本地 Hugging Face 模型。
  - 当前本地模型主流程使用 `hf`。

- `--batch-size`
  - 单个 `call_api.py` 子进程每批推理样本数。
  - 长上下文建议保守使用 `1`，避免 OOM。

- `--poll-interval`
  - runner 打印运行状态和检查任务结束的间隔秒数。
  - 常用 `10`。

- `--log-batch-progress`
  - 让子进程输出 `[BATCH_START]`、`[BATCH_DONE]`、`[BATCH_FAILED]`。
  - runner 会同步回显这些日志。

- `--skip-existing`
  - 如果预测文件已经存在，就跳过该任务。
  - 适合断点补跑。
  - 不能和 `--overwrite-existing` 同时使用。

- `--overwrite-existing`
  - 启动任务前删除该任务已有预测、attention 摘要和日志，然后从头重跑。
  - 会删除：
    - `<任务>.jsonl`
    - `<任务>.attention.jsonl`
    - `<任务>.attention.md`
    - `<任务>.log`
  - 不会删除 `summary.csv`、`submission.csv` 或已有 xlsx 汇总。

- `--log-generation-ppl`
  - 让本地 HF 子进程记录生成阶段 PPL。
  - 写入每条预测 jsonl 的字段：
    - `generation_logprob_sum`
    - `generation_token_count`
    - `generation_nll`
    - `generation_ppl`
  - 这个 PPL 是模型对自己生成答案 token 的困惑度，不是参考答案困惑度。

- `--auto-evaluate`
  - runner 所有任务结束后自动调用 `RULER/scripts/eval/collect_results.py`。
  - 用于生成统一 xlsx 汇总表。

- `--timing-file`
  - 结构化耗时 jsonl 输出路径。
  - 如果不传，默认写到 `RULER/benchmark_root/local_eval/ruler_timing.jsonl`。

- `--report-file`
  - 统一 xlsx 汇总输出路径。
  - 如果不传，默认写到 `RULER/benchmark_root/local_eval/ruler_results.xlsx`。

- `--dry-run`
  - 只打印任务矩阵和将要执行的命令，不启动推理。
  - 改命令前建议先 dry-run。

dry-run 示例：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python -B run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-generation-ppl \
  --dry-run
```

## 单独调用评测文件

如果预测已经跑完，或者没有在 runner 中传 `--auto-evaluate`，可以单独调用 `RULER/scripts/eval/collect_results.py` 生成统一测评表，不需要重新推理：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/collect_results.py \
  --output-root ../benchmark_root/local_eval \
  --data-root ../benchmark_root/parquet_data/synthetic \
  --benchmark synthetic \
  --seq-lengths 4096 \
  --tasks all \
  --timing-file ../benchmark_root/local_eval/ruler_4k_timing.jsonl \
  --output-file ../benchmark_root/local_eval/ruler_4k_results.xlsx
```

如果你使用的是上面四模型命令中的绝对路径，也可以这样单独评测：

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

这就是“单独评测”的推荐方式：它读取已有预测 jsonl、输入 jsonl 和 timing jsonl，重新生成 xlsx，不会重新加载模型，也不会重新跑预测。

参数说明：

- `--output-root`
  - 预测、日志和汇总所在根目录。
  - 默认是 `RULER/benchmark_root/local_eval`。

- `--data-root`
  - 转换后的 jsonl 输入数据根目录。
  - 用于统计输入样本数。

- `--benchmark`
  - benchmark 名称，当前使用 `synthetic`。

- `--models`
  - 可选，逗号分隔模型过滤。
  - 不传时从 `output-root` 自动发现模型目录。

- `--seq-lengths`
  - 长度过滤。
  - 可以是 `4096`、`4096,8192` 或 `all`。

- `--tasks`
  - 任务过滤。
  - 可以是 `all` 或 `niah_single_1,qa_2`。

- `--timing-file`
  - runner 产生的耗时 jsonl。
  - 没有该文件时，分数和 PPL 仍可汇总，但耗时字段为空。

- `--output-file`
  - xlsx 输出文件。

### xlsx 表格说明

- `detail`
  - 最核心的明细表。
  - 每一行唯一对应 `model + length + task`。
  - 包含状态、分数、空预测、样本数、预测行数、PPL、耗时、GPU、预测文件、日志文件。

- `summary_by_model`
  - 每个模型一行。
  - 用于比较模型整体平均分、平均 PPL、任务耗时总和、自然运行时间和完成比例。

- `summary_by_model_and_length`
  - 每个 `model + length` 一行。
  - 用于观察同一模型在不同上下文长度下的表现变化。

- `summary_by_task`
  - 每个任务一行。
  - 用于观察哪些任务更难，以及最高分来自哪个模型和长度。

- `run_info`
  - 记录汇总时间、输入输出根目录、模型列表、长度列表、任务列表和指标口径。

## RULER 原生评分

如果只想使用 RULER 原生评分脚本，对某个模型某个长度的 `pred/` 目录评分：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

输出：

- 多任务：`summary.csv`
- 单任务：`summary-<任务>.csv`
- 提交格式：`submission.csv`

注意：`evaluate.py` 只处理一个 `pred/` 目录，不跨模型、不跨长度，也不统计耗时或 PPL。需要统一汇总时使用 `eval/collect_results.py`。

## 直接调用单任务预测

一般优先使用 `run_parquet_parallel.py`。只有单任务调试时才直接调用 `pred/call_api.py`：

```bash
cd /home/test05/czyprojects/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u pred/call_api.py \
  --data_dir ../benchmark_root/parquet_data/synthetic/4096/data \
  --save_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic \
  --task niah_single_1 \
  --subset validation \
  --server_type hf \
  --model_name_or_path ../../models/Meta-Llama-3.1-8B \
  --temperature 0.0 \
  --top_k 32 \
  --top_p 1.0 \
  --batch_size 1 \
  --max_retries 3 \
  --log_batch_progress \
  --log_generation_ppl
```

常用参数：

- `--data_dir`：包含 `<任务>/<subset>.jsonl` 的数据目录。
- `--save_dir`：预测 jsonl 输出目录。
- `--task`：单个任务名。
- `--subset`：split 名，通常是 `validation`。
- `--server_type hf`：使用本地 Hugging Face 模型。
- `--model_name_or_path`：本地模型路径。
- `--batch_size`：每次送入模型的样本数。
- `--max_retries`：单个 batch 连续失败后的最大重试次数。
- `--log_batch_progress`：输出 batch 级进度。
- `--log_generation_ppl`：写入生成答案 token 的 PPL 字段。

## Attention 诊断命令

### 查看已有 attention dump

你给出的查看命令如下：

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

参数说明：

- `--dump-dir`
  - attention dump 目录。
  - 目录内应包含 `metadata.json`、`prompt_tokens.jsonl`、`generated_tokens.jsonl` 和 `token_XXXX.npy`。

- `--generated-token`
  - 查看第几个生成 token。
  - 从 `0` 开始。

- `--layer`
  - 查看第几层 attention。
  - 从 `0` 开始。

- `--head`
  - 查看第几个 attention head。
  - 从 `0` 开始。

- `--sort-by`
  - 表格排序字段。
  - 可选 `position` 或 `attention`。
  - 默认 `position`。

- `--descending`
  - 降序输出。
  - 和 `--sort-by attention` 配合时，会把 attention 权重从大到小显示。

输出表格字段：

- `position`：上下文 token 位置。
- `source`：token 来源，通常是 `prompt` 或 `generated`。
- `token_id`：token id。
- `token_text`：token 文本。
- `attention`：该 layer/head 对该 token 的 attention 权重。

### 导出一个 Llama 样本的完整 attention

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

常用参数：

- `--model-path`：本地 Llama 模型目录。
- `--data-file`：单个 RULER jsonl 数据文件。
- `--sample-offset`：读取 jsonl 中第几条样本，从 `0` 开始。
- `--output-dir`：attention dump 输出目录。
- `--dtype`：保存 attention 数组的数据类型，常用 `float32` 或 `float16`。
- `--max-new-tokens`：最多生成多少个 token。
- `--overwrite`：输出目录非空时允许覆盖。

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

每个 `token_XXXX.npy` 的 shape 是：

```text
[num_layers, num_heads, key_length]
```

表示第 `XXXX` 个生成 token 作为 query 时，对 prompt 和此前已生成 token 的完整 attention。

## 常见检查命令

查看每个长度是否都有 13 个输入任务：

```bash
cd /home/test05/czyprojects
for d in RULER/benchmark_root/parquet_data/synthetic/[0-9]*; do
  printf '%s ' "$(basename "$d")"
  find "$d/data" -name 'validation.jsonl' | wc -l
done
```

查看某个模型 4k 预测文件：

```bash
cd /home/test05/czyprojects
find RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  -name '*.jsonl' | sort
```

比较输入行数和预测行数：

```bash
cd /home/test05/czyprojects
wc -l RULER/benchmark_root/parquet_data/synthetic/4096/data/*/validation.jsonl
wc -l RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred/*.jsonl
```

## 测试和语法检查

修改代码后至少运行：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python -B -m unittest discover -s tests
conda run -n dl-a800 python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py
```

文档-only 变更至少运行：

```bash
cd /home/test05/czyprojects
git diff --check
git -C RULER diff --check
```

## 注意事项

- 默认不要触发网络下载。
- 不要提交模型权重、benchmark parquet、预测输出、评分输出、attention dump、缓存或日志。
- 长上下文任务很容易 OOM，`--batch-size 1` 是保守默认值。
- `--overwrite-existing` 会删除当前任务旧预测和日志，确认目标任务后再使用。
- `generation_ppl` 衡量模型对自己生成答案的置信程度，不等价于准确率。
- `summary.csv` 是 RULER 原生单目录评分结果；`ruler_results.xlsx` 是本地统一汇总结果。
