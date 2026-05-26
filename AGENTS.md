# 仓库协作规范

## 项目总览

本仓库是一个本地 RULER benchmark 测评工作区，用于把已经下载好的长上下文 benchmark 数据送入本地 Hugging Face 模型，生成预测结果，再用 RULER 原生评分脚本得到分数。当前主流程不是重新生成 RULER 数据，而是复用 `benchmark/RULER-llama3-1M/` 中已经下载好的 parquet 数据。

核心数据流如下：

1. 原始 benchmark：`benchmark/RULER-llama3-1M/<任务>_<长度>/validation-*.parquet`
2. 格式转换：`RULER/scripts/data/prepare_parquet.py`
3. RULER jsonl 输入：`RULER/benchmark_root/parquet_data/synthetic/<长度>/data/<任务>/validation.jsonl`
4. 并行预测：`RULER/scripts/run_parquet_parallel.py` 调度 `RULER/scripts/pred/call_api.py`
5. 预测和日志：`RULER/benchmark_root/local_eval/<模型别名>/synthetic/<长度>/pred/` 和 `RULER/benchmark_root/local_eval/<模型别名>/synthetic/<长度>/logs/`
6. 评分：`RULER/scripts/eval/evaluate.py`
7. 统一汇总：`RULER/scripts/eval/collect_results.py`

截至 2026-05-20，当前工作区已经迁移到新服务器，路径为 `/data/czy/ICLR-2027`。旧说明中的 `/home/test05/czyprojects`、`dl-a800` 和 `ruler-glm44` 不再适用。当前主要 RULER 环境是 conda 环境 `model`，命令示例应优先使用 `conda run --no-capture-output -n model ...`。

当前状态摘要：

- 当前主机名：`dilab`。
- 当前 conda：`/data/czy/miniconda3/bin/conda`。
- 当前 Python：`/data/czy/miniconda3/bin/python`，版本 `Python 3.13.13`。
- 当前可见 conda 环境：`base`、`data`、`model`、`model_download`。
- 当前 RULER 主环境：`/data/czy/miniconda3/envs/model/bin/python`，Python `3.10.20`。
- `model` 环境当前可导入 `torch`、`transformers`、`pyarrow`、`pandas`、`nltk`、`yaml`、`nemo`、`accelerate`、`safetensors`、`sentencepiece` 和 `numpy`。
- `model` 环境当前关键版本：`torch 2.4.1+cu121`、`torch.version.cuda 12.1`、`transformers 4.47.1`、`pyarrow 24.0.0`、`pandas 2.3.3`。
- `model` 环境当前不能导入 `openpyxl`，但本地 `RULER/scripts/eval/collect_results.py` 使用标准库写出 csv，不依赖 `openpyxl`。
- `nvidia-smi` 当前无法和 NVIDIA driver 通信，GPU 状态需要重新确认。
- `model` 环境中 `torch.cuda.is_available()` 当前为 `False`，正式 GPU 推理前仍需修复 NVIDIA driver 或设备节点。
- `RULER/benchmark_root/parquet_data/synthetic/` 当前已经存在，覆盖 9 个长度和 13 个任务，共 117 个 `validation.jsonl` 输入文件。
- `RULER/benchmark_root/local_eval/` 当前已经存在，包含四个模型的 4k 输出目录、52 个任务日志、11 个预测 jsonl、`ruler_timing.jsonl` 和旧的 `ruler_results_4k_all_models.xlsx`；这是一份不完整或曾失败的 4k 运行结果，复用前应检查日志和预测文件完整性。当前新 workflow 的统一汇总输出应使用 csv。
- `benchmark/RULER-llama3-1M/` 当前存在 117 个任务长度目录，覆盖 9 个长度和 13 个 synthetic 任务；共有 118 个 `validation-*.parquet`，其中 `qa_1_1M` 有两个 parquet 分片。
- `ruler_sample_counts_4k_64k.csv` 当前记录了 4k、8k、16k、32k、64k 各任务样本数；这五个长度下 13 个任务均为每任务每长度 500 条。
- 根目录 `README.md` 已按用户要求删除，后续由用户重新整理。
- `RULER/docker/` 已按用户要求删除；当前不再维护上游旧 Docker 模板。

除非用户明确要求，否则不要提交大型模型权重、benchmark parquet 文件、缓存目录、转换后的 jsonl、预测输出、评分输出或运行日志。

## 当前模型目录

当前 `models/` 下可见的模型目录如下：

- `models/Llama-3.1-8B/`
  - 旧环境中曾写作 `models/Meta-Llama-3.1-8B/`，当前目录名已经变化。
  - `config.json` 中 `model_type` 为 `llama`。
  - `max_position_embeddings` 为 `131072`。
- `models/Qwen2.5-7B-Instruct-1M/`
  - 旧环境中曾使用 `models/Qwen3-8B/`，当前 Qwen 模型已经替换。
  - `config.json` 中 `model_type` 为 `qwen2`。
  - `max_position_embeddings` 为 `1010000`。
  - 目录中包含 `sparse_attention_config.json`。
- `models/GLM-4-9B-Chat-1M/`
  - 旧环境中曾使用 `models/glm-4-9b/`，当前 GLM 模型已经替换。
  - `config.json` 中 `model_type` 为 `chatglm`。
  - `seq_length` 为 `1048576`。
  - 目录中包含 `configuration_chatglm.py`、`modeling_chatglm.py`、`tokenization_chatglm.py`，运行时通常需要允许自定义模型代码。
- `models/Yi-9B-200K/`
  - 目录名和旧说明一致。
  - `config.json` 中 `model_type` 为 `llama`。
  - `max_position_embeddings` 为 `262144`。
  - 注意当前 `tokenizer_config.json` 中 `model_max_length` 显示为 `4096`，长上下文运行前需要确认 tokenizer 和模型 wrapper 的实际处理方式。

这些目录通常来自 Hugging Face 模型下载，里面的 `config.json`、`tokenizer_config.json`、`model-*.safetensors`、`model.safetensors.index.json` 等文件由模型仓库提供。不要把模型权重提交进 git。

## 目录与文件说明

### 顶层目录

- `AGENTS.md`
  - 当前协作说明和项目说明文档。
  - 需要让后续维护者能理解目录来源、当前环境状态、测评流程、命令参数、输出位置和协作规则。
- `CODEX_CHANGES.md`
  - Codex 每次变更后的中文变更记录。
  - 只要发生文件变更，就必须更新。
- `models/`
  - 本地模型权重和 tokenizer 文件。
  - 当前模型目录见“当前模型目录”小节。
- `benchmark/RULER-llama3-1M/`
  - 已下载的 RULER parquet benchmark 数据。
  - 目录名形如 `<任务>_<长度>`，例如 `niah_single_1_4k/`、`qa_2_128k/`、`cwe_1M/`。
  - 每个任务长度目录下通常有 `validation-00000-of-00001.parquet`，`qa_1_1M` 当前有两个 parquet 分片。
  - `benchmark/RULER-llama3-1M/README.md` 是 Hugging Face datasets 风格的 dataset metadata。
  - `benchmark/RULER-llama3-1M/.cache/huggingface/` 是下载缓存，不应提交。
- `RULER/`
  - 本地 RULER 项目副本。
  - 当前 `git -C RULER rev-parse --show-toplevel` 指向 `/data/czy/ICLR-2027`，说明它由外层 git 仓库跟踪，不是独立 `.git` 子仓库。
  - 包含上游原生的数据准备、预测、服务和评分脚本。
  - `RULER/docker/` 已删除，不再作为当前工作流的一部分。
- `tests/`
  - 本地回归测试。
  - 覆盖 parquet 转换、并行 runner、`RULER/scripts/pred/call_api.py` batch 进度与失败重试、模型 wrapper 兼容、统一汇总和 attention 工具。
- `tools/`
  - 本地辅助诊断脚本，不属于 RULER 原生流程。
  - `tools/dump_llama_attention.py` 用于对一个 Llama 样本导出生成阶段的完整 attention。
  - `tools/inspect_attention_dump.py` 用于查看某个生成 token、某一层、某个 head 的完整 attention 分布表格。
  - `tools/summarize_bos_attention.py` 用于统计完整 attention dump 中 `<|begin_of_text|>` 在所有生成 token、层和 head 上作为 top-1 attention token 的次数和平均 attention。
  - `tools/compare_pooling_attention.py` 用于对比一个 Llama 样本中 max/avg pooling token block 分数和细粒度 full attention token 分数。
  - `tools/count_ruler_samples.py` 用于统计转换后 RULER jsonl 输入中，4k 到 64k 各长度、各任务的样本数。
- `ruler_sample_counts_4k_64k.csv`
  - 当前工作区 4k、8k、16k、32k、64k 样本数统计输出。
  - 每行对应一个任务，每列对应一个长度，当前 13 个任务每个长度均为 500 条。

### RULER 关键目录和脚本

- `RULER/README.md`
  - RULER 上游项目说明，当前保留。
  - 根目录 `README.md` 已删除，两者不要混淆。
- `RULER/scripts/synthetic.yaml`
  - synthetic benchmark 的 13 个任务配置。
  - 当前 runner 的 `--tasks all` 对应这些任务：`niah_single_1`、`niah_single_2`、`niah_single_3`、`niah_multikey_1`、`niah_multikey_2`、`niah_multikey_3`、`niah_multivalue`、`niah_multiquery`、`vt`、`cwe`、`fwe`、`qa_1`、`qa_2`。
- `RULER/scripts/data/prepare.py`
  - RULER 上游原生数据生成入口，用于从任务配置和 tokenizer 重新生成 synthetic 样本。
  - 当前主流程不用它，因为本地已经有 parquet benchmark。
- `RULER/scripts/data/prepare_parquet.py`
  - 本地新增或适配的 parquet 转 jsonl 脚本。
  - 读取 `benchmark/RULER-llama3-1M/` 的 parquet，写出 `RULER/scripts/pred/call_api.py` 能直接读取的 RULER jsonl。
  - 不重新生成 prompt，不改模型模板，只做字段转换：`answers` 会转换成 RULER 预测脚本需要的 `outputs`。
- `RULER/scripts/run_parquet_parallel.py`
  - 本地并行预测调度器。
  - 按模型、长度和任务展开任务矩阵，为每个任务启动一个 `RULER/scripts/pred/call_api.py` 子进程。
  - 通过 `CUDA_VISIBLE_DEVICES=<gpu>` 让每个子进程只看到一张 GPU，避免一个本地 HF 模型自动占满全部 GPU。
- `RULER/scripts/pred/call_api.py`
  - RULER 原生预测入口。
  - 读取 `--data_dir/<task>/<subset>.jsonl`，加载指定模型或服务客户端，写出带 `pred` 字段的预测 jsonl。
  - 本地增加了 `--log_batch_progress`、`--max_retries`、`--log_generation_ppl`、`--log_generation_token_ppl`、`--log_prefill_decode_timing` 和 `--profile_attention_kernels`，用于观察进度、避免单个样本无限重试，并在 Hugging Face 生成阶段记录 PPL、prefill/decode 耗时和第 0 行样本的 attention kernel profiler。
- `RULER/scripts/pred/model_wrappers.py`
  - Hugging Face、本地模型和 Mamba wrapper。
  - 本地包含 GLM 配置兼容逻辑。
  - 本地 Hugging Face wrapper 支持基于 `generate(..., output_scores=True)` 的生成答案 token PPL 统计，并可额外返回每个生成 token 的 logprob、NLL 和 PPL 明细。
  - 本地 Hugging Face wrapper 可用 CUDA event 记录每条样本的 prefill/decode forward 耗时；也可用 `torch.profiler` 对每个任务第 0 行样本统计严格 attention CUDA kernel 时间。
- `RULER/scripts/eval/evaluate.py`
  - RULER 原生评分入口。
  - 读取预测 jsonl，按 `RULER/scripts/eval/synthetic/constants.py` 中的 metric 计算每个任务分数。
- `RULER/scripts/eval/collect_results.py`
  - 本地统一汇总脚本。
  - 跨模型、长度和任务读取预测 jsonl、生成阶段 PPL 字段、`<任务>.generation_timing.jsonl` 和 runner timing jsonl。
  - 输出一组 csv 文件：主文件 `ruler_results.csv` 为 `detail` 明细，其余表写到同名前缀的 `summary_by_model`、`summary_by_model_and_length`、`summary_by_task` 和 `run_info` csv 文件。
- `RULER/benchmark_root/parquet_data/synthetic/`
  - `RULER/scripts/data/prepare_parquet.py` 生成的 RULER jsonl 输入数据。
  - 当前新服务器上该目录已经存在，包含 117 个 `validation.jsonl` 输入文件。
- `RULER/benchmark_root/local_eval/`
  - `RULER/scripts/run_parquet_parallel.py` 生成的预测、日志和后续评分结果。
  - 当前新服务器上该目录已经存在，但 4k 四模型运行结果不完整；继续实验前应按任务日志和预测行数检查是否需要 `--skip-existing` 续跑或 `--overwrite-existing` 覆盖重跑。

## 数据来源与格式

`benchmark/RULER-llama3-1M/` 是已经下载好的 parquet benchmark。它的字段包括：

- `index`：样本编号。
- `input`：已经构造好的模型输入 prompt。
- `answers`：参考答案列表。
- `length`：样本目标上下文长度。
- `predictions`：数据集自带的其他模型预测字段，当前本地测评流程不依赖它。

`RULER/scripts/data/prepare_parquet.py` 会把每条 parquet 样本转换为：

- `index`：沿用原始编号。
- `input`：沿用原始 prompt。
- `outputs`：由 `answers` 转换而来，供 RULER evaluator 作为参考答案。
- `others`：默认空字典。
- `truncation`：默认 `-1`。
- `length`：沿用原始长度。

预测完成后，`RULER/scripts/pred/call_api.py` 会额外写入：

- `pred`：模型生成的答案文本。
- `generation_logprob_sum`：开启 `--log_generation_ppl` 后写入，表示生成答案 token 的 log probability 总和。
- `generation_token_count`：开启 `--log_generation_ppl` 后写入，表示参与 PPL 计算的生成 token 数。
- `generation_nll`：开启 `--log_generation_ppl` 后写入，计算方式为 `-generation_logprob_sum / generation_token_count`。
- `generation_ppl`：开启 `--log_generation_ppl` 后写入，计算方式为 `exp(generation_nll)`。

如果开启 `--log_generation_token_ppl`，`RULER/scripts/pred/call_api.py` 会额外写出 sidecar 文件 `<任务>.generation_tokens.jsonl`。每行对应一个样本，包含 `index`、`task`、`generation_token_count` 和 `tokens`；`tokens` 中每个元素记录生成 token 的 `position`、`token_id`、`token`、`logprob`、`nll` 和 `ppl`。该 token 级 PPL 仍然是模型对自己生成答案 token 的困惑度，不是参考答案困惑度。

如果开启 `--log_prefill_decode_timing` 或 `--profile_attention_kernels`，`RULER/scripts/pred/call_api.py` 会额外写出 sidecar 文件 `<任务>.generation_timing.jsonl`。其中 `record_type=sample_timing` 每条样本一行，记录 `sample_line_no`、`sample_index`、`prefill_forward_ms`、`decode_forward_ms_total`、`decode_forward_ms_per_token_avg` 和 `generated_token_count`；`record_type=attention_profile` 每个任务最多一行，固定采输入 jsonl 第 0 行，记录 `sample_line_no=0`、原始 `sample_index`、`prefill_attention_kernel_ms`、`decode_attention_kernel_ms_total`、`decode_attention_kernel_ms_per_token_avg` 和 `attention_kernel_event_count`。`prefill/decode forward` 耗时是完整 forward 阶段耗时；`attention_kernel` 字段才是 profiler 按 kernel 名称过滤后的严格 GPU attention 时间。

评分时，`RULER/scripts/eval/evaluate.py` 会比较 `pred` 和 `outputs`，并统计空预测数量。

## 当前环境状态与剩余阻塞

当前已经有可用于脚本转换、统计、dry-run 和部分 CPU 侧检查的 RULER 环境 `model`。开始正式 GPU 推理前，仍需处理 GPU driver 或设备节点问题：

1. `conda run --no-capture-output -n model python ...` 是当前默认命令前缀。
2. `model` 环境已经安装基础依赖，能导入 `torch`、`transformers`、`pyarrow`、`pandas`、`nltk`、`yaml`、`nemo` 等。
3. `RULER/benchmark_root/parquet_data/synthetic/` 已经转换完成，通常不需要重新转换。
4. `nvidia-smi` 当前仍不能和 NVIDIA driver 通信，`torch.cuda.is_available()` 当前为 `False`。正式推理前必须先让 CUDA 可用。
5. 当前 4k 四模型运行目录已有部分输出和日志，但结果不完整，继续跑时根据目标选择 `--skip-existing` 续跑或 `--overwrite-existing` 干净重跑。

旧命令中所有 `conda run -n dl-a800 ...`、`conda run -n ruler-glm44 ...`、`conda run -n <ruler-env> ...`、`/home/test05/czyprojects`、`/home/test05/miniconda3/...` 路径都需要替换。当前文档示例统一使用 `/data/czy/ICLR-2027` 和 conda 环境 `model`。

## 推荐测评流程

### 1. 确认环境

当前默认使用 `model` 环境。正式推理前，先运行：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python -c "import torch, transformers, pyarrow, pandas, nltk, yaml, nemo; print('ok')"
conda run --no-capture-output -n model python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

如果 GLM 或其他模型需要独立环境，可以继续使用 `RULER/scripts/run_parquet_parallel.py` 的 `--model-python NAME=PYTHON` 机制，但不要再假设旧的 `ruler-glm44` 环境存在。

### 2. 将 parquet 转换成 RULER jsonl

当前 `RULER/benchmark_root/parquet_data/synthetic/` 已经存在，通常不需要重新转换。只有原始 parquet 更新、目标 jsonl 缺失，或需要重新生成输入时才运行转换。

转换全部数据：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python RULER/scripts/data/prepare_parquet.py
```

只转换一个任务和一个长度，适合检查流程：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python RULER/scripts/data/prepare_parquet.py \
  --tasks niah_single_1 \
  --lengths 4k \
  --max_samples 5 \
  --skip_existing
```

`RULER/scripts/data/prepare_parquet.py` 参数说明：

- `--parquet_data_dir`：原始 parquet benchmark 根目录，默认是 `benchmark/RULER-llama3-1M`。
- `--save_dir`：转换后 jsonl 的保存根目录，默认是 `RULER/benchmark_root/parquet_data/synthetic`。
- `--subset`：要转换的数据 split，默认 `validation`，会匹配 `validation-*.parquet`。
- `--tasks`：逗号分隔的任务过滤列表，例如 `niah_single_1,qa_2`；不传表示转换全部任务。
- `--lengths`：逗号分隔的长度过滤列表，支持 `4k,128k,1M` 或 `4096,131072`；不传表示转换全部长度。
- `--max_samples`：每个任务长度最多写出多少条样本；适合 smoke test，不传表示写出全部样本。
- `--skip_existing`：目标 jsonl 已存在时跳过，避免覆盖已有转换结果。
- `--report_file`：转换报告路径；默认写到 `save_dir/conversion_report.json`。

### 3. 先 dry-run 检查并行预测命令

进入 RULER scripts 目录后运行 dry-run：

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
  --dry-run
```

dry-run 只打印任务矩阵和将要执行的 `RULER/scripts/pred/call_api.py` 命令，不启动模型推理。

`RULER/scripts/run_parquet_parallel.py` 参数说明：

- `--model NAME=PATH`：模型别名和模型路径；可以重复传入多个模型。`NAME` 会成为输出目录名。
- `--seq-lengths`：要评测的长度。可以传 `4096`、`4096,8192`，也可以传 `all` 自动发现 `--data-root` 下的全部数字长度目录。
- `--tasks`：要评测的任务。可以传 `niah_single_1,qa_2`，也可以传 `all` 使用 13 个默认 synthetic 任务。
- `--gpus`：物理 GPU 编号列表，例如 `0,1,2,3`。runner 会给每个子进程设置对应的 `CUDA_VISIBLE_DEVICES`。
- `--max-workers`：最多同时运行多少个子进程；默认等于 GPU 数量。
- `--data-root`：转换后 jsonl 数据根目录，默认 `RULER/benchmark_root/parquet_data/synthetic`。
- `--output-root`：预测、日志和评分输出根目录，默认 `RULER/benchmark_root/local_eval`。
- `--benchmark`：RULER benchmark 名称，当前使用 `synthetic`。
- `--subset`：读取的数据 split，默认 `validation`。
- `--server-type`：传给 `RULER/scripts/pred/call_api.py` 的模型后端类型。跑本地 Hugging Face 权重时使用 `hf`。
- `--python`：启动 `RULER/scripts/pred/call_api.py` 的 Python 解释器，默认是当前解释器。
- `--model-python NAME=PYTHON`：按模型别名指定子进程 Python，可重复传入。未指定的模型继续使用全局 `--python`。
- `--temperature`：采样温度。默认 `0.0`，表示确定性生成。
- `--top-k`：top-k 采样参数，默认 `32`。
- `--top-p`：top-p 采样参数，默认 `1.0`。
- `--random-seed`：随机种子，默认 `0`。
- `--stop-words`：逗号分隔的停止词，会透传给模型 wrapper。
- `--batch-size`：当前固定为 `1`。传入任何非 `1` 的值都会报错；runner 也始终向 `RULER/scripts/pred/call_api.py` 传 `--batch_size 1`。
- `--poll-interval`：runner 检查子进程状态和打印进度的间隔秒数，默认 `10`。
- `--skip-existing`：如果目标预测文件已存在，则跳过该任务。
- `--overwrite-existing`：启动任务前删除对应任务已有的预测 jsonl、attention 摘要文件、生成 token PPL 明细文件、生成 timing 明细文件和日志文件，强制重新生成。它不能和 `--skip-existing` 同时使用。
- `--log-batch-progress`：让子进程输出 `[BATCH_START]`、`[BATCH_DONE]`、`[BATCH_FAILED]`，runner 会把这些行同步回显并写入日志。
- `--log-generation-ppl`：让 Hugging Face 子进程在生成阶段写入 `generation_logprob_sum`、`generation_token_count`、`generation_nll` 和 `generation_ppl`。该 PPL 是模型对自己生成答案 token 的困惑度，不是参考答案困惑度。
- `--log-generation-token-ppl`：让 Hugging Face 子进程额外写出 `<任务>.generation_tokens.jsonl`，记录每个生成 token 的 `logprob`、`nll` 和 `ppl`；该参数会自动启用样本级 `--log-generation-ppl`。
- `--log-prefill-decode-timing`：让 Hugging Face 子进程额外写出 `<任务>.generation_timing.jsonl`，每条样本记录一次 `sample_timing`，包含 prefill/decode forward 阶段耗时。
- `--profile-attention-kernels`：让 Hugging Face 子进程对每个任务输入 jsonl 第 0 行额外运行一次 `torch.profiler`，写出一条 `attention_profile`，统计严格 attention CUDA kernel 时间。该参数不能和 `--log-attention-scores` 同时使用。
- `--attention-profile-sample-offset`：当前固定为 `0`，表示只 profile 输入 jsonl 第 0 行；传入其他值会报错。
- `--mask-bos-token`：仅支持 `--server-type hf`。保留 tokenizer 自动插入的 BOS token，例如 Llama 的 `<|begin_of_text|>`，但把 position 0 的 `attention_mask` 置为 0，并显式保留原始 `position_ids`，用于对比“遮住整段文本起点 token”和正常输入的 RULER 分数差异。开启后预测 jsonl 会保留 `attention_mask_ablation` 元信息。
- `--timing-file`：结构化任务耗时 jsonl；默认写到 `RULER/benchmark_root/local_eval/ruler_timing.jsonl`。
- `--auto-evaluate`：全部子任务结束后调用 `RULER/scripts/eval/collect_results.py` 生成统一 csv 汇总。
- `--report-file`：统一汇总 csv 主输出路径，必须使用 `.csv` 后缀；默认写到 `RULER/benchmark_root/local_eval/ruler_results.csv`。汇总脚本还会写出同名前缀的 summary 和 run_info csv 文件。
- `--dry-run`：只打印命令，不启动推理。

### 4. 正式运行预测

确认 `model` 环境、转换后 jsonl 和 GPU 可用后，再运行预测。

跑单模型、单长度、单任务：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python -B run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress
```

跑当前四个模型的 4k 全任务时，模型路径应使用当前目录名：

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
  --report-file ../benchmark_root/local_eval/ruler_results_4k_all_models.csv \
  --skip-existing
```

当前 workflow 已经固定使用 `--batch-size 1`。如果要覆盖旧结果，应删除 `--skip-existing` 并改用 `--overwrite-existing`。

如果要在预测时记录每个生成 token 的困惑度，应加上 `--log-generation-token-ppl`。该参数会同时保留主预测 jsonl 中的样本级 PPL 字段，并额外为每个任务写出 `<任务>.generation_tokens.jsonl`。

如果要记录每条样本的 prefill/decode forward 耗时和每个任务第 0 行样本的严格 attention kernel 时间，应同时加上 `--log-prefill-decode-timing --profile-attention-kernels`。普通阶段耗时会覆盖每条样本；严格 profiler 固定每个 `model + length + task` 只采输入 jsonl 第 0 行。

如果要跑遮住 `<|begin_of_text|>` 的 RULER 4k 全任务实验，可以给 Llama 单独起一个新的模型别名，并加上 `--mask-bos-token`：

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

如果某个模型需要独立 Python，使用 `--model-python`：

```bash
  --python /data/czy/miniconda3/envs/model/bin/python \
  --model-python GLM-4-9B-Chat-1M=/data/czy/miniconda3/envs/<glm-env>/bin/python
```

如果需要重新跑已经生成过的任务，不要传 `--skip-existing`，改传 `--overwrite-existing`。该参数会在任务启动前删除对应 `<任务>.jsonl`、`<任务>.attention.jsonl`、`<任务>.attention.md`、`<任务>.generation_tokens.jsonl`、`<任务>.generation_timing.jsonl` 和 `<任务>.log`，然后让 `pred/call_api.py` 从头生成。

### 5. 直接调用 call_api.py 的场景

一般优先用 `RULER/scripts/run_parquet_parallel.py`。只有需要单独调试某个任务、修改 `--max_retries`，或绕开 runner 时才直接调用 `RULER/scripts/pred/call_api.py`：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n model python -u pred/call_api.py \
  --data_dir ../benchmark_root/parquet_data/synthetic/4096/data \
  --save_dir ../benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic \
  --task niah_single_1 \
  --subset validation \
  --server_type hf \
  --model_name_or_path ../../models/Llama-3.1-8B \
  --temperature 0.0 \
  --top_k 32 \
  --top_p 1.0 \
  --batch_size 1 \
  --max_retries 3 \
  --log_batch_progress
```

`RULER/scripts/pred/call_api.py` 常用参数说明：

- `--data_dir`：包含 `<任务>/<subset>.jsonl` 的数据目录，例如 `../benchmark_root/parquet_data/synthetic/4096/data`。
- `--save_dir`：预测 jsonl 保存目录。
- `--benchmark`：benchmark 名称，当前为 `synthetic`。
- `--task`：任务名，例如 `niah_single_1`。
- `--subset`：split 名，默认 `validation`。
- `--server_type`：后端类型，可选 `trtllm`、`vllm`、`sglang`、`openai`、`gemini`、`hf`、`mamba`。
- `--model_name_or_path`：模型名称、API 模型名或本地模型路径。
- `--temperature`、`--top_k`、`--top_p`、`--random_seed`：生成采样参数。
- `--stop_words`：逗号分隔停止词。
- `--threads`：并发线程数。`hf` 和 `gemini` 会被脚本强制设为 `1`。
- `--batch_size`：当前固定为 `1`；传入任何非 `1` 的值都会报错。
- `--log_batch_progress`：输出 batch 开始、完成和失败日志。
- `--max_retries`：单个 batch 失败后的最大重试次数，默认 `3`。超过后进程非零退出，runner 会释放该 GPU 并记录失败。
- `--log_generation_ppl`：仅支持 `--server_type hf`。开启后在每条预测 jsonl 记录中追加生成答案 token 的 PPL 统计字段。
- `--log_generation_token_ppl`：仅支持 `--server_type hf`。开启后额外写出 `<任务>.generation_tokens.jsonl`，记录每个生成 token 的 `logprob`、`nll` 和 `ppl`，并自动启用 `--log_generation_ppl`。
- `--log_prefill_decode_timing`：仅支持 `--server_type hf`。开启后额外写出 `<任务>.generation_timing.jsonl` 的 `sample_timing` 记录，每条样本一行。
- `--profile_attention_kernels`：仅支持 `--server_type hf`。开启后只对输入 jsonl 第 0 行额外写一条 `attention_profile` 记录，用 `torch.profiler` 统计 attention CUDA kernel 时间。
- `--attention_profile_sample_offset`：当前固定为 `0`；传入其他值会报错。
- `--mask_bos_token`：仅支持 `--server_type hf`。保留 BOS token，但把 position 0 的 `attention_mask` 置为 0；适合单独调试遮住 `<|begin_of_text|>` 的一项任务。

### 6. 运行评分和统一测评汇总

对一个模型、一个长度的全部预测评分：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run --no-capture-output -n model python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

只要 `pred/` 目录里有对应任务的 `<任务>.jsonl`，`RULER/scripts/eval/evaluate.py` 就会评这些任务。缺失的任务会打印 `Prediction file <任务>.jsonl is not found.` 并跳过。

生成跨模型、跨长度、跨任务的统一 csv 汇总：

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

`ruler_results_4k_all_models.csv` 是 `detail` 明细主表，汇总脚本还会生成同名前缀的四个 csv：

- `ruler_results_4k_all_models_summary_by_model.csv`：每个模型一行，包含完成数量、平均分、平均 PPL、`total_task_elapsed_seconds`、`wall_time_seconds`、prefill/decode forward 聚合耗时、attention profiler 聚合字段和样本总数。
- `ruler_results_4k_all_models_summary_by_model_and_length.csv`：每个 `model + length` 一行，按长度观察分数、PPL、任务耗时、prefill/decode forward 耗时和 attention profiler 指标变化。
- `ruler_results_4k_all_models_summary_by_task.csv`：每个任务一行，记录平均分、最高分对应模型和长度、平均 PPL、平均耗时和 timing/profiler 聚合字段。
- `ruler_results_4k_all_models_run_info.csv`：记录汇总时间、输入目录、模型/长度/任务列表和 score/PPL/time 口径。

### 7. 检查已有数据和结果

查看每个长度是否都有 13 个输入任务：

```bash
cd /data/czy/ICLR-2027
for d in RULER/benchmark_root/parquet_data/synthetic/[0-9]*; do
  printf '%s ' "$(basename "$d")"
  find "$d/data" -name 'validation.jsonl' | wc -l
done
```

查看某个模型 4k 预测是否齐全：

```bash
cd /data/czy/ICLR-2027
find RULER/benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred \
  -name '*.jsonl' | sort
```

查看预测行数是否等于输入行数：

```bash
cd /data/czy/ICLR-2027
wc -l RULER/benchmark_root/parquet_data/synthetic/4096/data/*/validation.jsonl
wc -l RULER/benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred/*.jsonl
```

查看 4k 到 64k 每个任务有多少条输入样本：

```bash
cd /data/czy/ICLR-2027
python -B tools/count_ruler_samples.py
python -B tools/count_ruler_samples.py --format csv > ruler_sample_counts_4k_64k.csv
```

### 8. Llama 单样本完整 attention 导出

`tools/dump_llama_attention.py` 是独立诊断脚本，不接入 `RULER/scripts/pred/call_api.py`，也不修改 benchmark 预测输出。它适合只观察一个 Llama 模型在一个 RULER 样本上的生成过程：先生成回答 token，再用 KV cache replay 逐个导出每个生成 token 的完整 attention。

典型运行命令：

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n model python -u tools/dump_llama_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --output-dir attention_dumps/llama_niah_single_1_sample_0 \
  --dtype float32 \
  --max-new-tokens 128 \
  --overwrite
```

如果要保留 `<|begin_of_text|>` token 但禁止生成和 replay 阶段 attend 到它，用于对比 BOS token 作为 attention sink 的影响，可以额外加：

```bash
  --mask-bos-token
```

查看第 0 个生成 token、第 0 层、第 0 个 head 的完整分布：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0 \
  --sort-by attention \
  --descending
```

统计 `<|begin_of_text|>` 在所有 `generated token × layer × head` 组合中 attention 是否排第一，以及它的平均 attention：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python tools/summarize_bos_attention.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0
```

`tools/summarize_bos_attention.py` 默认从 `prompt_tokens.jsonl` 中按 `token_text=<|begin_of_text|>` 定位 BOS token；如果需要手动指定位置，可以额外传 `--bos-position 0`。脚本只打印汇总，不写新的 dump 文件。

这个工具默认用于本地 Llama 诊断场景，不保证 GLM、Qwen、Yi 的自定义模型代码可直接复用。完整 attention 文件会比较大，长上下文或较多生成 token 时需要提前确认磁盘空间。

### 9. Llama pooling token 与细粒度 attention 对照

`tools/compare_pooling_attention.py` 是独立诊断脚本，不接入 `RULER/scripts/pred/call_api.py`，也不修改 benchmark 预测输出。它适合检查 top-k block 选择中 pooling token 是否能代表其覆盖的多个原始 token：脚本会对一个样本生成回答，replay 指定生成 token 的 full attention，然后按固定 block size 输出 max pooling、avg pooling 的 block 级分数，以及 block 内每个细粒度 token 在 full attention 中的真实注意力分数。

典型运行命令：

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n model python -u tools/compare_pooling_attention.py \
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

如果要跑遮住 `<|begin_of_text|>` 的 pooling 对照实验，可以额外加 `--mask-bos-token`。该参数不会从 `input_ids` 删除 BOS token，而是将它在 prompt 的 `attention_mask` 置为 0，并在生成和 replay attention 时保留原始 position ids，从而只测试“不能 attend 到 BOS”这一项变化。

主要输出文件：

- `pooling_tokens.jsonl`：每行一个 pooling token/block，包含 `pooling_score_max`、`pooling_score_avg`、`full_attention_sum`、`rank_by_max` 和 `rank_by_avg` 等字段。
- `fine_tokens.jsonl`：每行一个原始 prompt token，包含所属 `block_id`、token 文本和 full attention 分数；默认还包含每层每 head 明细。
- `pooling_vs_fine_summary.jsonl`：每行一个 block，把 pooling 分数和该 block 覆盖的原始 token attention 分数放在一起，便于直接对照。
- `attention_detail.npz`：压缩保存完整 `[num_layers, num_heads, key_position]` attention 数组。

开启 `--mask-bos-token` 后，`metadata.json` 会记录 `attention_mask_ablation`，prompt token 明细中 position 0 会带有 `attention_mask=0` 和 `masked=true`。

如果不想在 jsonl 中写入每层每 head 的长明细，可以加 `--omit-layer-head-details`，此时完整矩阵仍保留在 `attention_detail.npz` 中。

## 任务含义速查

- `niah_single_1`、`niah_single_2`、`niah_single_3`：Needle-in-a-haystack 单 key 检索，不同 haystack 和 value 类型。
- `niah_multikey_1`、`niah_multikey_2`、`niah_multikey_3`：多 key 检索，干扰更强。
- `niah_multivalue`：一个 key 对应多个 value。
- `niah_multiquery`：一次询问多个 query。
- `vt`：variable tracking，多跳变量绑定追踪。
- `cwe`：common words extraction，找常见词。
- `fwe`：frequent words extraction，找高频词。
- `qa_1`：基于 SQuAD 的问答任务。
- `qa_2`：基于 HotpotQA 的问答任务。

长度目录使用 token 数表示：

- `4096` = 4k
- `8192` = 8k
- `16384` = 16k
- `32768` = 32k
- `65536` = 64k
- `131072` = 128k
- `262144` = 256k
- `524288` = 512k
- `1048576` = 1M

## 构建、测试与开发命令

本项目没有构建步骤。当前默认使用 `model` 环境运行测试和脚本检查。

修改 RULER parquet 转换、并行 runner、`RULER/scripts/pred/call_api.py` 或模型 wrapper 后，至少运行：

```bash
cd /data/czy/ICLR-2027
conda run --no-capture-output -n model python -B -m unittest discover -s tests
conda run --no-capture-output -n model python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py \
  tools/count_ruler_samples.py
```

再运行一个 runner dry-run：

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
  --dry-run
```

文档-only 变更至少运行：

```bash
git diff --check
git -C RULER diff --check
```

如果新增测试，请放在 `tests/` 目录下，并将文件命名为 `test_*.py`。测试重点应覆盖 prompt 转换、任务发现、parquet 加载、runner 命令构造、进度日志和评分逻辑。

## 代码风格与命名规范

使用 Python 3，采用 4 空格缩进。适合添加类型标注时应添加类型标注。解析、加载、评分、生成等逻辑应拆分为小型辅助函数。文件系统路径优先使用 `pathlib.Path`，命令行接口优先使用 `argparse`。

脚本文件名应使用小写字母和下划线，例如 `run_llama_ruler_4k.py`。

默认行为应保持本地优先：正常运行时避免网络访问。除非用户明确要求下载，否则保留 `local_files_only=True` 这类行为。

公共函数和类必须包含 docstring。对于不容易一眼看懂的逻辑，需要添加清晰注释，尤其是 prompt 转换、评分、batching、模型加载等决策点。不要添加无意义注释，例如“递增 i”，也不要添加只是重复代码字面含义的注释。

本项目中的所有代码注释、docstring、解释文档、变更说明和面向开发者的说明文件必须使用中文。命令、路径、Python API 名称、模型名、benchmark 名称和必要的英文专有名词可以保留原文。

## 提交与 Pull Request 规范

当前 git 历史较少，因此提交信息应简短、使用祈使句，并清楚描述动作，例如：

- `Add RULER parquet runner`
- `Fix RULER scoring output`
- `Document local RULER workflow`

Pull Request 应说明本次变更、列出运行过的命令、说明影响到的模型或数据集，并注明是否有意更新了生成输出。

## Agent 专用说明

以后每次开始 coding 前，默认应先使用子代理做并行辅助工作。适合的默认方式是：先快速判断当前任务中哪些检查、定位、测试设计或小范围实现可以独立并行，再把这些工作交给子代理；主代理继续处理当前关键路径。只有在任务非常小、用户明确要求不要使用子代理、或当前环境没有可用子代理能力时，才可以不使用，并需要在回复中简单说明原因。

每次文件变更后，都必须创建或更新 `CODEX_CHANGES.md`。该文件必须说明：

- 哪些文件发生了变更。
- 每个变更文件的目的。
- 新增或修改的主要函数或类。
- 如何运行代码。
- 如何测试或验证变更。
- 假设和限制。

结束工作前，必须检查注释和 docstring 是否足够清晰，并确认 `CODEX_CHANGES.md` 是最新的。如果没有发生代码变更，需要明确说明这一点，不要创建误导性的变更记录。

`CODEX_CHANGES.md` 必须使用中文编写。

检查工作区状态时，应同时运行外层状态命令和 `RULER/` 路径下的状态命令。当前 `RULER/` 由外层 git 跟踪，因此第二条命令会显示同一个外层仓库的相对路径；如果以后换成独立子仓库，这个习惯也能及时暴露子仓库状态：

```bash
git status --short
git -C RULER status --short
```

不要提交 `__pycache__/`、`.pyc`、模型权重、benchmark 原始 parquet、预测输出、评分输出或运行日志，除非用户明确要求。如果运行命令新产生了 `__pycache__/` 或 `.pyc`，结束前应清理或确认它们仍被忽略。

## 安全与配置建议

模型目录和 benchmark 数据应视为大型本地资产。不要提交凭据、访问 token、下载缓存，或未在文档示例中说明的环境相关路径。

默认不要触发网络下载。需要新增模型或数据集时，应先明确下载来源、保存目录、磁盘占用、许可证限制和是否会进入 git 跟踪。
