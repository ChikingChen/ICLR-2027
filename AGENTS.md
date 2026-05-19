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
7. 评分输出：对应 `pred/` 目录下的 `summary.csv`、`summary-<任务>.csv` 和 `submission.csv`

截至 2026-05-13，本地可见的模型目录包括 `models/Meta-Llama-3.1-8B/`、`models/Qwen3-8B/`、`models/Yi-9B-200K/` 和 `models/glm-4-9b/`。本地已转换的 RULER jsonl 数据覆盖 9 个长度：`4096`、`8192`、`16384`、`32768`、`65536`、`131072`、`262144`、`524288`、`1048576`，每个长度下有 13 个 synthetic 任务，共 117 个 `validation.jsonl`。当前 `RULER/benchmark_root/local_eval/` 下已有 4k 预测结果：`Meta-Llama-3.1-8B`、`Qwen3-8B`、`Yi-9B-200K` 各 13 个任务，`glm-4-9b` 有 3 个任务；当前未发现 `summary*.csv` 评分文件。

除非用户明确要求，否则不要提交大型模型权重、benchmark parquet 文件、缓存目录或生成输出。

## 目录与文件说明

### 顶层目录

- `AGENTS.md`
  - 当前协作说明和项目说明文档。
  - 需要让后续维护者能理解目录来源、测评流程、命令参数、输出位置和协作规则。
- `CODEX_CHANGES.md`
  - Codex 每次变更后的中文变更记录。
  - 只要发生文件变更，就必须更新。
- `models/`
  - 本地模型权重和 tokenizer 文件。
  - 当前包含 `models/Meta-Llama-3.1-8B/`、`models/Qwen3-8B/`、`models/Yi-9B-200K/`、`models/glm-4-9b/`。
  - 这些目录通常来自 Hugging Face 模型下载，里面的 `config.json`、`tokenizer_config.json`、`model-*.safetensors`、`model.safetensors.index.json` 等文件由模型仓库提供。
  - `models/glm-4-9b/` 还包含 `configuration_chatglm.py`、`modeling_chatglm.py`、`tokenization_chatglm.py` 这类自定义模型代码，运行时可能需要 `trust_remote_code` 类支持。
- `benchmark/RULER-llama3-1M/`
  - 已下载的 RULER parquet benchmark 数据。
  - 目录名形如 `<任务>_<长度>`，例如 `niah_single_1_4k/`、`qa_2_128k/`、`cwe_1M/`。
  - 每个任务长度目录下通常有 `validation-00000-of-00001.parquet`，少数任务可能有多个 parquet 分片。
  - `benchmark/RULER-llama3-1M/README.md` 是 Hugging Face datasets 风格的 dataset metadata，记录字段、split、样本数和数据大小。
  - `benchmark/RULER-llama3-1M/.cache/huggingface/` 是下载缓存，不应提交。
- `RULER/`
  - 本地 RULER 项目副本。
  - 当前工作区中 `git -C RULER rev-parse --show-toplevel` 指向外层 `/home/test05/czyprojects`，说明它现在由外层 git 仓库跟踪，不是独立 `.git` 子仓库。
  - 包含上游原生的数据准备、预测、服务和评分脚本。
  - 本仓库对其中部分脚本做了本地适配，尤其是 parquet 转换、并行调度、Hugging Face wrapper 兼容和 batch 进度日志。
- `tests/`
  - 本地回归测试。
  - 覆盖 parquet 转换、并行 runner、`RULER/scripts/pred/call_api.py` batch 进度与失败重试、模型 wrapper 兼容等逻辑。
- `tools/`
  - 本地辅助诊断脚本，不属于 RULER 原生流程。
  - `tools/dump_llama_attention.py` 用于对一个 Llama 样本导出生成阶段的完整 attention。
  - `tools/inspect_attention_dump.py` 用于查看某个生成 token、某一层、某个 head 的完整 attention 分布表格。

### RULER 关键目录和脚本

- `RULER/README.md`
  - RULER 上游项目说明，介绍 RULER 论文、任务类型、原始运行方式和可配置任务复杂度。
  - 当前本地工作流主要复用其中的预测和评分脚本，不完全使用上游 `run.sh` 流程。
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
  - 某个 GPU 上任务结束后，会自动补下一个任务。
- `RULER/scripts/pred/call_api.py`
  - RULER 原生预测入口。
  - 读取 `--data_dir/<task>/<subset>.jsonl`，加载指定模型或服务客户端，写出带 `pred` 字段的预测 jsonl。
  - 本地增加了 `--log_batch_progress`、`--max_retries` 和 `--log_generation_ppl`，用于观察 batch 进度、避免单个 batch 无限重试，并在 Hugging Face 生成阶段记录生成 token PPL。
- `RULER/scripts/pred/model_wrappers.py`
  - Hugging Face、本地模型和 Mamba wrapper。
  - 本地包含 GLM 配置兼容逻辑，例如给只有 `num_layers` 的配置补 `num_hidden_layers` 别名。
  - 本地 Hugging Face wrapper 支持基于 `generate(..., output_scores=True)` 的生成答案 token PPL 统计。
- `RULER/scripts/pred/client_wrappers.py`
  - OpenAI、Gemini、vLLM、TensorRT-LLM、SGLang 等服务端客户端 wrapper。
  - 当前本地模型主流程通常使用 `--server-type hf`，不会走远程 API。
- `RULER/scripts/eval/evaluate.py`
  - RULER 原生评分入口。
  - 读取预测 jsonl，按 `RULER/scripts/eval/synthetic/constants.py` 中的 metric 计算每个任务分数。
  - 多个任务时写 `summary.csv`，单个任务时写 `summary-<任务>.csv`，同时写 `submission.csv`。
- `RULER/scripts/eval/collect_results.py`
  - 本地统一汇总脚本。
  - 跨模型、长度和任务读取预测 jsonl、生成阶段 PPL 字段和 runner timing jsonl。
  - 输出单个 `ruler_results.xlsx`，包含 `detail`、`summary_by_model`、`summary_by_model_and_length`、`summary_by_task` 和 `run_info`。
- `RULER/benchmark_root/parquet_data/synthetic/`
  - `RULER/scripts/data/prepare_parquet.py` 生成的 RULER jsonl 输入数据。
  - 目录结构为 `<长度>/data/<任务>/validation.jsonl`。
  - `RULER/benchmark_root/parquet_data/synthetic/conversion_report.json` 记录 parquet 输入、jsonl 输出、转换条数、失败项和忽略项。
- `RULER/benchmark_root/local_eval/`
  - `RULER/scripts/run_parquet_parallel.py` 生成的预测、日志和后续评分结果。
  - 标准结构为 `<模型别名>/synthetic/<长度>/pred/<任务>.jsonl` 和 `<模型别名>/synthetic/<长度>/logs/<任务>.log`。

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

评分时，`RULER/scripts/eval/evaluate.py` 会比较 `pred` 和 `outputs`，并统计空预测数量。

## 推荐测评流程

### 1. 确认环境

RULER 相关命令默认使用 `dl-a800` conda 环境。该环境需要包含 `torch`、`transformers`、`pyarrow`、`nemo`、`nltk`、`pandas`、`tqdm`、`yaml` 等依赖。

常用前置检查：

```bash
conda run -n dl-a800 python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
```

### 2. 将 parquet 转换成 RULER jsonl

如果 `RULER/benchmark_root/parquet_data/synthetic/<长度>/data/<任务>/validation.jsonl` 已经存在，通常不需要重复转换。需要重建或补齐时运行：

```bash
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py
```

只转换一个任务和一个长度，适合检查流程：

```bash
conda run -n dl-a800 python RULER/scripts/data/prepare_parquet.py \
  --tasks niah_single_1 \
  --lengths 4k \
  --max_samples 5 \
  --skip_existing
```

`RULER/scripts/data/prepare_parquet.py` 参数说明：

- `--parquet_data_dir`：原始 parquet benchmark 根目录，默认是 `/home/test05/czyprojects/benchmark/RULER-llama3-1M`。
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

dry-run 只打印任务矩阵和将要执行的 `RULER/scripts/pred/call_api.py` 命令，不启动模型推理。

`RULER/scripts/run_parquet_parallel.py` 参数说明：

- `--model NAME=PATH`：模型别名和模型路径；可以重复传入多个模型。`NAME` 会成为输出目录名，`PATH` 是模型权重目录，例如 `../../models/Qwen3-8B`。
- `--seq-lengths`：要评测的长度。可以传 `4096`、`4096,8192`，也可以传 `all` 自动发现 `--data-root` 下的全部数字长度目录。
- `--tasks`：要评测的任务。可以传 `niah_single_1,qa_2`，也可以传 `all` 使用 13 个默认 synthetic 任务。
- `--gpus`：物理 GPU 编号列表，例如 `0,1,2,3`。runner 会给每个子进程设置对应的 `CUDA_VISIBLE_DEVICES`。
- `--max-workers`：最多同时运行多少个子进程；默认等于 GPU 数量。如果想只用 8 张卡中的 4 张并发，可以设为 `4`。
- `--data-root`：转换后 jsonl 数据根目录，默认 `RULER/benchmark_root/parquet_data/synthetic`。
- `--output-root`：预测、日志和评分输出根目录，默认 `RULER/benchmark_root/local_eval`。
- `--benchmark`：RULER benchmark 名称，当前使用 `synthetic`。
- `--subset`：读取的数据 split，默认 `validation`。
- `--server-type`：传给 `RULER/scripts/pred/call_api.py` 的模型后端类型。跑本地 Hugging Face 权重时使用 `hf`。
- `--python`：启动 `RULER/scripts/pred/call_api.py` 的 Python 解释器，默认是当前解释器。
- `--model-python NAME=PYTHON`：按模型别名指定子进程 Python，可重复传入。未指定的模型继续使用全局 `--python`。例如 `--model-python glm-4-9b=/home/test05/miniconda3/envs/ruler-glm44/bin/python` 表示只有模型别名 `glm-4-9b` 的子进程使用 GLM 环境。
- `--temperature`：采样温度。默认 `0.0`，表示确定性生成。
- `--top-k`：top-k 采样参数，默认 `32`。
- `--top-p`：top-p 采样参数，默认 `1.0`。
- `--random-seed`：随机种子，默认 `0`。
- `--stop-words`：逗号分隔的停止词，会透传给模型 wrapper。
- `--batch-size`：`RULER/scripts/pred/call_api.py` 每次送入模型的样本数。长上下文模型很容易 OOM，保守使用 `1`；短长度和显存充足时可增大。
- `--poll-interval`：runner 检查子进程状态和打印进度的间隔秒数，默认 `10`。
- `--skip-existing`：如果目标预测文件已存在，则跳过该任务。
- `--overwrite-existing`：启动任务前删除对应任务已有的预测 jsonl、attention 摘要文件和日志文件，强制重新生成。它不能和 `--skip-existing` 同时使用。
- `--log-batch-progress`：让子进程输出 `[BATCH_START]`、`[BATCH_DONE]`、`[BATCH_FAILED]`，runner 会把这些行同步回显并写入日志。
- `--log-generation-ppl`：让 Hugging Face 子进程在生成阶段写入 `generation_logprob_sum`、`generation_token_count`、`generation_nll` 和 `generation_ppl`。该 PPL 是模型对自己生成答案 token 的困惑度，不是参考答案困惑度。
- `--timing-file`：结构化任务耗时 jsonl；默认写到 `RULER/benchmark_root/local_eval/ruler_timing.jsonl`。
- `--auto-evaluate`：全部子任务结束后调用 `RULER/scripts/eval/collect_results.py` 生成统一 xlsx 汇总。
- `--report-file`：统一汇总 xlsx 输出路径；默认写到 `RULER/benchmark_root/local_eval/ruler_results.xlsx`。
- `--dry-run`：只打印命令，不启动推理。

### 4. 正式运行预测

跑单模型、单长度、单任务：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
  --seq-lengths 4096 \
  --tasks niah_single_1 \
  --gpus 0 \
  --server-type hf \
  --batch-size 1 \
  --log-batch-progress
```

跑多个模型的 4k 全任务：

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

当前 conda 环境约定是：非 GLM 三个模型使用 `dl-a800`，GLM 使用 `ruler-glm44`。runner 可以由 `dl-a800` 启动，但 GLM 的 `pred/call_api.py` 子进程会使用 `ruler-glm44` 的 Python。

如果需要重新跑已经生成过的任务，不要传 `--skip-existing`，改传 `--overwrite-existing`。该参数会在任务启动前删除对应 `<任务>.jsonl`、`<任务>.attention.jsonl`、`<任务>.attention.md` 和 `<任务>.log`，然后让 `pred/call_api.py` 从头生成。

#### 覆盖重跑和自动测评汇总

覆盖已有预测并在跑完后自动生成统一测评表时，使用 `--overwrite-existing`、`--log-generation-ppl` 和 `--auto-evaluate`：

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

关键约束：

- `--overwrite-existing` 和 `--skip-existing` 互斥；覆盖重跑时不要同时传这两个参数。
- `--overwrite-existing` 只删除当前任务对应的预测 jsonl、attention 摘要和日志，不删除 `summary.csv`、`submission.csv` 或 `ruler_results.xlsx`。
- `--log-generation-ppl` 只支持 `--server-type hf`，记录的是模型对自己生成答案 token 的困惑度。
- `--auto-evaluate` 会在 runner 结束后调用 `RULER/scripts/eval/collect_results.py`，默认把统一结果写到 `RULER/benchmark_root/local_eval/ruler_results.xlsx`。
- 如果有任务失败，runner 仍会写 `ruler_timing.jsonl`，统一测评表会用 `missing` 或 `failed` 标出缺失和未完成任务。

输出位置示例：

- 预测：`RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred/niah_single_1.jsonl`
- 日志：`RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/logs/niah_single_1.log`

### 5. 直接调用 call_api.py 的场景

一般优先用 `RULER/scripts/run_parquet_parallel.py`。只有需要单独调试某个任务、修改 `--max_retries`，或绕开 runner 时才直接调用 `RULER/scripts/pred/call_api.py`：

```bash
cd /home/test05/czyprojects/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dl-a800 python -u pred/call_api.py \
  --data_dir ../benchmark_root/parquet_data/synthetic/4096/data \
  --save_dir ../benchmark_root/local_eval/glm-4-9b/synthetic/4096/pred \
  --benchmark synthetic \
  --task niah_single_1 \
  --subset validation \
  --server_type hf \
  --model_name_or_path ../../models/glm-4-9b \
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
- `--chunk_idx`、`--chunk_amount`：分片预测参数。`chunk_amount > 1` 时会输出 `<任务>-<chunk_idx>.jsonl`，后续 `RULER/scripts/eval/evaluate.py` 会合并 chunk。
- `--server_type`：后端类型，可选 `trtllm`、`vllm`、`sglang`、`openai`、`gemini`、`hf`、`mamba`。
- `--model_name_or_path`：模型名称、API 模型名或本地模型路径。本地 HF 模型使用 `../../models/<模型目录>`。
- `--server_host`、`--server_port`、`--ssh_server`、`--ssh_key_path`：服务式后端连接参数；本地 `hf` 流程通常不用。
- `--temperature`、`--top_k`、`--top_p`、`--random_seed`：生成采样参数。
- `--stop_words`：逗号分隔停止词。
- `--sliding_window_size`：部分后端的滑动窗口参数。
- `--threads`：并发线程数。`hf` 和 `gemini` 会被脚本强制设为 `1`。
- `--batch_size`：每次推理的样本数。
- `--log_batch_progress`：输出 batch 开始、完成和失败日志。
- `--max_retries`：单个 batch 失败后的最大重试次数，默认 `3`。超过后进程非零退出，runner 会释放该 GPU 并记录失败。
- `--log_generation_ppl`：仅支持 `--server_type hf`。开启后在每条预测 jsonl 记录中追加生成答案 token 的 PPL 统计字段。

### 6. 运行评分和统一测评汇总

对一个模型、一个长度的全部预测评分：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

只要 `pred/` 目录里有对应任务的 `<任务>.jsonl`，`RULER/scripts/eval/evaluate.py` 就会评这些任务。缺失的任务会打印 `Prediction file <任务>.jsonl is not found.` 并跳过。

`RULER/scripts/eval/evaluate.py` 参数说明：

- `--data_dir`：预测 jsonl 所在目录，例如 `../benchmark_root/local_eval/Qwen3-8B/synthetic/4096/pred`。
- `--benchmark`：benchmark 名称，当前为 `synthetic`。
- `--verbose`：打印多少条输入、参考答案和预测样例；默认 `0`，不打印样例。

评分结果位置：

- 多任务评分：`RULER/benchmark_root/local_eval/<模型别名>/synthetic/<长度>/pred/summary.csv`
- 单任务评分：`RULER/benchmark_root/local_eval/<模型别名>/synthetic/<长度>/pred/summary-<任务>.csv`
- 提交格式：`RULER/benchmark_root/local_eval/<模型别名>/synthetic/<长度>/pred/submission.csv`

生成跨模型、跨长度、跨任务的统一 xlsx 汇总：

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

`ruler_results.xlsx` 包含：

- `detail`：每行唯一对应 `model + length + task`，记录 score、nulls、samples、pred_lines、生成阶段 PPL、任务耗时、GPU、预测文件和日志文件。
- `summary_by_model`：每个模型一行，包含完成数量、平均分、平均 PPL、`total_task_elapsed_seconds`、`wall_time_seconds` 和样本总数。
- `summary_by_model_and_length`：每个 `model + length` 一行，按长度观察分数、PPL 和耗时变化。
- `summary_by_task`：每个任务一行，记录平均分、最高分对应模型和长度、平均 PPL 和平均耗时。
- `run_info`：记录汇总时间、输入目录、模型/长度/任务列表和 score/PPL/time 口径。

### 7. 检查已有数据和结果

查看每个长度是否都有 13 个输入任务：

```bash
for d in RULER/benchmark_root/parquet_data/synthetic/[0-9]*; do
  printf '%s ' "$(basename "$d")"
  find "$d/data" -name 'validation.jsonl' | wc -l
done
```

查看某个模型 4k 预测是否齐全：

```bash
find RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred \
  -name '*.jsonl' | sort
```

查看预测行数是否等于输入行数：

```bash
wc -l RULER/benchmark_root/parquet_data/synthetic/4096/data/*/validation.jsonl
wc -l RULER/benchmark_root/local_eval/Meta-Llama-3.1-8B/synthetic/4096/pred/*.jsonl
```

### 8. Llama 单样本完整 attention 导出

`tools/dump_llama_attention.py` 是独立诊断脚本，不接入 `RULER/scripts/pred/call_api.py`，也不修改 benchmark 预测输出。它适合只观察一个 Llama 模型在一个 RULER 样本上的生成过程：先生成回答 token，再用 KV cache replay 逐个导出每个生成 token 的完整 attention。

典型运行命令：

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

导出目录结构：

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

每个 `token_XXXX.npy` 表示第 `XXXX` 个生成 token 作为 query 时，对 prompt 和此前已生成 token 的 attention，shape 为 `[num_layers, num_heads, key_length]`。每个 `[layer, head, :]` 的 attention 权重和应接近 1。

查看第 0 个生成 token、第 0 层、第 0 个 head 的完整分布：

```bash
cd /home/test05/czyprojects
conda run -n dl-a800 python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0
```

如果希望表格仍保持相同格式，但按 attention 权重从大到小显示，使用：

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

这个工具默认用于本地 Llama 诊断场景，不保证 GLM、Qwen、Yi 的自定义模型代码可直接复用。完整 attention 文件会比较大，4k prompt 下每个生成 token 的 float32 attention 可能达到十几 MB；长上下文或较多生成 token 时需要提前确认磁盘空间。

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

本项目没有构建步骤。修改 RULER parquet 转换、并行 runner、`RULER/scripts/pred/call_api.py` 或模型 wrapper 后，至少运行：

```bash
conda run -n dl-a800 python -B -m unittest discover -s tests
conda run -n dl-a800 python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py
```

再运行一个 runner dry-run：

```bash
cd /home/test05/czyprojects/RULER/scripts
conda run -n dl-a800 python -B run_parquet_parallel.py \
  --model Meta-Llama-3.1-8B=../../models/Meta-Llama-3.1-8B \
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
