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

截至 2026-05-19，当前工作区已经迁移到新服务器，路径为 `/data/czy/ICLR-2027`。旧说明中的 `/home/test05/czyprojects`、`dl-a800` 和 `ruler-glm44` 不再适用。当前 conda 只可见 `base` 和 `model_download`，这两个环境都没有安装完整 RULER 推理依赖，暂不能直接运行本地 Hugging Face 推理测评。

当前状态摘要：

- 当前主机名：`dilab`。
- 当前 conda：`/data/czy/miniconda3/bin/conda`。
- 当前 Python：`/data/czy/miniconda3/bin/python`，版本 `Python 3.13.13`。
- 当前可见 conda 环境：`base`、`model_download`。
- `base` 和 `model_download` 当前都不能导入 `torch`。
- `nvidia-smi` 当前无法和 NVIDIA driver 通信，GPU 状态需要重新确认。
- `RULER/benchmark_root/` 当前不存在，因此已转换 jsonl 输入、预测输出、评分输出和 timing 文件都还没有在新服务器上生成。
- `benchmark/RULER-llama3-1M/` 当前存在 117 个任务长度目录，覆盖 9 个长度和 13 个 synthetic 任务；共有 118 个 `validation-*.parquet`，其中 `qa_1_1M` 有两个 parquet 分片。
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
  - 本地增加了 `--log_batch_progress`、`--max_retries` 和 `--log_generation_ppl`，用于观察 batch 进度、避免单个 batch 无限重试，并在 Hugging Face 生成阶段记录生成 token PPL。
- `RULER/scripts/pred/model_wrappers.py`
  - Hugging Face、本地模型和 Mamba wrapper。
  - 本地包含 GLM 配置兼容逻辑。
  - 本地 Hugging Face wrapper 支持基于 `generate(..., output_scores=True)` 的生成答案 token PPL 统计。
- `RULER/scripts/eval/evaluate.py`
  - RULER 原生评分入口。
  - 读取预测 jsonl，按 `RULER/scripts/eval/synthetic/constants.py` 中的 metric 计算每个任务分数。
- `RULER/scripts/eval/collect_results.py`
  - 本地统一汇总脚本。
  - 跨模型、长度和任务读取预测 jsonl、生成阶段 PPL 字段和 runner timing jsonl。
  - 输出单个 `ruler_results.xlsx`，包含 `detail`、`summary_by_model`、`summary_by_model_and_length`、`summary_by_task` 和 `run_info`。
- `RULER/benchmark_root/parquet_data/synthetic/`
  - `RULER/scripts/data/prepare_parquet.py` 生成的 RULER jsonl 输入数据。
  - 当前新服务器上该目录不存在，需要配置环境后重新转换。
- `RULER/benchmark_root/local_eval/`
  - `RULER/scripts/run_parquet_parallel.py` 生成的预测、日志和后续评分结果。
  - 当前新服务器上该目录不存在，需要重新运行预测后生成。

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

## 新服务器待配置事项

当前没有可直接使用的 RULER 推理环境。开始正式测评前，需要先完成以下事项：

1. 创建或指定新的 Python/conda 环境。
2. 安装 `torch`、`transformers`、`accelerate`、`safetensors`、`sentencepiece`、`pyarrow`、`pandas`、`nltk`、`tqdm`、`yaml` 等基础依赖。
3. 针对 `Qwen2.5-7B-Instruct-1M`、`GLM-4-9B-Chat-1M`、`Llama-3.1-8B` 和 `Yi-9B-200K` 分别确认所需 `transformers` 版本、`trust_remote_code` 支持和长上下文配置。
4. 确认 GPU driver、CUDA 和 PyTorch CUDA 可用；当前 `nvidia-smi` 不能正常工作。
5. 配好环境后，先跑小样本转换和 runner dry-run，再进行正式推理。

旧命令中所有 `conda run -n dl-a800 ...`、`conda run -n ruler-glm44 ...`、`/home/test05/czyprojects`、`/home/test05/miniconda3/...` 路径都需要替换。新文档示例统一使用 `/data/czy/ICLR-2027` 和占位环境名 `<ruler-env>`。

## 推荐测评流程

### 1. 确认环境

当前环境未配置完成。配置完成后，先运行：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python -c "import torch, transformers, pyarrow, pandas, nltk; print('ok')"
conda run -n <ruler-env> python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

如果 GLM 或其他模型需要独立环境，可以继续使用 `RULER/scripts/run_parquet_parallel.py` 的 `--model-python NAME=PYTHON` 机制，但不要再假设旧的 `ruler-glm44` 环境存在。

### 2. 将 parquet 转换成 RULER jsonl

当前 `RULER/benchmark_root/parquet_data/synthetic/` 不存在，配置好环境后需要重新转换。

转换全部数据：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python RULER/scripts/data/prepare_parquet.py
```

只转换一个任务和一个长度，适合检查流程：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python RULER/scripts/data/prepare_parquet.py \
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
- `--batch-size`：`RULER/scripts/pred/call_api.py` 每次送入模型的样本数。长上下文模型很容易 OOM，保守使用 `1`。
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

配置好环境、转换好 jsonl 并确认 GPU 可用后，再运行预测。

跑单模型、单长度、单任务：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run -n <ruler-env> python run_parquet_parallel.py \
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
conda run -n <ruler-env> python run_parquet_parallel.py \
  --model Llama-3.1-8B=../../models/Llama-3.1-8B \
  --model Qwen2.5-7B-Instruct-1M=../../models/Qwen2.5-7B-Instruct-1M \
  --model Yi-9B-200K=../../models/Yi-9B-200K \
  --model GLM-4-9B-Chat-1M=../../models/GLM-4-9B-Chat-1M \
  --seq-lengths 4096 \
  --tasks all \
  --gpus 0,1,2,3 \
  --server-type hf \
  --batch-size 1 \
  --poll-interval 10 \
  --log-batch-progress \
  --skip-existing
```

如果某个模型需要独立 Python，使用 `--model-python`：

```bash
  --python /data/czy/miniconda3/envs/<ruler-env>/bin/python \
  --model-python GLM-4-9B-Chat-1M=/data/czy/miniconda3/envs/<glm-env>/bin/python
```

如果需要重新跑已经生成过的任务，不要传 `--skip-existing`，改传 `--overwrite-existing`。该参数会在任务启动前删除对应 `<任务>.jsonl`、`<任务>.attention.jsonl`、`<任务>.attention.md` 和 `<任务>.log`，然后让 `pred/call_api.py` 从头生成。

### 5. 直接调用 call_api.py 的场景

一般优先用 `RULER/scripts/run_parquet_parallel.py`。只有需要单独调试某个任务、修改 `--max_retries`，或绕开 runner 时才直接调用 `RULER/scripts/pred/call_api.py`：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n <ruler-env> python -u pred/call_api.py \
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
- `--batch_size`：每次推理的样本数。
- `--log_batch_progress`：输出 batch 开始、完成和失败日志。
- `--max_retries`：单个 batch 失败后的最大重试次数，默认 `3`。超过后进程非零退出，runner 会释放该 GPU 并记录失败。
- `--log_generation_ppl`：仅支持 `--server_type hf`。开启后在每条预测 jsonl 记录中追加生成答案 token 的 PPL 统计字段。

### 6. 运行评分和统一测评汇总

对一个模型、一个长度的全部预测评分：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run -n <ruler-env> python eval/evaluate.py \
  --data_dir ../benchmark_root/local_eval/Llama-3.1-8B/synthetic/4096/pred \
  --benchmark synthetic
```

只要 `pred/` 目录里有对应任务的 `<任务>.jsonl`，`RULER/scripts/eval/evaluate.py` 就会评这些任务。缺失的任务会打印 `Prediction file <任务>.jsonl is not found.` 并跳过。

生成跨模型、跨长度、跨任务的统一 xlsx 汇总：

```bash
cd /data/czy/ICLR-2027/RULER/scripts
conda run -n <ruler-env> python eval/collect_results.py \
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

### 8. Llama 单样本完整 attention 导出

`tools/dump_llama_attention.py` 是独立诊断脚本，不接入 `RULER/scripts/pred/call_api.py`，也不修改 benchmark 预测输出。它适合只观察一个 Llama 模型在一个 RULER 样本上的生成过程：先生成回答 token，再用 KV cache replay 逐个导出每个生成 token 的完整 attention。

典型运行命令：

```bash
cd /data/czy/ICLR-2027
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n <ruler-env> python -u tools/dump_llama_attention.py \
  --model-path models/Llama-3.1-8B \
  --data-file RULER/benchmark_root/parquet_data/synthetic/4096/data/niah_single_1/validation.jsonl \
  --sample-offset 0 \
  --output-dir attention_dumps/llama_niah_single_1_sample_0 \
  --dtype float32 \
  --max-new-tokens 128 \
  --overwrite
```

查看第 0 个生成 token、第 0 层、第 0 个 head 的完整分布：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python tools/inspect_attention_dump.py \
  --dump-dir attention_dumps/llama_niah_single_1_sample_0 \
  --generated-token 0 \
  --layer 0 \
  --head 0 \
  --sort-by attention \
  --descending
```

这个工具默认用于本地 Llama 诊断场景，不保证 GLM、Qwen、Yi 的自定义模型代码可直接复用。完整 attention 文件会比较大，长上下文或较多生成 token 时需要提前确认磁盘空间。

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

本项目没有构建步骤。当前新服务器没有配置完整 RULER 运行环境，因此下面命令需要在 `<ruler-env>` 配好后运行。

修改 RULER parquet 转换、并行 runner、`RULER/scripts/pred/call_api.py` 或模型 wrapper 后，至少运行：

```bash
cd /data/czy/ICLR-2027
conda run -n <ruler-env> python -B -m unittest discover -s tests
conda run -n <ruler-env> python -B -m py_compile \
  RULER/scripts/data/prepare_parquet.py \
  RULER/scripts/run_parquet_parallel.py \
  RULER/scripts/pred/call_api.py \
  RULER/scripts/pred/model_wrappers.py \
  RULER/scripts/eval/collect_results.py
```

再运行一个 runner dry-run：

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
