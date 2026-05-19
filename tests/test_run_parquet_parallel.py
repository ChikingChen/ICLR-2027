import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "run_parquet_parallel.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_parquet_parallel", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunParquetParallelTest(unittest.TestCase):
    """验证 parquet 数据并行 runner 的参数展开和命令构造。"""

    def test_parse_model_specs_requires_name_and_path(self):
        module = _load_module()

        models = module.parse_model_specs(
            ["llama=../../models/Meta-Llama-3.1-8B", "qwen=/models/Qwen3-8B"]
        )

        self.assertEqual([model.name for model in models], ["llama", "qwen"])
        self.assertEqual(models[0].path, Path("../../models/Meta-Llama-3.1-8B"))
        with self.assertRaises(ValueError):
            module.parse_model_specs(["missing_equals"])

    def test_resolve_seq_lengths_supports_all_and_csv(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            (data_root / "4096").mkdir()
            (data_root / "1048576").mkdir()
            (data_root / "notes").mkdir()

            self.assertEqual(module.resolve_seq_lengths("all", data_root), [4096, 1048576])
            self.assertEqual(module.resolve_seq_lengths("4096, 8192,1048576", data_root), [4096, 8192, 1048576])

    def test_resolve_tasks_supports_all_and_rejects_unknown(self):
        module = _load_module()

        tasks = module.resolve_tasks("all")

        self.assertEqual(len(tasks), 13)
        self.assertEqual(tasks[0], "niah_single_1")
        self.assertEqual(module.resolve_tasks("niah_single_1,qa_2"), ["niah_single_1", "qa_2"])
        with self.assertRaises(ValueError):
            module.resolve_tasks("niah_single_1,unknown_task")

    def test_build_jobs_interleaves_models_per_task(self):
        module = _load_module()
        models = module.parse_model_specs(["llama=/models/llama", "qwen=/models/qwen"])

        jobs = module.build_jobs(models, [4096], ["niah_single_1", "qa_2"])

        self.assertEqual(
            [(job.seq_length, job.task, job.model.name) for job in jobs],
            [
                (4096, "niah_single_1", "llama"),
                (4096, "niah_single_1", "qwen"),
                (4096, "qa_2", "llama"),
                (4096, "qa_2", "qwen"),
            ],
        )

    def test_batch_progress_lines_are_selected_for_terminal_echo(self):
        module = _load_module()

        self.assertTrue(module.is_batch_progress_line("[BATCH_START] task=niah_single_1"))
        self.assertTrue(module.is_batch_progress_line("[BATCH_DONE] task=niah_single_1"))
        self.assertTrue(module.is_batch_progress_line("[BATCH_FAILED] task=niah_single_1"))
        self.assertTrue(
            module.is_batch_progress_line(
                "  0%|          | 0/500 [00:00<?, ?it/s][BATCH_START] task=niah_single_1"
            )
        )
        self.assertFalse(module.is_batch_progress_line("[RUNNING] task=niah_single_1"))

    def test_build_call_api_command_uses_parquet_layout_and_progress_flag(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
                "--data-root",
                "/data/parquet/synthetic",
                "--output-root",
                "/data/local_eval",
                "--python",
                "/usr/bin/python",
                "--batch-size",
                "1000",
                "--log-batch-progress",
                "--log-attention-scores",
                "--attention-top-k",
                "6",
                "--log-generation-ppl",
            ]
        )
        config = module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))
        job = module.build_jobs(module.parse_model_specs(args.model), [4096], ["niah_single_1"])[0]

        command = module.build_call_api_command(job, config)

        self.assertEqual(command[0], "/usr/bin/python")
        self.assertIn("pred/call_api.py", command)
        self.assertIn("--log_batch_progress", command)
        self.assertIn("--log_attention_scores", command)
        self.assertIn("--log_generation_ppl", command)
        self.assertIn("--attention_top_k", command)
        self.assertIn("6", command)
        self.assertEqual(module.prediction_file_for(job, config), Path("/data/local_eval/llama/synthetic/4096/pred/niah_single_1.jsonl"))
        self.assertEqual(module.log_file_for(job, config), Path("/data/local_eval/llama/synthetic/4096/logs/niah_single_1.log"))
        self.assertIn("/data/parquet/synthetic/4096/data", command)
        self.assertIn("/data/local_eval/llama/synthetic/4096/pred", command)

    def test_model_python_overrides_command_python_for_matching_model(self):
        module = _load_module()
        glm_python = "/home/test05/miniconda3/envs/ruler-glm44/bin/python"
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--model",
                "glm-4-9b=/models/glm-4-9b",
                "--model-python",
                f"glm-4-9b={glm_python}",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
                "--python",
                "/envs/dl-a800/bin/python",
            ]
        )
        models = module.parse_model_specs(args.model)
        config = module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))
        jobs = module.build_jobs(models, [4096], ["niah_single_1"])

        commands = {
            job.model.name: module.build_call_api_command(job, config)
            for job in jobs
        }

        self.assertEqual(commands["glm-4-9b"][0], glm_python)
        self.assertEqual(commands["llama"][0], "/envs/dl-a800/bin/python")

    def test_model_python_rejects_unknown_model_alias(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--model-python",
                "glm-4-9b=/home/test05/miniconda3/envs/ruler-glm44/bin/python",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
            ]
        )

        with self.assertRaises(ValueError):
            module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))

    def test_overwrite_existing_rejects_skip_existing(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
                "--skip-existing",
                "--overwrite-existing",
            ]
        )

        with self.assertRaises(ValueError):
            module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))

    def test_overwrite_existing_removes_prediction_log_and_attention_outputs(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
                "--overwrite-existing",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args.data_root = root / "parquet"
            args.output_root = root / "local_eval"
            config = module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))
            job = module.build_jobs(module.parse_model_specs(args.model), [4096], ["niah_single_1"])[0]
            pred_file = module.prediction_file_for(job, config)
            log_file = module.log_file_for(job, config)
            attention_jsonl = pred_file.with_suffix(".attention.jsonl")
            attention_markdown = pred_file.with_suffix(".attention.md")
            unrelated_file = pred_file.parent / "qa_2.jsonl"
            for path in [pred_file, log_file, attention_jsonl, attention_markdown, unrelated_file]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old\n", encoding="utf-8")

            removed = module.delete_existing_outputs(job, config)

            self.assertEqual(
                sorted(path.name for path in removed),
                sorted([
                    "niah_single_1.jsonl",
                    "niah_single_1.log",
                    "niah_single_1.attention.jsonl",
                    "niah_single_1.attention.md",
                ]),
            )
            for path in [pred_file, log_file, attention_jsonl, attention_markdown]:
                self.assertFalse(path.exists())
            self.assertTrue(unrelated_file.exists())

    def test_timing_record_is_written_as_jsonl(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args.data_root = root / "parquet"
            args.output_root = root / "local_eval"
            args.timing_file = root / "timing.jsonl"
            config = module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))
            job = module.build_jobs(module.parse_model_specs(args.model), [4096], ["niah_single_1"])[0]

            module.write_timing_record(
                config=config,
                job=job,
                gpu=3,
                started_at=10.0,
                ended_at=12.5,
                elapsed_seconds=2.5,
                pred_lines=7,
                total_samples=9,
                exit_code=0,
            )

            records = [json.loads(line) for line in args.timing_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["model"], "llama")
            self.assertEqual(records[0]["length"], 4096)
            self.assertEqual(records[0]["task"], "niah_single_1")
            self.assertEqual(records[0]["gpu"], 3)
            self.assertEqual(records[0]["elapsed_seconds"], 2.5)
            self.assertEqual(records[0]["pred_lines"], 7)
            self.assertEqual(records[0]["total_samples"], 9)
            self.assertEqual(records[0]["exit_code"], 0)

    def test_collect_results_command_uses_runner_outputs(self):
        module = _load_module()
        args = module.build_parser().parse_args(
            [
                "--model",
                "llama=/models/llama",
                "--seq-lengths",
                "4096",
                "--tasks",
                "niah_single_1",
                "--gpus",
                "0",
                "--auto-evaluate",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args.data_root = root / "parquet"
            args.output_root = root / "local_eval"
            config = module.build_config(args, scripts_dir=Path("/repo/RULER/scripts"))

            command = module.build_collect_results_command(
                config=config,
                models=["llama"],
                seq_lengths=[4096],
                tasks=["niah_single_1"],
            )

            self.assertEqual(command[0], args.python)
            self.assertIn("eval/collect_results.py", command)
            self.assertIn("--output-root", command)
            self.assertIn(str(args.output_root), command)
            self.assertIn("--data-root", command)
            self.assertIn(str(args.data_root), command)
            self.assertIn("--timing-file", command)
            self.assertIn(str(config.timing_file), command)
            self.assertIn("--output-file", command)
            self.assertIn(str(config.report_file), command)


if __name__ == "__main__":
    unittest.main()
