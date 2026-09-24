import json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from common.io import read_config, load_rows, sha


class Configuration(unittest.TestCase):
    def test_training_parameters(self):
        models = {
            "llama32_3b": (0.001, 1, 3),
            "qwen3_4b": (0.01, 1, 4),
            "llama31_8b": (0.003, 2, 3),
            "qwen3_8b": (0.001, 1, 3),
            "olmo2_13b": (0.001, 1, 3),
            "qwen3_14b": (0.001, 1, 3),
        }
        for model, (beta, weight, q) in models.items():
            microbatch = 1 if model in {"olmo2_13b", "qwen3_14b"} else 2
            c = read_config(ROOT / f"corver/configs/{model}.yaml")
            t = c["training"]
            self.assertEqual(
                (
                    t["learning_rate"],
                    t["beta"],
                    c["judge_weight"],
                    c["questions_per_step"],
                ),
                (1e-5, beta, weight, q),
            )
            self.assertEqual(
                (
                    t["max_steps"],
                    t["seed"],
                    t["num_generations"],
                    t["max_completion_length"],
                    t["temperature"],
                ),
                (100, 42, 16, 1000, 1),
            )
            self.assertEqual(
                (
                    c["lora"]["r"],
                    c["lora"]["alpha"],
                    t["per_device_train_batch_size"],
                    t["gradient_accumulation_steps"],
                ),
                (128, 256, microbatch, q * 16 // microbatch),
            )

    def test_scaling_model_sources_and_loaders(self):
        c = read_config(ROOT / "corver/configs/olmo2_13b.yaml")
        self.assertEqual(c["model"], {"id": "allenai/OLMo-2-1124-13B-Instruct", "revision": "3a5c85baefbb1896a54d56fe2e76c0395627ddf4", "loader": "fast_model"})
        self.assertEqual(c["training"]["generation_batch_size"], 48)
        self.assertEqual(c["runtime"]["policy_gpu_memory_utilization"], 0.55)
        c = read_config(ROOT / "corver/configs/qwen3_14b.yaml")
        self.assertEqual(c["model"], {"id": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "loader": "fast_language_model"})
        self.assertEqual(c["training"]["generation_batch_size"], 48)
        self.assertEqual(c["runtime"]["policy_gpu_memory_utilization"], 0.55)

    def test_pool_manifest(self):
        m = json.loads((ROOT / "data/manifest.json").read_text())
        self.assertEqual(len(load_rows(ROOT / "data/train.json", 6621)), m["rows"])
        self.assertEqual(sha(ROOT / "data/train.json"), m["sha256"])


if __name__ == "__main__":
    unittest.main()
