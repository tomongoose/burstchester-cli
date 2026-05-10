import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train
import train_gemma_2b_it_lora
import train_gemma4_e2b_full


class _CudaAvailable:
    @staticmethod
    def is_available():
        return True


class _CudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _TorchWithCuda:
    float16 = "float16"
    float32 = "float32"
    cuda = _CudaAvailable()


class _TorchWithoutCuda:
    float16 = "float16"
    float32 = "float32"
    cuda = _CudaUnavailable()


class TrainNotebookParityTests(unittest.TestCase):
    def test_gemma_uses_notebook_lora_target_modules(self):
        self.assertEqual(
            train.resolve_lora_target_modules("google/gemma-2b-it"),
            ["q_proj", "v_proj"],
        )

    def test_non_gemma_keeps_broader_lora_target_modules(self):
        self.assertEqual(
            train.resolve_lora_target_modules("Qwen/Qwen3-0.6B"),
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    def test_gemma_lora_loads_with_auto_device_map_and_half_precision_when_cuda_exists(self):
        self.assertEqual(
            train.resolve_model_load_kwargs(
                model_repo="google/gemma-2b-it",
                training_method="lora",
                torch_module=_TorchWithCuda(),
            ),
            {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        )


class Gemma4WrapperTests(unittest.TestCase):
    def test_wrapper_preserves_explicit_model_repo(self):
        captured = {}

        with mock.patch.object(
            train_gemma4_e2b_full,
            "parse_args",
            return_value=SimpleNamespace(config="/tmp/train-config.json"),
        ), mock.patch.object(
            train_gemma4_e2b_full,
            "load_config",
            return_value={"modelRepo": "google/gemma-3-4b-it"},
        ), mock.patch.object(
            train_gemma4_e2b_full,
            "train_from_config",
            side_effect=lambda config: captured.update(config),
        ):
            train_gemma4_e2b_full.main()

        self.assertEqual(captured["modelRepo"], "google/gemma-3-4b-it")
        self.assertEqual(captured["trainingMethod"], "full")


class GemmaLoraWrapperTests(unittest.TestCase):
    def test_wrapper_preserves_explicit_model_repo(self):
        captured = {}

        with mock.patch.object(
            train_gemma_2b_it_lora,
            "parse_args",
            return_value=SimpleNamespace(config="/tmp/train-config.json"),
        ), mock.patch.object(
            train_gemma_2b_it_lora,
            "load_config",
            return_value={"modelRepo": "google/gemma-3-1b-it"},
        ), mock.patch.object(
            train_gemma_2b_it_lora,
            "train_from_config",
            side_effect=lambda config: captured.update(config),
        ):
            train_gemma_2b_it_lora.main()

        self.assertEqual(captured["modelRepo"], "google/gemma-3-1b-it")
        self.assertEqual(captured["trainingMethod"], "lora")
        self.assertEqual(captured["maxSeqLength"], 128)

    def test_gemma_lora_falls_back_to_float32_without_cuda(self):
        self.assertEqual(
            train.resolve_model_load_kwargs(
                model_repo="google/gemma-2b-it",
                training_method="lora",
                torch_module=_TorchWithoutCuda(),
            ),
            {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": "float32",
            },
        )


if __name__ == "__main__":
    unittest.main()
