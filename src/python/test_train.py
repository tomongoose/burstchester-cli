import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train
import train_gemma_2b_it_lora
import train_gemma4_e2b_full
import train_gemma4_full


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
            "train_gemma4_full_from_config",
            side_effect=lambda config: captured.update(config),
        ):
            train_gemma4_e2b_full.main()

        self.assertEqual(captured["modelRepo"], "google/gemma-3-4b-it")
        self.assertEqual(captured["trainingMethod"], "full")

    def test_gemma4_full_load_kwargs_use_bfloat16_when_supported(self):
        class _Bf16Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_bf16_supported():
                return True

        class _Torch:
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"
            cuda = _Bf16Cuda()

        self.assertEqual(
            train_gemma4_full.resolve_gemma4_model_load_kwargs(_Torch()),
            {
                "dtype": "bfloat16",
            },
        )

    def test_gemma4_model_loader_falls_back_to_torch_dtype(self):
        class _ModelClass:
            calls = []

            @classmethod
            def from_pretrained(cls, model_repo, **kwargs):
                cls.calls.append((model_repo, kwargs))
                if "dtype" in kwargs:
                    raise TypeError("unexpected keyword argument 'dtype'")
                return {"repo": model_repo, "kwargs": kwargs}

        model = train_gemma4_full.load_gemma4_model(
            _ModelClass,
            "google/gemma-4-E2B",
            {"dtype": "bfloat16"},
        )

        self.assertEqual(model["kwargs"]["torch_dtype"], "bfloat16")
        self.assertEqual(len(_ModelClass.calls), 2)

    def test_gemma4_text_renderer_uses_processor_chat_template(self):
        class _Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return f"rendered:{messages[0]['content']}"

        processor = _Processor()

        self.assertEqual(
            train_gemma4_full.render_gemma4_messages_for_text_only(
                processor,
                [{"role": "user", "content": "hello"}],
            ),
            "rendered:hello",
        )
        self.assertFalse(processor.kwargs["add_generation_prompt"])
        self.assertFalse(processor.kwargs["enable_thinking"])


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
