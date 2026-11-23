"""CPU smoke test for the aijiajia generation branch with synthetic text."""
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uer.utils import str2tokenizer


class GenerationSmokeTest(unittest.TestCase):
    def test_generation_pipeline(self):
        cpm_model = "models/cpm/spiece.model"
        tokenizer = str2tokenizer["cpm"](Namespace(spm_model_path=cpm_model, vocab_path=None))
        markers = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
        ids = [1, 4, 5, 6, 0]
        self.assertEqual(tokenizer.convert_tokens_to_ids(markers), ids)
        self.assertEqual(tokenizer.convert_ids_to_tokens(ids), markers)

        def run(script, *args):
            result = subprocess.run(
                [sys.executable, script, *map(str, args)], cwd=ROOT, check=False,
                capture_output=True, text=True, timeout=90,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            return result.stdout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "poems.txt"
            corpus.write_text("明月照松林，清泉流石上。\n" * 4, encoding="utf-8")
            prompt = root / "prompt.txt"
            prompt.write_text("明月\n", encoding="utf-8")

            for name, config, tokenizer_args in (
                ("bert", "models/gpt-config.json", ["--vocab_path", "models/my_vocab.txt"]),
                ("cpm", "models/cpm-config.json",
                 ["--tokenizer", "cpm", "--spm_model_path", cpm_model]),
            ):
                with self.subTest(tokenizer=name):
                    config_data = json.loads((ROOT / config).read_text(encoding="utf-8"))
                    self.assertEqual(config_data["max_seq_length"], 1024)
                    self.assertEqual(config_data["embedding"], ["word", "pos"])
                    config_data.update(emb_size=8, hidden_size=8, feedforward_size=16,
                                       layers_num=1, heads_num=2, max_seq_length=32)
                    tiny_config = root / f"{name}.json"
                    tiny_config.write_text(json.dumps(config_data), encoding="utf-8")
                    dataset = root / f"{name}.pt"
                    checkpoint = root / f"{name}.bin"
                    run("preprocess.py", "--corpus_path", corpus, "--dataset_path", dataset,
                        "--processes_num", 2, "--seq_length", 8, "--data_processor", "lm",
                        *tokenizer_args)
                    self.assertGreater(dataset.stat().st_size, 0)
                    train = ["--dataset_path", dataset, "--config_path", tiny_config,
                             "--total_steps", 1, "--save_checkpoint_steps", 1, "--report_steps",
                             1, "--batch_size", 2, "--instances_buffer_size", 2,
                             "--learning_rate", 0, *tokenizer_args]
                    run("pretrain.py", "--output_model_path", checkpoint, *train)
                    current = torch.load(f"{checkpoint}-1", map_location="cpu", weights_only=True)
                    legacy_keys = {
                        "embedding.word.embedding.weight": "embedding.word_embedding.weight",
                        "embedding.pos.embedding.weight": "embedding.position_embedding.weight",
                        "target.lm.output_layer.weight": "target.output_layer.weight",
                    }
                    legacy = root / f"{name}-legacy.bin"
                    torch.save({legacy_keys.get(key, key): value for key, value in current.items()}, legacy)
                    resumed = root / f"{name}-resumed.bin"
                    run("pretrain.py", "--output_model_path", resumed,
                        "--pretrained_model_path", legacy, *train)
                    resumed_state = torch.load(f"{resumed}-1", map_location="cpu", weights_only=True)
                    for key in legacy_keys:
                        torch.testing.assert_close(resumed_state[key], current[key], rtol=0, atol=0)
                    converted = root / f"{name}-hf.bin"
                    run("scripts/convert_gpt2_from_uer_to_huggingface.py", "--input_model_path", legacy,
                        "--output_model_path", converted, "--layers_num", 1)
                    converted_state = torch.load(converted, map_location="cpu", weights_only=True)
                    torch.testing.assert_close(converted_state["transformer.wte.weight"],
                                               current["embedding.word.embedding.weight"])
                    torch.testing.assert_close(converted_state["transformer.wpe.weight"],
                                               current["embedding.pos.embedding.weight"])
                    prediction = root / f"{name}-prediction.txt"
                    run("scripts/generate_lm.py", "--load_model_path", legacy,
                        "--test_path", prompt, "--prediction_path", prediction,
                        "--config_path", tiny_config, "--seq_length", 8, "--top_k", 1,
                        *tokenizer_args)
                    provided, generated = prediction.read_text(encoding="utf-8").split("\n", 1)
                    self.assertEqual(provided, "明月")
                    self.assertIn(provided, generated)
                    self.assertGreater(len(generated), len(provided))


if __name__ == "__main__":
    unittest.main()
