import argparse
import os
import sys

import pytest
import yaml

from src.parser.ac_agent import AgentConfig, get_agent_config
from src.parser.env import EnvConfig, get_env_config
from src.parser.main_config import create_run_name
from src.parser.optimizer import OptimizerConfig, get_optimizer_config
from src.parser.recorder import RecorderConfig, get_recorder_config
from src.parser.reward import RewardConfig, get_reward_config
from src.parser.search import SearchConfig, get_search_config
from src.parser.tokenizer import TokenizerConfig, get_tokenizer_config
from src.parser.transformer import TransformerConfig, get_transformer_config
from src.parser.ucb import UCBConfig, get_ucb_config


def ns(**kwargs):
    return argparse.Namespace(**kwargs)


EMPTY = ns()


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestFrozen:
    @pytest.mark.parametrize("cfg", [
        EnvConfig(4, 1, 120, 600, 50000),
        OptimizerConfig(1e-4),
        RecorderConfig(50, 10, 10, 100),
        RewardConfig(0.0, 0.0, 180.0, 3),
        SearchConfig(1000, 1.5, 5, 20, 4, "search", 0),
        TokenizerConfig(512, 100, "roberta-base", "none", "name", 50),
        TransformerConfig(512, 4, 2, 8, 1024, 0.0, True),
        UCBConfig(3200, 0.001, 0.99, 200, 64, 8, False, 0.0),
        AgentConfig(32, 800, 0.0, 10, 10000, 64),
    ])
    def test_all_configs_frozen(self, cfg):
        with pytest.raises(AttributeError):
            cfg.fake_field = 42


# ---------------------------------------------------------------------------
# Factory defaults and field mappings
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_all_defaults(self):
        assert get_env_config(EMPTY) == EnvConfig(4, 1, 120, 600, 100000.0)
        assert get_optimizer_config(EMPTY) == OptimizerConfig(1e-4)
        assert get_recorder_config(EMPTY) == RecorderConfig(50, 10, 10, 100)
        assert get_reward_config(EMPTY) == RewardConfig(0.0, 0.0, 90.0, 3)
        assert get_search_config(EMPTY) == SearchConfig(1000, 1.5, 5, 20, 4, "search", 0)
        assert get_tokenizer_config(EMPTY) == TokenizerConfig(512, 100000, "roberta-base", None, "name", 50)
        assert get_transformer_config(EMPTY) == TransformerConfig(512, 4, 2, 8, 1024, 0.0, True)
        assert get_ucb_config(EMPTY) == UCBConfig(100, 1.0, 0.99, 10, 8, 8, False, 0.0, False)
        assert get_agent_config(EMPTY) == AgentConfig(32, 800, 0.0, 10, 10000, 64)


class TestFieldMappings:
    def test_transformer_mappings(self):
        cfg = get_transformer_config(ns(n_transformer_head=16, dim_transformer_feedforward=2048))
        assert cfg.n_head == 16
        assert cfg.dim_feedforward == 2048

    def test_tokenizer_mappings(self):
        cfg = get_tokenizer_config(ns(tokenizer="gpt2", tokenizer_max_length=256,
                                      tokenizer_cache_size=500, id_range_var=100))
        assert cfg.model_name == "gpt2"
        assert cfg.max_length == 256
        assert cfg.cache_size == 500
        assert cfg.id_range == 100

    def test_search_mappings(self):
        cfg = get_search_config(ns(search_budget=500, search_budget_increase_factor=2.0))
        assert cfg.budget == 500
        assert cfg.budget_increase_factor == 2.0

    def test_agent_batch_size_mapping(self):
        assert get_agent_config(ns(max_final_batch_size=1600)).underlying_batch_size == 1600


# ---------------------------------------------------------------------------
# Boolean flag inversions and sweep overrides
# ---------------------------------------------------------------------------


class TestBooleanFlags:
    def test_no_pointer_inversion(self):
        assert get_transformer_config(ns(no_pointer=True)).use_pointer is False
        assert get_transformer_config(ns(no_pointer=False)).use_pointer is True

    def test_ucb_sweep_overrides(self):
        # invert_and_sweep overrides invert_and
        assert get_ucb_config(ns(invert_and=False, invert_and_sweep=True)).invert_and is True
        assert get_ucb_config(ns(invert_and=True, invert_and_sweep=None)).invert_and is True
        # deactivate_model_sweep overrides deactivate_model
        assert get_ucb_config(ns(deactivate_model=False, deactivate_model_sweep=True)).deactivate_model is True
        assert get_ucb_config(ns(deactivate_model=True, deactivate_model_sweep=None)).deactivate_model is True


# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------


class TestCreateRunName:
    def test_new_run_random_name(self, tmp_path):
        path, name = create_run_name(ns(save_path_prefix=str(tmp_path), seed=42))
        assert len(name) == 8 and name.isalnum()
        assert path == os.path.join(str(tmp_path), name) + "/"
        assert os.path.isdir(path)

    def test_resume_uses_given_name(self, tmp_path):
        path, name = create_run_name(ns(resume_run="myrun123", save_path_prefix=str(tmp_path), seed=42))
        assert name == "myrun123"
        assert "myrun123" in path


# ---------------------------------------------------------------------------
# Custom args override defaults
# ---------------------------------------------------------------------------


class TestCustomArgs:
    def test_env_custom(self):
        cfg = get_env_config(ns(num_envs=8, request_timeout=60))
        assert cfg.num_envs == 8
        assert cfg.request_timeout == 60
        assert cfg.stall_frequency == 1  # default preserved

    def test_reward_custom(self):
        cfg = get_reward_config(ns(branch_penalty=0.5, timeout_penalty=10))
        assert cfg.branch_penalty == 0.5
        assert cfg.timeout_penalty == 10
        assert cfg.time_penalty == 0.0  # default preserved


# ---------------------------------------------------------------------------
# YAML/CLI config merge
# ---------------------------------------------------------------------------


class TestConfigMerge:
    def test_yaml_values_not_clobbered_by_defaults(self, tmp_path):
        """YAML values should survive when CLI doesn't explicitly override them."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"batch_size": 64, "num_envs": 16}))

        from src.parser.parser_rl import parse_args_rl

        argv = ["train", "--config_path", str(config_file),
                "--protocol", str(tmp_path)]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["prog"] + argv)
            cfg = parse_args_rl()

        assert cfg.agent.batch_size == 64
        assert cfg.env.num_envs == 16

    def test_cli_overrides_yaml(self, tmp_path):
        """Explicitly passed CLI args should override YAML values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"batch_size": 64}))

        from src.parser.parser_rl import parse_args_rl

        argv = ["train", "--config_path", str(config_file),
                "--batch_size", "128",
                "--protocol", str(tmp_path)]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["prog"] + argv)
            cfg = parse_args_rl()

        assert cfg.agent.batch_size == 128
