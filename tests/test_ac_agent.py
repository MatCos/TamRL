import os

import pytest
import torch

from src.environment.environment import State
from src.parser import AgentConfig, OptimizerConfig, TransformerConfig
from src.parser.lemma import LemmaConfig
from src.parser.main_config import MainConfig
from src.parser.recorder import RecorderConfig
from src.parser.tokenizer import TokenizerConfig
from src.rl.ac_agent import ACAgent
from src.rl.agent import Agent
from src.rl.replay_buffer import ACTrainingExample, ReplayBuffer
from src.utils.recorder import Recorder

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def _recorder(tmp_path) -> Recorder:
    cfg = RecorderConfig(
        log_freq=1,
        log_avg_window_model_step=1,
        log_avg_window_env_step=1,
        log_avg_window=1,
    )
    return Recorder(cfg, path=str(tmp_path))


def _agent_config(**overrides) -> AgentConfig:
    defaults = dict(
        batch_size=4,
        underlying_batch_size=800,
        warmup_fraction=0.0,
        min_usage_to_update=10,
        replay_buffer_capacity=100,
        test_set_size=0,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        max_length=64,
        cache_size=0,
        model_name="roberta-base",
        replace_var=None,
        key="name",
        id_range=50,
    )


def _transformer_config() -> TransformerConfig:
    return TransformerConfig(
        pfm_dim=32,
        transformer_layers=1,
        pfm_layers=1,
        n_head=2,
        dim_feedforward=64,
        dropout=0.0,
        use_pointer=False,
    )


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(learning_rate=1e-3)


def _make_agent(tmp_path, **agent_overrides) -> ACAgent:
    return ACAgent(
        device=DEVICE,
        logger=_recorder(tmp_path),
        tokenizer_config=_tokenizer_config(),
        agent_config=_agent_config(**agent_overrides),
        optimizer_config=_optimizer_config(),
        model_config=_transformer_config(),
    )


def _state(n_methods: int = 3) -> State:
    # Use "Simplify" proof methods since they don't require a "goal" sub-dict
    return State(
        proof_methods=[{"name": "Simplify"} for _ in range(n_methods)],
        encoded_methods=tuple(f"enc{i}" for i in range(n_methods)),
        encoded_sys="sys",
    )


def _example(
    n_methods: int = 3, action: int = 0, value: float = 1.0
) -> ACTrainingExample:
    return ACTrainingExample(state=_state(n_methods), action=action, value=value)


def _lemma_config() -> LemmaConfig:
    from src.utils.load import LemmaType

    return LemmaConfig(
        theory_path="/dummy/dummy.spthy",
        lemma_name="lem",
        lemma_type=LemmaType.FORALL,
        theory_name="T",
        diff_arg=False,
        suppress_output=True,
        heuristic="f",
        side=None,
    )


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------


class TestAgentABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Agent()

    def test_ac_agent_is_subclass(self):
        assert issubclass(ACAgent, Agent)


# ---------------------------------------------------------------------------
# ACAgent construction
# ---------------------------------------------------------------------------


class TestACAgentInit:
    def test_creates_model_and_optimizer(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.model is not None
        assert agent.optimizer is not None
        assert agent.model_updates == 0

    def test_stores_config_values(self, tmp_path):
        agent = _make_agent(tmp_path, batch_size=16, warmup_fraction=0.5)
        assert agent.batch_size == 16
        assert agent.warmup_fraction == 0.5


# ---------------------------------------------------------------------------
# Inference and action selection
# ---------------------------------------------------------------------------


class TestInference:
    def test_inference_returns_priors_and_values(self, tmp_path):
        agent = _make_agent(tmp_path)
        state = _state(n_methods=3)
        priors, values = agent.inference(state)
        assert priors.shape[0] == 3
        assert values.shape[0] == 1

    def test_priors_are_log_probabilities(self, tmp_path):
        agent = _make_agent(tmp_path)
        priors, _ = agent.inference(_state(n_methods=4))
        # log probs should be <= 0 and sum(exp) ≈ 1
        assert (priors <= 0).all()
        assert torch.exp(priors).sum().isclose(torch.tensor(1.0), atol=1e-5)

    def test_select_action_returns_valid_index(self, tmp_path):
        agent = _make_agent(tmp_path)
        action, confidence = agent.select_action(_state(n_methods=5))
        assert 0 <= action < 5
        assert isinstance(confidence, float)


# ---------------------------------------------------------------------------
# update_model gating logic
# ---------------------------------------------------------------------------


class TestUpdateModelGating:
    def test_skips_when_buffer_too_small(self, tmp_path):
        agent = _make_agent(tmp_path, batch_size=10)
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example() for _ in range(5)])  # < batch_size
        root = {_lemma_config(): _state()}
        assert agent.update_model(buf, env_steps=1, root_state=root) is False
        assert agent.model_updates == 0

    def test_skips_when_warmup_not_met(self, tmp_path):
        agent = _make_agent(tmp_path, batch_size=4, warmup_fraction=0.5)
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example() for _ in range(10)])  # 10% < 50% warmup
        root = {_lemma_config(): _state()}
        assert agent.update_model(buf, env_steps=1, root_state=root) is False

    def test_skips_when_min_usage_reached(self, tmp_path):
        agent = _make_agent(tmp_path, batch_size=4, min_usage_to_update=1)
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example() for _ in range(10)])
        # Sample repeatedly until every item has been used at least once
        for _ in range(50):
            buf.sample(4)
        assert buf.min_usage_count >= 1
        root = {_lemma_config(): _state()}
        assert agent.update_model(buf, env_steps=1, root_state=root) is False

    def test_updates_when_conditions_met(self, tmp_path):
        agent = _make_agent(
            tmp_path, batch_size=4, warmup_fraction=0.0, min_usage_to_update=10
        )
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example(action=i % 3) for i in range(10)])
        root = {_lemma_config(): _state()}
        assert agent.update_model(buf, env_steps=1, root_state=root) is True
        assert agent.model_updates == 1


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_creates_files(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model_updates = 5
        ckp_dir = str(tmp_path / "checkpoints")
        agent.save_checkpoint(ckp_dir, "_step_100", steps=100)

        assert os.path.exists(os.path.join(ckp_dir, "ckp_last.pt"))
        assert os.path.exists(os.path.join(ckp_dir, "model_last.pt"))
        assert os.path.exists(os.path.join(ckp_dir, "ckp_step_100.pt"))
        assert os.path.exists(os.path.join(ckp_dir, "model_step_100.pt"))

    def test_save_stores_metadata(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model_updates = 7
        ckp_dir = str(tmp_path / "checkpoints")
        agent.save_checkpoint(ckp_dir, "_step_50", steps=50)

        ckp = torch.load(os.path.join(ckp_dir, "ckp_step_50.pt"), map_location="cpu")
        assert ckp["total_updates"] == 7
        assert ckp["steps"] == 50
        assert "optimizer_state_dict" in ckp

    def test_load_restores_agent(self, tmp_path):
        # Save
        agent = _make_agent(tmp_path)
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example(action=i % 3) for i in range(10)])
        root = {_lemma_config(): _state()}
        agent.update_model(buf, env_steps=1, root_state=root)
        original_updates = agent.model_updates

        ckp_dir = str(tmp_path / "run1")
        agent.save_checkpoint(ckp_dir, "_last", steps=42)

        # Load
        main_cfg = MainConfig(
            run_name="run1",
            resume_run="run1",
            load_from_run=None,
            run_path=str(tmp_path / "run1"),
            checkpoint="last",
            save_path_prefix=str(tmp_path),
            use_wandb=False,
            force_cpu=True,
            seed=42,
            wandb_suffix="",
            slurm_id=None,
        )
        loaded_agent, steps = ACAgent.load_or_create_agent(
            main_config=main_cfg,
            agent_config=_agent_config(),
            optimizer_config=_optimizer_config(),
            tokenizer_config=_tokenizer_config(),
            model_config=_transformer_config(),
            device=DEVICE,
            logger=_recorder(tmp_path / "log2"),
        )
        assert steps == 42
        assert loaded_agent.model_updates == original_updates

    def test_create_new_agent_when_no_resume(self, tmp_path):
        main_cfg = MainConfig(
            run_name="newrun",
            resume_run=None,
            load_from_run=None,
            run_path=str(tmp_path / "newrun"),
            checkpoint="last",
            save_path_prefix=str(tmp_path),
            use_wandb=False,
            force_cpu=True,
            seed=42,
            wandb_suffix="",
            slurm_id=None,
        )
        agent, steps = ACAgent.load_or_create_agent(
            main_config=main_cfg,
            agent_config=_agent_config(),
            optimizer_config=_optimizer_config(),
            tokenizer_config=_tokenizer_config(),
            model_config=_transformer_config(),
            device=DEVICE,
            logger=_recorder(tmp_path / "log"),
        )
        assert steps == 0
        assert agent.model_updates == 0
