import argparse
from dataclasses import dataclass, replace

import yaml

from src.parser.ac_agent import AgentConfig, get_agent_config, parser_agent_params
from src.parser.env import EnvConfig, get_env_config, parser_env_params
from src.parser.lemma import LemmaConfig, get_lemma_configs, parser_lemma_params
from src.parser.main_config import MainConfig, get_main_config, parser_main_params
from src.parser.optimizer import OptimizerConfig, get_optimizer_config, parser_optimizer_params
from src.parser.recorder import (
    RecorderConfig,
    get_recorder_config,
    parser_recorder_params,
)
from src.parser.reward import RewardConfig, get_reward_config, parser_reward_params
from src.parser.search import SearchConfig, get_search_config, parser_search_params
from src.parser.tokenizer import (
    TokenizerConfig,
    get_tokenizer_config,
    parser_tokenizer_params,
)
from src.parser.transformer import (
    TransformerConfig,
    get_transformer_config,
    parser_transformer_params,
)
from src.parser.ucb import UCBConfig, get_ucb_config, parser_ucb_params


@dataclass(frozen=True)
class RLConfig:
    env: EnvConfig
    lemmas: list[LemmaConfig]
    optimizer: OptimizerConfig
    transformer: TransformerConfig
    tokenizer: TokenizerConfig
    ucb: UCBConfig
    agent: AgentConfig
    recorder: RecorderConfig
    reward: RewardConfig
    search: SearchConfig
    main: MainConfig


def parse_args_rl() -> RLConfig:
    """Parses command-line arguments for the RL training script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyperparameters for RL Training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the RL agent.")

    parser_env_params(train)
    parser_lemma_params(train)
    parser_optimizer_params(train)
    parser_transformer_params(train)
    parser_ucb_params(train)
    parser_tokenizer_params(train)
    parser_agent_params(train)
    parser_recorder_params(train)
    parser_reward_params(train)
    parser_search_params(train)
    parser_main_params(train)

    train.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to a flat (non-nested) YAML config file. "
        "Values from the file serve as defaults that can be overridden by explicit CLI arguments. "
        "If a CLI argument is not provided, the YAML value is used; if neither is provided, the argparse default applies. "
        "Note: explicitly passing a CLI argument that equals its default is indistinguishable from not passing it, "
        "so the YAML value will take precedence in that case.",
    )

    args = parser.parse_args()

    if args.command == "train":

        if args.config_path is not None:
            with open(args.config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                config_dict = {}

            assert all(
                not isinstance(v, dict) for v in config_dict.values()
            ), "Config dictionary should not be nested"

            defaults = vars(parser.parse_args([args.command]))
            args_dict = vars(args)
            for key, value in args_dict.items():
                if value != defaults.get(key):
                    config_dict[key] = value
            args = argparse.Namespace(**config_dict)

        env_config = get_env_config(args)
        lemma_configs = get_lemma_configs(args)
        optimizer_config = get_optimizer_config(args)
        transformer_config = get_transformer_config(args)
        tokenizer_config = get_tokenizer_config(args)
        ucb_config = get_ucb_config(args)
        agent_config = get_agent_config(args)
        recorder_config = get_recorder_config(args)
        reward_config = get_reward_config(args)
        search_config = get_search_config(args)
        main_config = get_main_config(args)

        if tokenizer_config.max_length > 512:
            # some hardcoded values for managing memory usage.
            agent_config = replace(
                agent_config,
                underlying_batch_size=min(agent_config.underlying_batch_size, 400),
            )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    return RLConfig(
        env_config,
        lemma_configs,
        optimizer_config,
        transformer_config,
        tokenizer_config,
        ucb_config,
        agent_config,
        recorder_config,
        reward_config,
        search_config,
        main_config,
    )
