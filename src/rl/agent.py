from abc import ABC, abstractmethod
from typing import Self

import torch

from src.parser import AgentConfig, OptimizerConfig, TokenizerConfig, TransformerConfig
from src.parser.lemma import LemmaConfig
from src.parser.main_config import MainConfig
from src.environment.environment import State
from src.rl.replay_buffer import ReplayBuffer


class Agent(ABC):

    @abstractmethod
    def select_action(
        self,
        state: State,
    ) -> tuple[int, float]:
        """Selects an action greedily using the policy network only"""

    @abstractmethod
    @torch.inference_mode()
    def inference(
        self,
        state: State,
    ) -> tuple[torch.Tensor, ...]:
        """Performs inference on the given state and returns the model outputs."""

    @abstractmethod
    def update_model(
        self,
        replay_buffer: ReplayBuffer,
        env_steps: int,
        root_state: dict[LemmaConfig, State],
    ) -> bool:
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, checkpoint_name: str, steps: int) -> None:
        pass

    # TODO saving all necessary model parameters for loading
    @classmethod
    @abstractmethod
    def load_or_create_agent(
        cls,
        main_config: MainConfig,
        agent_config: AgentConfig,
        optimizer_config: OptimizerConfig,
        tokenizer_config: TokenizerConfig,
        model_config: TransformerConfig,
        device: torch.device,
        logger,
    ) -> tuple[Self, int]:
        pass
