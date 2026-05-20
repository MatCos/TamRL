import sys
from collections import deque
from time import time

import numpy as np
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from tabulate import tabulate

import wandb
from src.parser import RecorderConfig


class Recorder:
    def __init__(
        self, recorder_config: RecorderConfig, path, wandb_run=None, num_envs=1
    ):
        self._setup_logger(path)
        self.use_wandb = wandb_run is not None
        if self.use_wandb:
            self.setup_wandb(wandb_run)
        self.logged_step_already = False

        self.config = recorder_config
        self.num_envs = num_envs

        self.system_state: list[list[str | int | None]] = []
        self.system_headers = [
            "Type",
            "PID",
            "mem\n(MB)",
            "t_pid",
            "t_mem\n(MB)",
            "Lemma",
            "Theory",
            "Budget",
            "Completed",
            "Search",
            "Last Act (s) /\nDelay (n_upd)",
        ]

        self._change_trackers = {}

        self.log_keys = {"episodes"}
        self.special_windows = {
            "ModelStep/n_updates": 1,
            "ModelStep/cache_usage": 1,
            "ModelStep/skipped_updates": 1,
            "ModelStep/size_replay": 1,
            "EnvStep/num_envs": 1,
            "Timesteps": 1,
        }
        self.change_keys = ["Search/lemma_completes"]

        self.step_log_freq = recorder_config.log_freq
        self.last_ts = time()
        self.stats_buffer = dict()
        self.start_time = time()

    def get_window(self, key):

        if key.startswith("Search/"):
            window = 1
        elif key.startswith("SearchOverview/"):
            window = 1
        elif key.startswith("EnvStep/"):
            window = self.config.log_avg_window_env_step
        elif key.startswith("ModelStep/"):
            window = self.config.log_avg_window_model_step
        elif key.startswith("SearchState/"):
            window = 1
        elif key.startswith("System/"):
            window = 1
        else:
            window = self.config.log_avg_window
        return self.special_windows.get(key, window)

    def _setup_logger(self, path):
        self.logger = Logger(
            folder=path,
            output_formats=[HumanOutputFormat(sys.stdout, max_length=500)],
        )
        self.logger.log("Logging to %s", path)
        # self.logger.set_level(self.opt["log_level"])

    # def log_eval_stat(self, eval_stats, eval_time, step, is_best=False):
    #     log_dict = {
    #         "Eval/AvgTotalReward": eval_stats["mean_reward"],
    #         "Eval/StdTotalReward": eval_stats["std_reward"],
    #         "Eval/AvgLength": eval_stats["mean_length"],
    #         "Eval/StdLength": eval_stats["std_length"],
    #         "Eval/Time": eval_time,
    #         "Timesteps": step,
    #     }
    #     self.print_log(log_dict, step)
    #     if self.use_wandb:
    #         if is_best:
    #             self.log_eval_table(eval_stats, step)
    #         self.log_wandb(log_dict, step)

    # def log_eval_table(self, eval_stats, step):
    #     table = wandb.Table(
    #         columns=["Rank", "Mean Reward", "Std Reward", "Mean Length", "Std Length"],
    #         log_mode="MUTABLE",
    #     )
    #     table.add_data(
    #         self.opt["rank"],
    #         eval_stats["mean_reward"],
    #         eval_stats["std_reward"],
    #         eval_stats["mean_length"],
    #         eval_stats["std_length"],
    #     )
    #     wandb.log({"Eval": table})

    def _get_metric_value(self, key, metrics_dict):
        """Helper to find flattened or nested keys."""
        if key in metrics_dict:
            return metrics_dict[key]
        if "/" in key:
            parts = key.split("/")
            if len(parts) == 2:
                main, sub = parts
                if main in metrics_dict and isinstance(metrics_dict[main], dict):
                    return metrics_dict[main].get(sub)
        return None

    def log_timesteps_at_first_change(self, trigger_key, log_dict, current_step):
        """
        Logs the current 'Timesteps' (current_step) to W&B Summary
        the FIRST time 'trigger_key' changes value.
        """
        if not self.use_wandb:
            return

        # 1. Get current value of the trigger
        val = self._get_metric_value(trigger_key, log_dict)
        if val is None:
            return

        # 2. Initialize tracker if new
        if trigger_key not in self._change_trackers:
            self._change_trackers[trigger_key] = {"prev": val, "found": False}
            return

        # 3. Check for change
        tracker = self._change_trackers[trigger_key]

        if not tracker["found"]:
            if val != tracker["prev"]:
                # --- CHANGE DETECTED ---

                # Name: "timesteps_at_first_EnvStep_num_cases_change"
                category, key = trigger_key.split("/")
                summary_name = f"timesteps_at_first_{key}_change"

                # Update W&B Summary
                wandb.run.summary[f"{category}/{summary_name}"] = current_step

                # Lock it so we don't update again
                tracker["found"] = True

                # Optional: print to console for debugging
                # print(f"Logged {summary_name}: {current_step}")

            # Update tracker
            tracker["prev"] = val

    def log_step(self, log_dict, step):
        for key, val in log_dict.items():
            if isinstance(val, dict):
                for subkey, v in val.items():
                    subkey = f"{key}/{subkey}"
                    if subkey not in self.stats_buffer:
                        self.stats_buffer[subkey] = deque(
                            maxlen=self.get_window(subkey)
                        )
                    self.stats_buffer[subkey].append(v)
            else:
                if key not in self.stats_buffer:
                    self.stats_buffer[key] = deque(maxlen=self.get_window(key))
                self.stats_buffer[key].append(val)

        self.write_log(step)

    def write_log(self, step, force=False):
        if step % self.step_log_freq != 0 and not force:
            self.logged_step_already = False
            return

        if self.logged_step_already and not force:
            return

        self.logged_step_already = True
        for key, val in self.stats_buffer.items():
            if None in val:
                assert False, f"Key {key} has None value in stats buffer."
        log_dict = {k: np.mean(v) for k, v in self.stats_buffer.items() if len(v) > 0}
        log_dict["Timesteps"] = step
        if self.use_wandb:
            self.log_wandb(log_dict, step)

        self.print_log(log_dict, step)

    def setup_wandb(self, wandb_run):
        wandb_run.define_metric("Timesteps")
        # define a metric we are interested in the minimum of
        wandb_run.define_metric("ModelStep/loss", summary="min")
        wandb_run.define_metric("Search/tree_size", summary="min")

    def log_wandb(self, log_dict, step):
        # wandb.log(log_dict, step=step)
        for key in log_dict:
            if key not in self.log_keys:
                self.log_keys.add(key)
                wandb.define_metric(key, step_metric="Timesteps")

        wandb_table = wandb.Table(columns=self.system_headers, data=self.system_state)
        wandb.log({**log_dict, "System/State": wandb_table}, step=step, commit=True)

    def print_log(self, log_dict, step):
        log_str = f"Step {step}: "
        if not self.use_wandb:
            for key, val in log_dict.items():
                if isinstance(val, dict):
                    for subkey, v in val.items():
                        self.logger.record(subkey, v)
                        log_str += f"{subkey}: {v:.5f}, "
                else:
                    log_str += f"{key}: {val:.5f}, "
                    self.logger.record(key, val)
            now = time()
            self.logger.record("Time/elapsed_time", now - self.start_time)
            self.logger.record("Time/step_time", now - self.last_ts)
            try:
                self.logger.dump(step)
            except ValueError as e:
                print(f"Logging error at step {step}: {e}. ignoring and continuing.")
            # print(log_str)
            self.last_ts = time()

        # Replace None values with empty strings for display
        display_data = [
            [cell if cell is not None else "" for cell in row]
            for row in self.system_state
        ]
        print(
            "\n"
            + tabulate(
                display_data,
                headers=[
                    h.replace("Budget", "💰")
                    .replace("Completed", "✓")
                    .replace("Search", "🔍")
                    .replace("Last Act (s) /\nDelay (n_upd)", "⏱️")
                    for h in self.system_headers
                ],
                tablefmt="fancy_grid",
            )
            + "\n",
            flush=True,
        )

    def log_system_state(self, data: list[list[str | int | None]], step: int):
        self.system_state = data
        self.write_log(step)
