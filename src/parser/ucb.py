import argparse
from dataclasses import dataclass

DEFAULTS = {
    "pb_c_base": 100,
    "pb_c_init": 1.0,
    "value_discount": 0.99,
    "temperature": 10,
    "c_and": 8,
    "value_penalty": 8,
    "heuristic_weight": 0.0,
    "deactivate_model": False,
    "invert_and": False,
}


def parser_ucb_params(parser: argparse.ArgumentParser) -> None:
    """Adds UCB-specific hyperparameters to the parser."""
    # pb_c = [log((N_parent + pb_c_base + 1) / pb_c_base) + pb_c_init] * sqrt(N_parent) / (N_child + 1)
    # The log term grows with parent visits, increasing exploration pressure over time.
    # pb_c_base controls how slowly this growth happens. Set proportional to typical
    # visit counts for the log term to be meaningful (e.g., 100 for ~100 visits/node).
    # At visit counts << pb_c_base, the log term is negligible and pb_c_init dominates.
    parser.add_argument(
        "--pb_c_base",
        type=int,
        default=DEFAULTS["pb_c_base"],
        help=f"(default: {DEFAULTS['pb_c_base']})",
    )
    # Constant baseline for the exploration coefficient pb_c. Controls the overall
    # strength of the prior/exploration term relative to value_score. Higher values
    # make the prior more influential in action selection. Must be large enough for
    # prior_score to compete with value_score (which is ~0.9-1.0 with value_discount=0.99).
    parser.add_argument(
        "--pb_c_init",
        type=float,
        default=DEFAULTS["pb_c_init"],
        help=f"(default: {DEFAULTS['pb_c_init']})",
    )
    # value_score = value_discount ^ (-1 - value). Controls how much value differences
    # affect action selection. Close to 1.0 (e.g., 0.99) compresses the value_score
    # range, making value a weak tiebreaker — the prior drives selection and value
    # naturally gains influence as the network learns to produce meaningful differences.
    # Lower values (e.g., 0.5) amplify value differences, strongly exploiting
    # high-value children (suited for well-trained value networks).
    parser.add_argument(
        "--value_discount",
        type=float,
        default=DEFAULTS["value_discount"],
        help=f"(default: {DEFAULTS['value_discount']})",
    )
    # Flattens the model's prior distribution: prior_probs = exp(logprobs / temperature).
    # Higher temperature = more uniform prior (less trust in model predictions).
    # Also flattens the heuristic bias — the effective heuristic strength is
    # heuristic_weight / temperature. High temperature suits well-calibrated pretrained
    # models; low temperature suits untrained models where the heuristic should dominate.
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULTS["temperature"],
        help=f"(default: {DEFAULTS['temperature']})",
    )
    # Multiplier for the prior_score at AND nodes. AND nodes have uniform priors
    # (1/num_cases), so this compensates to keep exploration meaningful. Scaled by
    # num_children internally, so the effective boost is c_and regardless of branching
    # factor. Controls how much AND node selection favors exploration (round-robin
    # across subgoals) vs exploitation (focus on high-value subgoals).
    parser.add_argument(
        "--c_and",
        type=float,
        default=DEFAULTS["c_and"],
        help=f"(default: {DEFAULTS['c_and']})",
    )
    # Pessimistic offset applied to unvisited children's value estimate:
    # value = parent.value - value_penalty. Makes unexplored nodes look worse,
    # encouraging the system to revisit known children over trying new ones.
    # Higher = more exploitation of visited children; lower = more willing to
    # explore unvisited children. Interacts with value_discount — with value_discount
    # close to 1.0, even large penalties produce small value_score differences.
    parser.add_argument(
        "--value_penalty",
        type=int,
        default=DEFAULTS["value_penalty"],
        help=f"(default: {DEFAULTS['value_penalty']})",
    )
    parser.add_argument(
        "--invert_and",
        action="store_true",
        help="Whether to invert the value score for AND nodes. This encourages to explore subgoals that seem harder",
    )
    parser.add_argument(
        "--invert_and_sweep",
        type=lambda x: str(x).lower() == "true",
        help="Whether to invert the value score for AND nodes (sweep compatible).",
        default=None,
    )
    parser.add_argument(
        "--deactivate_model",
        action="store_true",
        help="Whether to deactivate the model and only use heuristic values and visit counts.",
    )
    parser.add_argument(
        "--deactivate_model_sweep",
        type=lambda x: str(x).lower() == "true",
        help="Whether to deactivate the model and only use heuristic values and visit counts (sweep compatible).",
        default=None,
    )
    parser.add_argument(
        "--heuristic_weight",
        type=float,
        default=DEFAULTS["heuristic_weight"],
        help=f"Weight for heuristic rank bias in the prior. Effective strength is heuristic_weight / temperature, so set heuristic_weight = desired_effective_weight * temperature to compensate for temperature flattening. (default: {DEFAULTS['heuristic_weight']})",
    )


@dataclass(frozen=True)
class UCBConfig:
    pb_c_base: float
    pb_c_init: float
    value_discount: float
    temperature: float
    c_and: float
    value_penalty: int
    invert_and: bool
    heuristic_weight: float
    deactivate_model: bool = False


def get_ucb_config(args) -> UCBConfig:
    invert_and = getattr(args, "invert_and", False)
    if getattr(args, "invert_and_sweep", None) is not None:
        invert_and = args.invert_and_sweep

    deactivate_model = getattr(args, "deactivate_model", False)
    if getattr(args, "deactivate_model_sweep", None) is not None:
        deactivate_model = args.deactivate_model_sweep

    return UCBConfig(
        pb_c_base=getattr(args, "pb_c_base", DEFAULTS["pb_c_base"]),
        pb_c_init=getattr(args, "pb_c_init", DEFAULTS["pb_c_init"]),
        value_discount=getattr(args, "value_discount", DEFAULTS["value_discount"]),
        temperature=getattr(args, "temperature", DEFAULTS["temperature"]),
        c_and=getattr(args, "c_and", DEFAULTS["c_and"]),
        value_penalty=getattr(args, "value_penalty", DEFAULTS["value_penalty"]),
        invert_and=invert_and,
        heuristic_weight=getattr(
            args, "heuristic_weight", DEFAULTS["heuristic_weight"]
        ),
        deactivate_model=deactivate_model,
    )
