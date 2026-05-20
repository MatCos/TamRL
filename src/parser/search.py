import argparse
from dataclasses import dataclass

DEFAULTS = {
    "search_budget": 1000,
    "search_budget_increase_factor": 1.5,
    "num_lemma_completes": 5,
    "num_searches": 20,
    "max_workers_per_lemma": 4,
    "first_step": "search",
    "expand_top_n": 0,
}


def parser_search_params(parser: argparse.ArgumentParser) -> None:
    """Adds search-specific hyperparameters to the parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
    """
    parser.add_argument(
        "--search_budget",
        type=int,
        default=DEFAULTS["search_budget"],
        help=f"Maximum number of steps per iddf search. Search wil restart with fresh bound if budget is exceeded. (default: {DEFAULTS['search_budget']})",
    )
    parser.add_argument(
        "--search_budget_increase_factor",
        type=float,
        default=DEFAULTS["search_budget_increase_factor"],
        help=f"Factor to increase search budget by if failed to proof. (default: {DEFAULTS['search_budget_increase_factor']})",
    )
    parser.add_argument(
        "--num_lemma_completes",
        type=int,
        default=DEFAULTS["num_lemma_completes"],
        help=f"Number of full proofs before stopping training. Will train until either this or num_searches is reached. (default: {DEFAULTS['num_lemma_completes']})",
    )
    parser.add_argument(
        "--num_searches",
        type=int,
        default=DEFAULTS["num_searches"],
        help=f"Number of searches before stopping training. Will train until either this or num_lemma_complete is reached. (default: {DEFAULTS['num_searches']})",
    )
    parser.add_argument(
        "--max_workers_per_lemma",
        type=int,
        default=DEFAULTS["max_workers_per_lemma"],
        help=f"Maximum number of parallel workers for the same lemma. A lemma is still assigned more workers if it is the only one remaining in the queue. (default: {DEFAULTS['max_workers_per_lemma']})",
    )
    parser.add_argument(
        "--first_step",
        type=str,
        choices=["search", "heuristic", "induction", "simplify"],
        default=DEFAULTS["first_step"],
        help="How to handle the first proof step (Induction/Simplify). "
        "'search': MCTS searches over it (default). "
        "'heuristic': pick Tamarin's top-ranked (index 0). "
        "'induction': always pick Induction. "
        "'simplify': always pick Simplify.",
    )
    parser.add_argument(
        "--expand_top_n",
        type=int,
        default=DEFAULTS["expand_top_n"],
        help=f"Only expand the top N children by prior during node expansion. "
        f"Remaining children become virtual actions that are materialized on demand "
        f"when UCB selects them. 0 means expand all (default: {DEFAULTS['expand_top_n']})",
    )


@dataclass(frozen=True)
class SearchConfig:
    budget: int
    budget_increase_factor: float
    num_lemma_completes: int
    num_searches: int
    max_workers_per_lemma: int
    first_step: str
    expand_top_n: int


def get_search_config(args) -> SearchConfig:
    return SearchConfig(
        budget=getattr(args, "search_budget", DEFAULTS["search_budget"]),
        budget_increase_factor=getattr(
            args,
            "search_budget_increase_factor",
            DEFAULTS["search_budget_increase_factor"],
        ),
        num_lemma_completes=getattr(
            args, "num_lemma_completes", DEFAULTS["num_lemma_completes"]
        ),
        num_searches=getattr(args, "num_searches", DEFAULTS["num_searches"]),
        max_workers_per_lemma=getattr(
            args, "max_workers_per_lemma", DEFAULTS["max_workers_per_lemma"]
        ),
        first_step=getattr(args, "first_step", DEFAULTS["first_step"]),
        expand_top_n=getattr(args, "expand_top_n", DEFAULTS["expand_top_n"]),
    )
