import argparse
from dataclasses import dataclass

DEFAULTS = {
    "tokenizer": "roberta-base",
    # "tokenizer": "Salesforce/codegen-350M-mono",
    "tokenizer_key": "name",
    "tokenizer_max_length": 512,
    "replace_var": None,
    "tokenizer_cache_size": 100000,
    "id_range_var": 50,
}


def parser_tokenizer_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULTS["tokenizer"],
        help=f"Tokenizer to use for processing the data (default: {DEFAULTS['tokenizer']})",
    )
    parser.add_argument(
        "--tokenizer_key",
        type=str,
        default=DEFAULTS["tokenizer_key"],
        help=f"Key to use for to_str in tokenizer (default: {DEFAULTS['tokenizer_key']})",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=DEFAULTS["tokenizer_max_length"],
        help=f"Maximum length for tokenizer (default: {DEFAULTS['tokenizer_max_length']})",
    )
    parser.add_argument(
        "--replace_var",
        type=str,
        default=DEFAULTS["replace_var"],
        help=f"Replace variables with random identifiers (default: {DEFAULTS['replace_var']})",
        choices=["user", "all", None],
    )
    parser.add_argument(
        "--tokenizer_cache_size",
        type=int,
        default=DEFAULTS["tokenizer_cache_size"],
        help=f"Cache size for tokenizer (default: {DEFAULTS['tokenizer_cache_size']})",
    )
    parser.add_argument(
        "--id_range_var",
        type=int,
        default=DEFAULTS["id_range_var"],
        help=f"Maximum number of types to replace with random identifiers. (default: {DEFAULTS['id_range_var']})",
    )


@dataclass(frozen=True)
class TokenizerConfig:
    max_length: int
    cache_size: int
    model_name: str
    replace_var: str
    key: str
    id_range: int


def get_tokenizer_config(args) -> TokenizerConfig:
    return TokenizerConfig(
        max_length=getattr(
            args, "tokenizer_max_length", DEFAULTS["tokenizer_max_length"]
        ),
        cache_size=getattr(
            args, "tokenizer_cache_size", DEFAULTS["tokenizer_cache_size"]
        ),
        model_name=getattr(args, "tokenizer", DEFAULTS["tokenizer"]),
        replace_var=getattr(args, "replace_var", DEFAULTS["replace_var"]),
        key=getattr(args, "tokenizer_key", DEFAULTS["tokenizer_key"]),
        id_range=getattr(args, "id_range_var", DEFAULTS["id_range_var"]),
    )
