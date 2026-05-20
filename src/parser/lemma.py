import argparse
import os
from dataclasses import dataclass

from src.utils.load import LemmaType, load_spthy_file

DEFAULTS = {
    "curriculum": None,
    "protocol": None,
    "lemma": None,
    "heuristic": "f",
    "lemma_type": "both",
}


def parser_lemma_params(parser: argparse.ArgumentParser) -> None:
    """Adds arguments that specify lemma to the parser."""

    parser.add_argument(
        "--protocol",
        type=str,
        required=False,
        default=DEFAULTS["protocol"],
        help="Path to a theory .spthy file or to a folder with theory files. If a folder is given, We recusrsively look for .spthy files. (default: None)",
    )
    parser.add_argument(
        "--lemma",
        type=str,
        default=DEFAULTS["lemma"],
        required=False,
        help="Name of the lemma to prove. If not specified, all lemmas in the protocol(s) will be considered. (default: None)",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=DEFAULTS["heuristic"],
        help="Which heuristic to use. Options are: f (for file, whatever is specified in the file) and s,S,c,C,i,I as specified in https://tamarin-prover.com/manual/master/book/011_advanced-features.html (default: f)",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default=DEFAULTS["curriculum"],
        help="Path to curriculum file specifying the exact lemmas to train on. If set, protocol and lemma arguments are ignored.Expected File format: Each line should contain a path with a theory file and, optionally followed by a comma and the name of the lemma to train on. If the lemma name is omitted, all lemmas in the specified folder will be included. Example line: /path/to/theory.spthy,lemma_name (default: None)",
    )
    parser.add_argument(
        "--lemma_type",
        type=str,
        choices=["exists-trace", "all-traces", "both"],
        default=DEFAULTS["lemma_type"],
        help="Type of lemmas to include. Both does not filter (default: both)",
    )


@dataclass(frozen=True)
class LemmaConfig:
    theory_path: str
    lemma_name: str
    lemma_type: LemmaType
    theory_name: str
    diff_arg: bool
    suppress_output: bool
    heuristic: str
    side: str | None


def get_lemma_configs(args) -> list[LemmaConfig]:

    curriculum: list[tuple[str, str | None]] = []

    arg_curriculum = getattr(args, "curriculum", DEFAULTS["curriculum"])
    arg_protocol = getattr(args, "protocol", DEFAULTS["protocol"])
    arg_lemma = getattr(args, "lemma", DEFAULTS["lemma"])
    arg_heuristic: str = getattr(args, "heuristic", DEFAULTS["heuristic"])  # type: ignore
    arg_lemma_type = getattr(args, "lemma_type", DEFAULTS["lemma_type"])

    if arg_curriculum is not None:
        if arg_protocol is not None or arg_lemma is not None:
            print("Warning: --curriculum is set, --protocol and --lemma will be ignored.")
        with open(arg_curriculum, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and comments
                parts = line.split(",")
                theory_path = parts[0].strip()
                curriculum_path = os.path.dirname(arg_curriculum)
                theory_path = os.path.join(curriculum_path, theory_path)
                lemma_name = parts[1].strip() if len(parts) > 1 else None
                curriculum.append((theory_path, lemma_name))
    else:
        if arg_protocol is None:
            raise ValueError("protocol must be set if curriculum is not")
        curriculum = [(arg_protocol, arg_lemma)]

    # Resolve curriculum entries to individual .spthy file paths
    theory_files: list[tuple[str, str | None]] = []
    for protocol, lemma in curriculum:
        if os.path.isfile(protocol):
            theory_files.append((protocol, lemma))
        else:
            for root, _, files in os.walk(protocol):
                for file in files:
                    if file.endswith(".spthy"):
                        theory_files.append((os.path.join(root, file), lemma))

    lemma_configs: list[LemmaConfig] = []
    for theory_path, lemma in theory_files:
        theory_name, theory_file, diff_arg, lemmas = load_spthy_file(
            os.path.dirname(theory_path), os.path.basename(theory_path)
        )
        for lemma_name, lemma_type in lemmas:
            if lemma is None or lemma == lemma_name:
                lemma_configs.append(
                    LemmaConfig(
                        theory_path=theory_file,
                        lemma_name=lemma_name,
                        lemma_type=lemma_type,
                        theory_name=theory_name,
                        diff_arg=diff_arg,
                        suppress_output=True,
                        heuristic=arg_heuristic,
                        side=None,  # TODO side
                    )
                )

    lemma_configs = [
        config
        for config in lemma_configs
        if arg_lemma_type in ["both", config.lemma_type.value]
    ]

    print(
        f"Found {len(theory_files)} theory files with a total of {len(lemma_configs)} lemmas to evaluate."
    )

    return lemma_configs
