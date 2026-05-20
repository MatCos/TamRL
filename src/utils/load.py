import os
import re
from dataclasses import dataclass
from enum import Enum

LEMMA_PATTERN = re.compile(
    r"lemma\s*((?!NOEXTRACT_)\w*)\s*(\[.*?\])?\s*:\s*(exists-trace)?\s*",
    flags=re.DOTALL,
)
DIFF_PATTERN = re.compile(r"diff\(.*?,.*?\)")
THEORY_PATTERN = re.compile(r"theory\s*(\w*)?\s*(\n|.)*?begin")


class LemmaType(Enum):
    EXISTS = "exists-trace"
    FORALL = "all-traces"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TheoryConfig:
    theory_path: str
    theory_name: str
    diff_arg: bool
    lemmas: list[str]


def open_spthy_file(
    protocol_path: str, file_name: str | None = None
) -> tuple[str, str]:
    spthy_files = [
        f
        for f in os.listdir(protocol_path)
        if f.endswith(".spthy") and (f == file_name or file_name is None)
    ]
    if not spthy_files:
        raise FileNotFoundError(
            f"No .spthy file found in the specified protocol_path directory. {protocol_path}, file_name: {file_name}"
        )
    theory_file = os.path.join(protocol_path, spthy_files[0])

    with open(theory_file, "r", encoding="utf-8") as file:
        content = file.read()
        content = remove_c_style_comments(content)

    return content, theory_file


def parse_theory_name(content: str, theory_file: str) -> str:
    match = THEORY_PATTERN.search(content)
    if match:
        theory_name = match.group(1)
    else:
        raise ValueError(f"Could not find theory name in the file {theory_file}.")
    return theory_name


def parse_diff_arg(content: str) -> bool:
    return DIFF_PATTERN.search(content) is not None


def parse_lemmas(content: str, theory_file: str) -> list[tuple[str, LemmaType]]:
    lemma_matches = re.findall(LEMMA_PATTERN, content)
    lemmas = []
    for name, _, _ in lemma_matches:
        lemmas.append((name, extract_lemma_type(content, name, theory_file)))
    return lemmas


def extract_lemma_type(content: str, lemma_name: str, theory_file: str) -> LemmaType:
    lemma_type = None
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"lemma {lemma_name}"):
            lemma_type = LemmaType.FORALL  # Default if no type
            # Check declaration line and subsequent lines for quantifier
            if "exists-trace" in line:
                lemma_type = LemmaType.EXISTS
            elif "all-traces" not in line:
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line.startswith("lemma"):
                        break
                    if "exists-trace" in next_line:
                        lemma_type = LemmaType.EXISTS
                        break
                    if "all-traces" in next_line:
                        break
            break

    if lemma_type is None:
        raise ValueError(f"lemma {lemma_name} not found in {theory_file}")
    return lemma_type


def load_spthy_file(
    protocol_path: str, file_name: str | None = None
) -> tuple[str, str, bool, list[tuple[str, LemmaType]]]:

    content, theory_file = open_spthy_file(protocol_path, file_name)
    theory_name = parse_theory_name(content, theory_file)
    diff_arg = parse_diff_arg(content)

    lemmas = parse_lemmas(content, theory_file)

    return theory_name, theory_file, diff_arg, lemmas


def remove_c_style_comments(text: str) -> str:
    # Remove /* ... */ comments (including multiline)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove // ... comments (single line)
    text = re.sub(r"//.*", "", text)
    return text
