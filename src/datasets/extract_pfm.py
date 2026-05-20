from src.environment.environment import State


def extract_pfm(state: State, timeout) -> list[tuple[str, tuple[dict, ...]]]:
    pfms = state.proof_methods

    proofmethods: list[tuple[str, tuple[dict, ...]]] = []
    for proofmethod in pfms:
        if proofmethod["name"] == "Simplify":
            proofmethods.append(("simplify", ()))
        elif proofmethod["name"] == "Induction":
            proofmethods.append(("induction", ()))
        elif proofmethod["goal"]["type"] == "Premise":
            proofmethods.append(("premise", proofmethod["goal"]["args"]))
        elif proofmethod["goal"]["type"] == "Chain":
            proofmethods.append(("chain", proofmethod["goal"]["args"]))
        elif proofmethod["goal"]["type"] == "Action":
            proofmethods.append(("action", proofmethod["goal"]["args"]))
        elif proofmethod["goal"]["type"] == "Split":
            proofmethods.append(("split", proofmethod["goal"]["args"]))
        elif proofmethod["goal"]["type"] == "Disj":
            proofmethods.append(("disj", proofmethod["goal"]["args"]))
        elif proofmethod["goal"]["type"] == "Subterm":
            proofmethods.append(("subterm", proofmethod["goal"]["args"]))
        else:
            raise NotImplementedError(
                f"Proofmethod type {proofmethod['goal']['type']} not implemented"
            )
    return proofmethods
