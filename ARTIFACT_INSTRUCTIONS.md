# Less Effort, Shorter Proofs: Reinforcement Learning for Security Protocol Analysis in Tamarin

This is the artifact for the paper *"Less Effort, Shorter Proofs: Reinforcement Learning for Security Protocol Analysis in Tamarin"*.

## About this Artifact

The artifact is distributed as a Docker image together with the pre-computed results:

| File                       | Contents                                                                         |
| -------------------------- | -------------------------------------------------------------------------------- |
| `tamarin-rl-cpu.tar.gz`    | Docker image with the RL framework, our Tamarin fork, and all 16 protocol models |
| `results.zip`              | Pre-computed results: RL run data, proofs, and the Tamarin baselines             |
| `ARTIFACT_INSTRUCTIONS.md` | This file                                                                        |


All three files are archived together at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22705842.svg)](https://doi.org/10.5281/zenodo.22705842)

Both components we contribute ship **as source inside the image**, each as a git checkout at the exact commit its binaries were built from, and each is also available as a public repository:

| Component                                                  | Path in the image           | Public repository                                               |
| ---------------------------------------------------------- | --------------------------- | --------------------------------------------------------------- |
| The reinforcement learning framework                       | `/workspace/TamRL`          | [RL Code](https://github.com/MatCos/TamRL)                      |
| Our fork of the Tamarin prover with the stateless HTTP API | `/workspace/Tamarin-ML-API` | [Tamarin API](https://github.com/niklasmedinger/Tamarin-ML-API) |


Two folders are mounted from outside so that everything the container produces is written back to your host and survives the container: `results/`, holding the pre-computed results and the tables, plots and summaries regenerated from them, and `output/`, where training runs you start yourself write their checkpoints and proof trees.

```bash
unzip results.zip                                # creates ./results
mkdir -p output                                  # for new runs
docker load -i tamarin-rl-cpu.tar.gz

docker run --rm -it --platform linux/amd64 \
  -v "$PWD/results:/workspace/TamRL/results" \
  -v "$PWD/output:/workspace/TamRL/output" \
  --user "$(id -u):$(id -g)" \
  tamarin-rl:cpu
```

This drops you into a shell in `/workspace/TamRL`, the repository root. All commands in this README are run from there. You can optionally pass `--network none`, the docker needs no internet access. `--user` is optional and ensures that the files written outside the container belong to your user rather than docker's root.

> [!TIP]
> We split the instructions into 3 parts.
> 1) **Reproducing tables and plots from pre-computed results**, plus smoke tests that solve a few lemmas from scratch. We structured the artifact so that every number in the paper's four result tables (Tables 1–4) can be re-derived from the recorded run data. Regenerating all tables and plots takes about 15 seconds on a consumer-grade laptop and needs no Tamarin installation. The smoke tests additionally require the Tamarin API and are a quick functional test that the RL training pipeline works and produces proofs; each takes a few minutes.
> 2) **Reproducing the results from scratch**, i.e. running the baselines and the RL experiments that produce the data used in part 1. Given the extensive evaluation on a large variety of protocols, this needs substantial resources and time, and goes far beyond the scope of a typical artifact evaluation. We provide it for transparency and to allow interested researchers to reproduce the results on their own hardware.
> 3) **Applying the RL framework to new protocols.** We provide quick guidelines on how to apply our approach to new protocols. They support no claim in the paper and are intended to ease the use of our framework for work of your own.


## Claims and Evaluation Roadmap

The paper makes three contributions:

1. **Reusable API for Tamarin** — A stateless HTTP API that allows programmatic interaction with Tamarin's proof search (see the [Tamarin API](https://github.com/niklasmedinger/Tamarin-ML-API) repository).
2. **RL agent for proof search** — An RL agent combining MCTS with a neural heuristic that learns from completed subproofs to guide Tamarin's proof search.
3. **Evaluation on 16 case studies** — Our approach solves more lemmas automatically than Tamarin's default heuristics, nearly matches human-engineered heuristics on the most complex models, and consistently produces shorter proofs.

This artifact supports these contributions as follows:

| Contribution                                                                                                               | Supported by                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Reusable stateless HTTP API for Tamarin's proof search                                                                     | Full source code in `/workspace/Tamarin-ML-API` (docker) and in the [Tamarin API](https://github.com/niklasmedinger/Tamarin-ML-API) repository |
| Our approach solves more lemmas than Tamarin's default heuristics and nearly matches human-engineered heuristics (Table 1) | Section: *Reproducing Tables and Plots*                                                                                                        |
| Our approach produces shorter proofs than both default and human-engineered heuristics (Table 2)                           | Section: *Reproducing Tables and Plots*                                                                                                        |
| Tamarin baselines are faithfully reported                                                                                  | Section: *Run the baselines*                                                                                                                   |
| The RL training pipeline is functional and produces proofs                                                                 | Section: *Run the RL experiments*                                                                                                              |
| Generated proofs are valid Tamarin proofs                                                                                  | Section: *Verifying proofs*                                                                                                                    |

## Repository Structure


Inside the container this tree is `/workspace/TamRL`, and it has two siblings:

```
/workspace/
  TamRL/                This repository (the shell starts here)
  Tamarin-ML-API/       Source of our Tamarin fork, as a git checkout
```

### RL Repository

The Reinforcement Learning components are in `TamRL/`, with the following structure:

```
src/                    Source code for the RL framework
  main/run.py           Entry point (train command)
  training/trainer.py   Training orchestrator, multi-worker management
  rl/mcts.py            MCTS search with UCB child selection
  rl/ac_agent.py        Actor-Critic agent (inference + model updates)
  models/               Neural network architectures (Transformer-based Actor-Critic)
  environment/          RL environment wrapping the Tamarin HTTP client
  client/               HTTP client for the Tamarin API
  parser/               Configuration system (dataclass + argparse + YAML)
  datasets/             Tokenizer and proof method extraction
eval/                   Protocol models
  <protocol>/           Protocol-specific files (.spthy), one folder per variant
  tamarin_wrapper.py    Runs a Tamarin baseline for one .spthy file
  Makefile              Batch-runs all baselines in parallel
scripts/
  result_collection.py  Aggregates results and generates summaries, tables, plots
  results_lib/          Helper modules for result_collection.py
tests/                  Unit tests
```

The pre-computed results are not part of the source tree. They live in a separate `results/` folder (shipped as `results.zip`, mounted into the container, see [Setup](#setup)):

```
results/
  best/                 Best run per protocol: raw.json, proofs, tables, plots (once generated from the shipped data)
  others/               All further runs: raw.json, proofs
  all/                  Combined (derived) view over best/ + others/: summary.yaml, hyperparameters.csv, tables, plots
  tamarin/              Tamarin baselines
    <protocol>/<variant>/recent_results_<model>.csv
```

### Tamarin HTTP API

`Tamarin-ML-API/` is a fork of the Tamarin prover, branched off at commit `bb1a083f6092f5827c8bea6980caf5927578b9df`. The changes relevant to the API are in its `src/Web/` directory; the rest of the repository sees only minor edits where necessary, for instance the pretty-printing of proof methods for the RL agent. The executable is called `tamarin-prover-json`.

Note that the API changes currently break Tamarin's GUI. To use the GUI, build the fork's `develop` branch or Tamarin from the upstream repository instead; this is also why the image ships upstream's `tamarin-prover` next to the fork's `tamarin-prover-json` (see [Run the baselines](#21-run-the-baselines)).

### Protocol names

The `eval/` folder stubs differ slightly from the official protocol names used in the paper. The mapping is:

| Paper               | `eval/` folder             | Tamarin theory                 |
| ------------------- | -------------------------- | ------------------------------ |
| 5G AKA              | `5G_AKA/`                  | `5G_aka_fix`                   |
| 5G Handover EPS N26 | `5G_handover_5gs_to_eps/`  | `5gs_to_eps_over_n26_handover` |
| 5G Handover XN      | `5G_handover_xn/`          | `5G_xn_handover`               |
| FOO Eligibility     | `foo/`                     | `FOO_Eligibility`              |
| KAS2 eCK            | `KAS2/`                    | `KAS2_eCK`                     |
| NAXOS eCK           | `Naxos_non_pfs/`           | `NAXOS_eCK`                    |
| NAXOS eCK PFS       | `Naxos/`                   | `NAXOS_eCK_PFS`                |
| PKCS11 AEAD         | `gcm/`                     | `PKCS11_aead`                  |
| Signal              | `signal_looping_for_good/` | `SignalRevealing`              |
| SPDM                | `spdm/`                    | `composition_spdm`             |
| Tutorial            | `Tutorial/`                | `Tutorial`                     |
| UM PFS              | `UM/`                      | `UM_PFS`                       |
| Wireguard           | `wireguard/`               | `wireguard`                    |
| WPA2                | `wpa2/`                    | `wpa2_four_way_handshake`      |
| YubiKey             | `yubikey/`                 | `Yubikey`                      |
| YubiKey HSM         | `yubikey_hsm/`             | `YubikeyHSM`                   |

## Setup

The Docker image ships everything needed to run the RL framework and the Tamarin baselines.

> [!NOTE]
> The image is **linux/amd64 and CPU-only**. It runs natively on x86-64 Linux, which is what we recommend. On Apple silicon or Windows it runs under emulation, which works but is significantly slower. There is no CUDA or MPS inside the container. For actual training runs we recommend a proper setup on respective hardware. Docker Desktop runs containers with capped CPU and memory resources, no matter what the docker run command specifies. Change under *Settings → Resources*.


Alternative native installation instructions (no docker) are available directly at the source repositories: [RL Framework](https://github.com/MatCos/TamRL) and [Tamarin API](https://github.com/niklasmedinger/Tamarin-ML-API). We refer to the corresponding README for instructions.

### Hardware

Regenerating tables and plots has no special requirements. The timings quoted in Part 1 of these instructions are for a 10-core Apple M-series laptop, running the image under amd64 emulation unless stated otherwise. Reproducing results from scratch requires more resources. As reported in Section 6 of the paper, the results were produced on:

- *Tamarin baselines*: Intel Xeon E5-4650L 2.60 GHz, 1 TB RAM, 8 cores per Tamarin call.
- *RL experiments*: one NVIDIA A100 (40 GB), up to 64 cores AMD Rome 7742 2.25 GHz, 250 GB RAM.

These are the specifications of the machines we had available, not requirements. Running without a GPU is generally possible, but we have not systematically evaluated consumer-grade CPU-only hardware; expect longer run times. The most important resource requirement is memory: we recommend **at least 60 GB of combined memory**.

Two settings dominate memory use. The training commands below pass both explicitly, at the values used for the paper's runs:

- `--num_envs` — training launches one Tamarin process per environment, so `--num_envs=8` means 8 concurrent provers and environments. Reduce to one or two if memory is constrained; this should generally achieve the same results, with significantly longer runtimes.
- `--cache_max_memory_mb` — an upper bound on the Tamarin state cache, which defaults to `100000`, i.e. **100 GB**. Nothing is preallocated: the cache is an LRU that grows on demand and only starts evicting once the bound is reached. On a machine near the 60 GB recommendation, lower it substantially.

### Running the tests

```bash
python -m pytest tests -p no:cacheprovider -q
```

This runs all tests from the repository root; it also checks whether the local setup is functional. Running takes up to two minutes. They cover the full RL pipeline, as well as the interface between the RL agent and Tamarin. `tests/test_tamarin_api.py` in particular starts a `tamarin-prover-json` server and proves the four lemmas of the (trivial) Tutorial theory through the HTTP API, so it tests the whole setup end-to-end.


## 1. Reproducing Tables and Plots (from pre-computed results)

### 1.1 Tamarin baselines

Tamarin baselines are included in `results/tamarin/<protocol>/<variant>/`. There are up to three variants per protocol, corresponding to the model files under `eval/<protocol>/<variant>/`:

- **`original`** — the unaltered model as published by its authors, including all human-written tactics, `heuristic:` annotations, and oracle files.
- **`base_s`** — the same model with all tactics, oracles, and heuristic annotations removed, so that Tamarin's default *smart* heuristic `s` is used.
- **`base_c`** — the same stripped model, but annotated with `heuristic: c`, the *consecutive* (classic) heuristic.

Protocols where `base_s/` is missing are those where the `original` model already uses the default `s` heuristic.


Each variant folder contains a CSV with one row per lemma:

```
<lemma_name>, <result>, <time_seconds>, <proof_tree_size>, <flags>
```

Here `<result>` is `True`/`False` (proved/disproved) or `timeout`, `<time_seconds>` is wall-clock time, and `<proof_tree_size>` is the number of proof steps (`-1` on timeout).

### 1.2 RL results

Our pre-computed RL results are in `results/best/` and `results/others/`, two subsets covering 68 runs in total:

- **`best/`**: The best-performing run per protocol — the configuration that solves the most lemmas (ties broken by proof time and proof size). 16 runs, one per protocol.
- **`others/`**: Runs that are not best overall but still contribute unique results — they solve at least one lemma the best run does not, or achieve the best time or proof size on some lemma. 52 runs.
- **`all/`**: A combined view of `best/` and `others/`, i.e. all 68 runs. The paper's tables are generated from this view (see [Interpreting results](#14-interpreting-results)).

Runs where every solved lemma is also solved faster/smaller by another run are excluded from the artifact as they do not contribute to the paper's results.

Each run carries a `_meta.state` of `finished` (48 runs), `crashed` or `failed` (20): the latter two mean the job hit its scheduler wall-clock limit and was killed, not that training malfunctioned.

Each subset contains:
- `raw.json` — Per-run, per-lemma metrics. Each entry is keyed by run ID and contains a `_meta` object (full hyperparameter config) and per-protocol results. Each lemma records `completes` (number of successful completions), `first_completion_time` (seconds), `tree_size` (proof steps), and `type` (`all-traces` or `exists-trace`).
- `artifacts/<run_id>/` — Proof artifacts for each run:
  - `proof_trees/` — Proof trees from successful completions, as `.json` (serialized tree) and `.spthy` (Tamarin proof script).

### 1.3 Generating summaries, tables, and plots

The command below regenerates the paper's tables and plots from the pre-computed results, run from the repository root.

```bash
python scripts/result_collection.py --folder results --steps 4,6,8 -y
```

This takes about 15 seconds and runs three steps:
- **Step 4** — (Re-)aggregate raw results into `summary.yaml` and `hyperparameters.csv` per subset, and create an `all/` folder combining `best/` and `others/`. This is already done in the shipped `results.zip`, but is included for completeness.
- **Step 6** — Generate the paper's tables as Markdown (`.md`).
- **Step 8** — Generate heatmap plots showing which run solved which lemma, the proof size, and the time taken.

> [!NOTE]
> The remaining steps (1, 2, 5, 7) were only used during development: they mostly fetch raw runs directly from our infrastructure. They are not needed for the artifact — every output used in the paper is regenerated by steps 4, 6, and 8. They might, however, be convenient for a user when running their own experiments.

Output is written to `results/best/`, `results/others/`, and `results/all/` (combined).

### 1.4 Interpreting results

After running the result collection steps, each subset folder has a `tables/` directory with the same file names, computed over the runs that subset contains. The paper's four tables are:

| Paper                                                  | Markdown table                                 |
| ------------------------------------------------------ | ---------------------------------------------- |
| Table 1 (Sec. 6) — lemmas solved                       | `results/all/tables/experiment1_table.md`      |
| Table 2 (Sec. 6) — average proof size (solved by both) | `results/all/tables/proofsize_table.md`        |
| Table 3 (App. D) — hyperparameters                     | `results/all/tables/hyperparameters_table.md`  |
| Table 4 (App. D) — wall-clock time                     | `results/all/tables/time_table.md`             |

**Tables 2 and 4 take the per-lemma best across all 68 runs.**
- *Proof size (Table 2)* — minimum `tree_size` per lemma, averaged over the protocol's lemmas. The intersection columns restrict the mean to lemmas solved by both the RL system and the respective baseline, so paired columns always cover the same lemma set.
- *Wall-clock time (Table 4)* — minimum `first_completion_time` per lemma, then the **maximum** over the protocol's lemmas, i.e. the point at which the protocol is fully proved. This is not the runtime of any single run: the 19h 48m for 5G AKA combines six runs, whereas the fastest single run solving all 13 lemmas needs 43h 19m.

**Tables 1 and 3 report the best single configuration per protocol**, selected from all 68 candidates. For Table 1 `best/` yields an identical `experiment1_table.md`, since the selected run is exactly the one `best/` contains. Table 3 differs in one place: WPA2 has two runs tied at 73 solved lemmas; `all/` reports both (the `†` rows in the paper), `best/` only the pre-selected one.

The claim that a second configuration solves one additional WPA2 lemma, bringing the union to 74 (Section 6, *Less Effort*), is read off `results/all/tables/lemma_completion_overview.md`: column `ML (any)` (74) next to `ML (best run)` (73).

The remaining files in `tables/` are supplementary: not used in the paper, but the same data at finer granularity. `plots/heatmap_*.png` are diagnostic per-protocol plots showing which run solved which lemma, with proof size and time.


## 2. Reproducing results


### 2.1 Run the baselines

Running the Tamarin baselines requires the [Tamarin prover](https://github.com/niklasmedinger/Tamarin-ML-API) installed and available as `tamarin-prover-json` on `PATH`.

The wrapper resolves model paths relative to the current directory, so these two commands are run **from `eval/`** rather than from the repository root:

a) All 4 lemmas are solved, takes about a minute:

```bash
cd eval
python tamarin_wrapper.py yubikey_hsm/original/yubikey_hsm.spthy \
  -t 7200 -c 8 -j 4 --resultfolder=../results/tamarin/yubikey_hsm/original
```

b) The lemma times out, so we cut the timeout down from two hours to two minutes:

```bash
cd eval
python tamarin_wrapper.py 5G_handover_5gs_to_eps/base_s/5G_handover_5gs_to_eps.spthy \
  -l injectiveagreement_enb_ue_k_enb \
  -t 120 -c 4 --resultfolder=../results/tamarin/5G_handover_5gs_to_eps/base_s
```

`-t` is the per-lemma timeout in seconds, `-c` the cores per Tamarin call, `-j` the lemmas run concurrently (default 1), `-l` restricts the run to a comma-separated list of lemmas. `-c` and `-j` multiply, so lower `-c` to 2 on a laptop.

The wrapper prints one row per lemma — lemma, result (`True`/`False`/`timeout`), seconds, proof size (`-1` on timeout) — and writes them to `<resultfolder>/recent_results_<model>.csv`, overwriting the shipped baseline CSV. To restore the shipped values, re-extract `results.zip`, or point `--resultfolder` somewhere else. Expect `True` for all 4 lemmas in (a), and in (b) a single `timeout` row after 120 s. Replacing `base_s` with `base_c` gives the same result for the `c` heuristic.

### 2.2 Run the RL experiments

Training requires the [Tamarin API](https://github.com/niklasmedinger/Tamarin-ML-API). Install natively as described in the repository or use the pre-built version in the Docker image. Tamarin instances are automatically launched by the training script.

All training commands must be run **from the repository root**.

Reproducing all our results require multiple GPU-days on serious hardware, so we provide two small examples that are small enough to run on a laptop. While chosen to be small, they still exercise the full RL pipeline — guided search, proof extraction, and independent verification. Both of the examples show (anecdotally) one claim from the paper.

a) **YubiKey HSM** (`eval/yubikey_hsm/original`, 4 lemmas) — *shorter proofs*. All approaches solve all 4 lemmas, but ours needs 20.8 proof steps on average (~40 for this smaller command) against 570 for `original`/`s` and 415 for `c` (Table 2). RL run: 49 s (Table 4), 4min on an Apple M-series laptop (10 cores), 25min in the container under emulation (!)

   ```bash
python -m src.main.run train \
  --protocol=eval/yubikey_hsm/original \
  --batch_size=32 --min_usage_to_update=5 --replay_buffer_capacity=100000 \
  --warmup_fraction=0 --backup_request_timeout=600 --cache_max_memory_mb=20000 \
  --num_envs=8 --request_timeout=120 --stall_frequency=1 \
  --learning_rate=0.0001 --branch_penalty=0 --time_penalty=0.1 \
  --time_penalty_clip=180 --timeout_penalty=10 --search_budget=1000 \
  --search_budget_increase_factor=1.75 --expand_top_n=50 --first_step=search \
  --max_workers_per_lemma=4 --num_lemma_completes=1 --num_searches=5 \
  --tokenizer_cache_size=100000 --tokenizer_max_length=512 --tokenizer=roberta-base \
  --dim_transformer_feedforward=1024 --dropout=0.2 --n_transformer_head=8 \
  --pfm_dim=512 --pfm_layers=2 --transformer_layers=4 \
  --c_and=64 --heuristic_weight=0.3 --pb_c_base=3200 --pb_c_init=0.001 \
  --temperature=10 --value_discount=0.99 --value_penalty=8
```

b) **5G Handover EPS N26**, single lemma `injectiveagreement_enb_ue_k_enb` (`eval/5G_handover_5gs_to_eps/base_s`) — *less effort*. Both default heuristics time out on this lemma after two hours; ours proves it in ~4 min on an Apple M-series laptop (10 cores). The full protocol (8 lemmas, 3 min 25 s in Table 4) is not laptop-scale,  so we single out one of the five lemmas the default heuristics fail on.

   ```bash
python -m src.main.run train \
  --protocol=eval/5G_handover_5gs_to_eps/base_s \
  --lemma=injectiveagreement_enb_ue_k_enb \
  --batch_size=32 --min_usage_to_update=5 --replay_buffer_capacity=100000 \
  --warmup_fraction=0 --backup_request_timeout=600 --cache_max_memory_mb=20000 \
  --num_envs=8 --request_timeout=120 --stall_frequency=1 \
  --learning_rate=0.0001 --branch_penalty=0 --time_penalty=0.1 \
  --time_penalty_clip=180 --timeout_penalty=10 --search_budget=1000 \
  --search_budget_increase_factor=1.75 --expand_top_n=3 --first_step=search \
  --max_workers_per_lemma=4 --num_lemma_completes=1 --num_searches=5 \
  --tokenizer_cache_size=100000 --tokenizer_max_length=512 --tokenizer=roberta-base \
  --dim_transformer_feedforward=1024 --dropout=0.2 --n_transformer_head=8 \
  --pfm_dim=512 --pfm_layers=2 --transformer_layers=4 \
  --c_and=64 --heuristic_weight=0.3 --pb_c_base=3200 --pb_c_init=0.001 \
  --temperature=10 --value_discount=0.99 --value_penalty=8
```

Both commands use `--num_lemma_completes=1`, so training stops as soon as every lemma has been solved once. This matches the wall-clock times in Table 4, which measure the time to the first proof of each lemma. The proof sizes in Table 2 come from full runs that keep searching for shorter proofs after the first success, so the proofs produced here are expected to be larger.

The run finishes with a summary line reporting steps and updates performed.

Output is saved to `output/<run_id>/` by default, including:
- `proof_trees/` — proof trees (`.json` and `.spthy`) for each solved lemma
- `model_last.pt` — trained model weights (split into `model_last.pt` plus `model_last_emb*.pt` shards due to size constraints)
- `ckp_last.pt` — optimizer state and training metadata
- `raw.json` — per-lemma metrics in the same format as `results/best/raw.json`

To generate tables and plots for your own training runs, point the pipeline at `output` itself:

```bash
python scripts/result_collection.py --folder output --steps 4,6,8 -y
```

It picks up every `output/*/raw.json` and writes the results next to each one, in `output/<run_id>/{summary.yaml,hyperparameters.csv,tables/,plots/}`, plus a pooled version over all runs in `output/all/`. Run it from the repository root, since the Tamarin baselines it compares against are looked up at the fixed relative path `results/tamarin`. To restrict the comparison to a few runs, copy their `raw.json` files into a folder of your own (one subfolder per run) and pass that folder instead.

### 2.3 Verifying proofs

Each solved lemma produces a `.spthy` proof script in the `proof_trees/` folder. The proofs are verified immediately after solving. However, they can also be independently verified by running them through Tamarin. Since each proof file contains the full theory with `sorry` placeholders for other lemmas, use `--prove=<lemma_name>` to verify only the proven lemma:

```bash
tamarin-prover --prove=sqn_ue_unique \
  results/best/artifacts/9agfdtxf/proof_trees/5G_aka_fix_sqn_ue_unique_search_1_complete_1.spthy
```

**Reading Tamarin's output.** The relevant part is the `summary of summaries` block at the end. The proof is confirmed if the lemma you passed to `--prove` is reported as `verified`:

```
  sqn_ue_unique (all-traces): verified (6 steps)
```

> [!NOTE]
> Tamarin reports wellformedness errors for several models. All of them appear identically on the original, unmodified models and were not introduced by us. The models predate these checks, i.e. the version of Tamarin used to write the models did not yet implement the checks for these kinds of errors. We want to highlight that failed wellformedness checks do not necessarily imply a bug in the model or mistakes by the model's authors or affect the verification result.

## 3. Reproducing the full evaluation from scratch

Full reproduction of all experiments requires multiple GPU-days. Table 4 of the paper lists the wall-clock time of each RL run on our hardware; use it to pick what to re-run. As a rough guide:

| Wall-clock (paper hardware) | Protocols                                                               |
| --------------------------- | ----------------------------------------------------------------------- |
| under a minute              | Tutorial, UM PFS, NAXOS eCK PFS, FOO Eligibility, KAS2 eCK, YubiKey HSM |
| a few minutes               | YubiKey, NAXOS eCK, 5G Handover EPS N26, Wireguard                      |
| tens of minutes to hours    | PKCS11 AEAD, Signal, 5G Handover XN                                     |
| many hours                  | WPA2, SPDM, 5G AKA                                                      |

The classical models and the cheaper complex models are therefore reproducible on CPU-only hardware in a reasonable time frame; expect noticeably longer than the times above, since those were measured with a GPU and up to 64 cores. The YubikeyHSM example above is the recommended functional test.

### 3.1 All Tamarin baselines

The `Makefile` in `eval/` auto-discovers every `.spthy` file below it and runs each through `tamarin_wrapper.py` with 8 cores, `TIMEOUT` (default 7200 s) and `--resultfolder=../results/tamarin/<model_dir>`. Run its targets from `eval/`, since that output path is relative to the working directory.

```bash
cd eval
make list              # discovered models and how they split into parts
make all               # run every model
make all TIMEOUT=14400 # set a different timeout for this invocation
make part3             # one of part1..part8, disjoint slices for separate machines
make clean             # delete results/tamarin
```

The full set takes days, hence the `part` targets. All of them overwrite the CSVs shipped in `results/tamarin/`; to restore those, re-extract `results.zip` or point `--resultfolder` elsewhere.

The `c` and `s` baselines of SPDM and WPA2 were run with `TIMEOUT=14400` (four hours, Section 6 of the paper); the `Makefile` has no per-model case, so pass it explicitly for those four, e.g. `make TIMEOUT=14400 spdm/base_c/spdm121_composition_fix.spthy`. All other models, including the `original` variants of both, use the default.

### 3.2 All RL runs


The full training command for each protocol's best configuration is recorded in `hyperparameters.csv` (column `command`). Run each command from the repository root, and adjust `--num_envs` and `--cache_max_memory_mb` to your hardware.

| file                                 | runs                          |
| ------------------------------------ | ----------------------------- |
| `results/best/hyperparameters.csv`   | 16, the best run per protocol |
| `results/others/hyperparameters.csv` | 52                            |
| `results/all/hyperparameters.csv`    | 68, the complete set          |

Each row carries `protocol`, `run_id`, `completed_lemmas`, a `command` column holding a ready-to-run invocation, and one column per hyperparameter.

Three things the generated command deliberately leaves out:

- **The seed.** Archived runs used `seed=42`. For reproduction of the robustness experiments, change the seed to 43...46 on all `best` runs, except 5G_AKA, spdm, and wpa2.
- **Individual lemmas.** The command trains on all lemmas of the theory, which is how the runs were performed. Use `--lemma <name>` to isolate one.
- **Defaults.** Arguments equal to the built-in default are omitted, so the command stays readable. It is still complete: anything absent is the default.

## 4. Applying the RL framework to new protocols

When using our tool on a new Tamarin model, the following steps are required. We assume that the model is already written and syntactically correct, and that proof attempts with the standard Tamarin can be made.

The model itself needs no annotation: the framework reads the `.spthy` file as it is. For clean experiments, we recommend removing any `tactic:` blocks and `heuristic:` lines, however leaving them in does not break the framework. The RL framework makes use of them as prior, however we have not evaluated the effect of doing so. Depending on the tactics, this can help or hurt the search.

Run the following command to take the model as input to our tool. Start with the hyperparameters from the best run of a similar protocol, and adjust them as needed.

Once running, the print output of the command will give a short summary of the progress and state from time to time. Connecting with Weights and Biases (W&B) is also supported: pass `--use_wandb` while logged in (`wandb login`), and the run is logged to the project `tamarin-rl` of your default entity under its run ID (`--wandb_suffix` appends a label to the displayed name).

We expect the baseline to not perform completely poorly on new protocols, given that the shared baselines works well on the full set of protocols we have evaluated. If tuning the hyperparameters is necessary, we recommend to start with the base configuration and adjust the parameters one by one, or through a grid search.

### Resuming Training

Every run writes `model_last.pt` and `ckp_last.pt` into `output/<run_id>/`. To start a new run from one of those checkpoints:

```bash
python -m src.main.run train --load_from_run=<run_id> ...
```

This loads the model weights and the optimizer state, then creates a **new** run with a fresh ID, leaving the original checkpoint intact. Checkpoints are looked up under `--save_path_prefix` (default `output`), which is also where the new run is written — so loading a checkpoint stored elsewhere means pointing `--save_path_prefix` at its parent directory. `--load_from_checkpoint` (default `last`) selects which snapshot to load, i.e. `model_last.pt`.

To continue a run *in place* instead — same ID, same output folder, same W&B run — use `--resume_run=<run_id>`.

Note that the MCTS search is an integral part of the training process — a trained model alone will not solve any lemmas without the search. The search tree itself is not checkpointed (too large) so resuming a run does not resume the progress of the run, simply the model weights and optimizer state.


## Known limitations

- **Non-determinism**: Due to parallel MCTS workers and asynchronous training, results may vary between runs even with the same seed and hyperparameters. The pre-computed results represent specific runs. We find that our training is generally very robust to those sources of non-determinism (and to different hyperparameters), but exact reproduction of the same proof trees and metrics is not guaranteed.

- **CPU-only runs**: on a CPU the bottleneck is the model forward pass rather than Tamarin, and all workers share one inference process, so a run gets slower the more lemmas it searches in parallel. A single lemma is fine on a laptop; a full protocol needs a GPU to come close to the times in Table 4.


## License

The RL framework is licensed under the MIT License — see `LICENSE.md`.

The `eval/` directory contains cryptographic protocol models (`.spthy` files) based on the examples in the [Tamarin Prover](https://tamarin-prover.com/) repository. These files inherit the GNU General Public License v3.0 (GPLv3) — see `eval/LICENSE.md`. Our tool reads and processes these files as input data and is not a derivative work of them.

The other parts of the Docker image are derived from the Tamarin prover and are therefore GPLv3 as well:

| In the image                         | License                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| `/workspace/TamRL`                   | MIT (`LICENSE.md`)                                               |
| `/workspace/Tamarin-ML-API`          | GPLv3 (`LICENSE` in that directory)                              |
| `/usr/local/bin/tamarin-prover-json` | GPLv3, built from `/workspace/Tamarin-ML-API`                    |
| `/usr/local/bin/tamarin-prover`      | GPLv3, upstream Tamarin at the commit in `/workspace/SOURCES.md` |