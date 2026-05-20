import concurrent.futures
import csv
import json
import os
import socket
import subprocess
import time
from typing import Any

import requests

from src.utils.load import load_spthy_file
from src.utils.utils import (
    CHOSEN_PROOF_METHOD,
    LEMMA_NAME,
    LEMMAS,
    NEW_THEORY_INDEX,
    NEXT_PROOF_PATH,
    PROOF_METHOD_NAME,
    PROOF_METHODS,
    PROOF_STATE,
    QUANTIFIER,
    SIDE,
    THEORY_INDEX,
    THEORY_KIND,
    TamarinCallTimeoutError,
    print_banner,
)

TIMEOUT = 600  # seconds (10 minutes)


def post_request(url: str, data: dict[str, Any], timeout=TIMEOUT) -> requests.Response:
    """
    Makes a POST request to the specified URL and returns the response.
    Retries on failure for a maximum of 10 seconds.
    """
    start_time = time.time()
    while True:
        try:
            return requests.post(url, data=data, timeout=timeout)
        except Exception as e:
            if time.time() - start_time > timeout:
                # traceback.print_exception(type(e), e, e.__traceback__)
                raise TamarinCallTimeoutError(
                    f"Failed to connect to server after {timeout} seconds"
                ) from e
        time.sleep(1)


def get_request(url: str, timeout=TIMEOUT):
    """
    Makes a GET request to the specified URL and returns the response.
    Retries on failure for a maximum of 10 seconds.
    """
    start_time = time.time()
    while True:
        try:
            return requests.get(url, timeout=timeout)
        except Exception as e:
            if time.time() - start_time > timeout:
                # traceback.print_exception(type(e), e, e.__traceback__)
                raise TamarinCallTimeoutError(
                    f"Failed to connect to server after {timeout} seconds"
                ) from e
        time.sleep(1)


class TamarinClient:
    # Ip -> Port -> Route
    server_overview_route = "http://{}:{}"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> Route
    theory_overview_route = "http://{}:{}/thy/{}/{}/overview/help"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> Option: Side -> LemmaName -> ProofPath -> Route
    proof_state_route = "http://{}:{}/thy/{}/{}/overview/proof/{}{}{}"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> Option: Side -> LemmaName -> ProofMethodIndex
    # ProofPath -> Route
    # The proof method index refers to the index of the chosen proof method in the list of
    # proof methods one can obtain by using the proof_state_route
    main_method_route = "http://{}:{}/thy/{}/{}/main/method/{}{}/{}{}"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> LemmaName -> Route
    initial_constraint_system_route = "http://{}:{}/thy/{}/{}/initialcs/{}"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> LemmaName -> Route
    apply_proofmethod_to_system_route = "http://{}:{}/thy/{}/{}/executemethod/{}"
    # Ip -> Port -> TheoryKind -> TheoryIdx -> LemmaName -> Route
    check_proof_route = "http://{}:{}/thy/{}/{}/checkproof/{}"

    @staticmethod
    def find_port() -> int:

        # Create a socket and bind to port 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
        sock.close()

        return port

    def __init__(
        self,
        theory_path: str,
        ip_addr="127.0.0.1",
        suppress_output=False,
        diff_arg: bool | None = None,
        theory_name: str | None = None,
        heuristic: str = "f",
        timeout: int = TIMEOUT,
    ):

        if theory_name is None or diff_arg is None:
            self.theory_name, _, diff_arg, _ = load_spthy_file(
                theory_path,
            )
        else:
            self.theory_name = theory_name

        self.ip_addr = ip_addr
        if theory_path is not None:
            for i in range(10):
                self.port = self.find_port()
                try:
                    args = [
                        "tamarin-prover-json",
                        "interactive",
                        "--port=" + str(self.port),
                    ]
                    if diff_arg:
                        args.append("--diff")
                    if heuristic != "f":
                        args.append(f"--heuristic={heuristic}")
                    args.append(theory_path)
                    self.process = subprocess.Popen(
                        args,
                        start_new_session=True,
                        stdout=subprocess.DEVNULL if suppress_output else None,
                        stderr=subprocess.DEVNULL if suppress_output else None,
                    )
                    self.server_overview = json.loads(
                        self.request_server_overview(timeout=timeout).text
                    )

                    break
                except Exception as e:
                    print(f"Error starting Tamarin Server: {e}, trying again at {i}/10")

            print(f"Started Tamarin Server at {theory_path}, port {self.port}")

    # @staticmethod
    # def port_is_used(port: int, host: str = "127.0.0.1") -> bool:
    #     """
    #     Returns if port is used. Port is considered used if the current process
    #     can't bind to it or the port doesn't refuse connections.
    #     """

    #     def _refuses_connection(port: int, host: str) -> bool:
    #         sock = socket.socket()
    #         with contextlib.closing(sock):
    #             sock.settimeout(1)
    #             err = sock.connect_ex((host, port))
    #             return err == errno.ECONNREFUSED

    #     def _can_bind(port: int, host: str) -> bool:
    #         sock = socket.socket()
    #         with contextlib.closing(sock):
    #             try:
    #                 sock.bind((host, port))
    #             except socket.error:
    #                 return False
    #         return True

    #     unused = _can_bind(port, host) and _refuses_connection(port, host)
    #     return not unused

    # @staticmethod
    # def get_free_port(port: int, port_range: int = 100) -> int:
    #     for p in range(port, port + port_range):
    #         if not TamarinClient.port_is_used(p):
    #             return p
    #     raise Exception(
    #         f"could not find free port in given range {port} - {port + port_range}"
    #     )

    def get_initial_constraint_system(
        self, theory_kind, theory_index, lemma, timeout=TIMEOUT
    ):
        """
        Gets the initial constraint system for a given lemma.

        Returns a _System_ JSON with the following schema:
        { prettyMethods: [String] } -- The proofmethods pretty printed for RL
        { encodedMethods: [String] } -- The proofmethods encoded in base64
        { encodedSys: String } -- The constraint system encoded in base64
        { status: ("Solved" | "Contradictory" | "Undetermined") }
        This can be used to compare constraint systems for equality.
        """
        return get_request(
            self.initial_constraint_system_route.format(
                self.ip_addr, self.port, theory_kind, theory_index, lemma
            ),
            timeout=timeout,
        )

    def check_proof(self, theory_kind, theory_index, lemma, proof, timeout=TIMEOUT):
        """
        Checks whether a given proof is correct for a given lemma.

        Returns a JSON with the following schema:
        { proofSize: Int } -- The size of the proof
        { proofFile: String } -- The Tamarin model file containing the proof
        { proofStatus: "CompleteProof" | "TraceFound" | "UnfinishableProof" | "InvalidatedProof" } -- The status of the proof
        "CompleteProof" indicates that no counterexample could be found and the proof is complete.
        "TraceFound" indicates that a counterexample was found.
        "UnfinishableProof" indicates that the proof could not be finished for internal reasons of Tamarin. Very rare.
        "InvalidatedProof" indicates that the proof was invalidated.

        """
        route = self.check_proof_route.format(
            self.ip_addr, self.port, theory_kind, theory_index, lemma
        )
        return post_request(route, data={"proof": proof}, timeout=timeout)

    def apply_proofmethod_to_system(
        self,
        theory_kind,
        theory_index,
        lemma,
        proof_method,
        system,
        timeout=TIMEOUT,
    ) -> requests.Response:
        """
        Applies a proofmethod to a given constraint system. Can fail returning
        an error message in the HTTP response.

        Returns a list of JSONs with the following schema on success:
        [ { caseName: String, system: System } ]
        See above for the schema of System.

        The array contains all resulting cases after applying the proof
        method to the given system. Each case has a name and the resulting system.

        The proof_method and system parameters are expected to be the base64
        encoded versions of the proof method and system, respectively, as
        returned by the get_initial_constraint_system function.
        """
        route = self.apply_proofmethod_to_system_route.format(
            self.ip_addr, self.port, theory_kind, theory_index, lemma
        )
        return post_request(
            route,
            data={"proofmethod": proof_method, "system": system},
            timeout=timeout,
        )

    def apply_proofmethod_to_system_bulk(
        self,
        theory_kind,
        theory_index,
        lemma,
        proof_methods,
        system,
        timeout=TIMEOUT,
    ) -> list[requests.Response]:
        """
        Applies a proofmethod to a given constraint system. Can fail returning
        an error message in the HTTP response.

        Returns a list of JSONs with the following schema on success:
        [ { caseName: String, system: System } ]
        See above for the schema of System.

        The array contains all resulting cases after applying the proof
        methods to the given system. Each case has a name and the resulting system.

        The proof_methods and system parameters are expected to be the base64
        encoded versions of the proof methods and system, respectively, as
        returned by the get_initial_constraint_system function.
        """
        route = self.apply_proofmethod_to_system_route.format(
            self.ip_addr, self.port, theory_kind, theory_index, lemma
        )

        if not proof_methods:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            responses = list(
                executor.map(
                    lambda pm: post_request(
                        route,
                        data={"proofmethod": pm, "system": system},
                        timeout=timeout,
                    ),
                    proof_methods,
                )
            )
        return responses

    def request_server_overview(self, timeout=TIMEOUT):
        """
        Gets an overview of the loaded protocol models, called theories, of the
        interactive Tamarin server.

        Returns a JSON with the following schema:
        { String: [ { 'theoryIndex': Int, 'theoryKind': ('trace' | 'diff'), 'theoryName': String } ] }

        where the strings used as index are the names of the loaded theories.
        NOTE: that this is not the name of the `.spthy` file. It is the name of the
        model inside of the file!

        Each entry in the JSON array corresponds to a protocol model. The 'theoryIndex'
        is used for index access to the model, the 'theoryKind' defines whether
        the model deals with trace or observational equivalence properties, and
        the 'theoryName' is the user-chosen name of the theory.
        """
        return get_request(
            self.server_overview_route.format(self.ip_addr, str(self.port)),
            timeout=timeout,
        )

    def request_theory_overview(self, theory_kind, theory_index, timeout=TIMEOUT):
        """
        Gets an overview of the theory status.

        Returns a JSON with the following schema:
        { 'theoryRaw': String -- The raw string of the theory
        , 'lemmas': [ Lemma ] -- The list of lemmas
        }

        where

        Lemma = { 'name': String                                -- The name of the lemma
                , 'proofState': Proof                           -- The (partial) proof of the lemma (see below)
                , 'quantifier': ('AllTraces' | 'ExistsTrace') } -- The quantifier of the lemma
                , 'side': ('LHS' | 'RHS') -- Only for diff theories

        """
        # Renaming the diff returned by serialization on the server side
        # to the actual path the server expects. Don't know why
        # they went with 'equiv' instead of 'diff'.
        if theory_kind == "diff":
            theory_kind = "equiv"
        return get_request(
            self.theory_overview_route.format(
                self.ip_addr, str(self.port), theory_kind, theory_index
            ),
            timeout=timeout,
        )

    def request_proof_state(
        self,
        theory_kind,
        theory_index,
        lemma,
        proof_path=None,
        side=None,
        timeout=TIMEOUT,
    ):
        """
        Gets a view of the proof state at a given proof path.

        Returns a `Proof` JSON.

        Proof = { 'cases': [ String ] -- The names of the cases at this node in the proof tree
                , 'chosenProofMethod': ProofMethod
                , 'proofStatus': (IncompleteProof | CompleteProof
                                 | TraceFound | InvalidatedProof
                                 | UnfinishableProof | UndeterminedProof)
                , 'constraintSystem': String
                , 'proofMethods': [ ProofMethod ]
                }

        """
        if theory_kind == "diff":
            theory_kind = "equiv"
        if proof_path is None:
            proof_path = []
        proof_path = ["_" if p == "" else p for p in proof_path]
        proof_path = "/".join(proof_path)
        # If path is NOT empty, we need to prepend a '/'.
        # Example route: ``http://127.0.0.1:3001/thy/trace/25/overview/proof/ExtractData_Executable/_/Create_Initiator''
        if len(proof_path) > 0:
            proof_path = "/" + proof_path

        # If we have a side, we add it to the route.
        # Example route: ``http://127.0.0.1:3001/thy/equiv/1/overview/proof/LHS/rand_autn_src''
        side_string = "" if not side else side + "/"

        # If the path is empty, we MUST NOT have a trailing '/'; otherwise the
        # route is not correct.
        # Example route: ``http://127.0.0.1:3001/thy/trace/25/overview/proof/ExtractData_Executable''
        route = self.proof_state_route.format(
            self.ip_addr,
            str(self.port),
            theory_kind,
            theory_index,
            side_string,
            lemma,
            proof_path,
        )
        return get_request(route, timeout=timeout)

    def apply_method_at_path(
        self,
        theory_kind,
        theory_index,
        lemma,
        proof_method_index,
        proof_path=None,
        side=None,
        timeout=TIMEOUT,
    ):
        """
        Applies a proof method at a given lemma and proof path. Can fail.

        If successful, it returns a JSON with the following schema:
        { 'newTheoryIndex': Int, 'nextProofpath': [ String ] }

        Here, the 'newTheoryIndex' is the new index which should be used to index
        the modified theory. Tamarin internally constructs a new theory when an
        existing theory is modified.

        The 'nextProofpath' is the path which Tamarin would have choosen if the
        path was clicked in the GUI. E.g., if multiple case distinction arise from
        the chosen proof method, it chooses the first one.

        TODO: Think about also returning the new cases here such that one
        can choose a different case.
        """
        if theory_kind == "diff":
            theory_kind = "equiv"
        if proof_path is None:
            proof_path = []
        side_string = "" if not side else side + "/"
        proof_path = ["_" if p == "" else p for p in proof_path]
        proof_path = "/".join(proof_path)
        # An empty proof path means we want to apply the chosen proof method
        # at the root of the proof tree.
        # Therefore, we completely omit the proof path from the route
        # Example route: "http://127.0.0.1:3001/thy/trace/1/main/method/ExtractData_Executable/1"
        if len(proof_path) > 0:
            # If the proof path is not empty, we apply the method at the path.
            # Therefore, we need to prepend a '/'.
            # Tamarin uses '_' to indicate that there is only single case
            # distinction at this node in the proof tree. We expect this to be
            # in the proof path as a string. E.g., path = ["_", "Create_Initiator"]
            # for the following example route.
            # Example route: "http://127.0.0.1:3001/thy/trace/1/main/method/ExtractData_Executable/1/_/Create_Initiator"
            proof_path = "/" + proof_path

        route = self.main_method_route.format(
            self.ip_addr,
            str(self.port),
            theory_kind,
            theory_index,
            side_string,
            lemma,
            proof_method_index,
            proof_path,
        )
        return get_request(route, timeout=timeout)

    def request_proofmethods_for_path(
        self,
        theory_kind,
        theory_index,
        lemma,
        proof_path=None,
        side=None,
        timeout=TIMEOUT,
    ):
        if proof_path is None:
            proof_path = []
        proof_methods = []
        for i in range(len(proof_path) + 1):
            r = self.request_proof_state(
                theory_kind, theory_index, lemma, proof_path[:i], side, timeout=timeout
            )
            r = json.loads(r.text)
            methods = r[PROOF_METHODS]
            idx = methods.index(r[CHOSEN_PROOF_METHOD])
            proof_methods.append(idx + 1)
        return proof_methods

    def overview(self, timeout=TIMEOUT):
        print_banner("Overview")

        # Get an overview of the server. This is the initial list of models
        # you will see when opening the GUI of the interactive mode.
        # See the documentation of the client for the exact format of the JSON.
        r = self.request_server_overview(timeout=timeout)
        r = json.loads(r.text)
        # print("==========================================================================")
        # print("# The server overview")
        # print(r)

        # Choose a theory from the response
        theory = r[self.theory_name]
        # The index and kind of theory are needed by Tamarin to index it internally
        theory_kind = theory[THEORY_KIND]
        theory_index = theory[THEORY_INDEX]
        # Get an overview of the theory.
        r = self.request_theory_overview(theory_kind, theory_index, timeout=timeout)
        r = json.loads(r.text)
        # The theory overview contains all the lemmas and their proof states.
        # This JSON gets huge! Thus, viewing it in its whole does not make much sense.
        # See the documentation of of the client for the exact JSON format.
        # print("==========================================================================")
        # print("# The theory overview")
        # print(r)

        # Inspect the lemmas
        for lemma in r[LEMMAS]:
            # Inspect name and the quantifier of the lemmas, i.e., all-traces or exists-trace
            print(
                "Lemma: " + lemma[LEMMA_NAME] + (" - " + lemma[SIDE])
                if SIDE in lemma and lemma[SIDE]
                else "" + " --- " + lemma[QUANTIFIER]
            )

            # Inspect the proof state of the lemmas
            # This JSON contains all proof paths and, at for each path, the constraint
            # system, the ranked proof methods, and the chosen proof method etc.
            # For more details see the documentation of the client.
            # As a result, this JSON also gets huge and should not be viewed in its whole.
            proof_state = lemma[PROOF_STATE]
            proof_method_name = proof_state[CHOSEN_PROOF_METHOD][PROOF_METHOD_NAME]
            print("\tChosen proof method: " + proof_method_name)
            print("\tRanked proof methods: " + str(proof_state[PROOF_METHODS]))

    def proof_state(self, lemma_name: str, timeout=TIMEOUT):
        # We will now take a look at how we can use the client to apply proof
        # methods and how the proof state JSON looks like.
        print_banner("Starting Proofstate Example")

        r = self.request_server_overview(timeout=timeout)
        r = json.loads(r.text)
        # Choose a theory from the response
        theory = r[self.theory_name]
        # The index and kind of theory are needed by Tamarin to index it internally
        theory_kind = theory[THEORY_KIND]
        theory_index = theory[THEORY_INDEX]

        # We can get the proof state of a specific lemma at the root of the proof tree
        valid_proof_state = self.request_proof_state(
            theory_kind, theory_index, lemma_name, timeout=timeout
        )
        # We can also get the proof state at a specific path of the proof
        invalid_proof_state = self.request_proof_state(
            theory_kind,
            theory_index,
            lemma_name,
            proof_path=["Non", "existing", "path"],
            timeout=timeout,
        )
        # Since this path does not exist, the response will have error code 400
        # The easiest way to check for this is to use the `.ok` field of the response
        # object
        # print(
        #     "Valid proof path: " + str(valid_proof_state.ok),
        #     "---",
        #     "Invalid proof path: " + str(invalid_proof_state.ok),
        # )

        # A proof path is just a list of strings. In Tamarin, each string corresponds
        # to the name of a case in a case distinction. If there is only a single
        # case, the "_" is used to `select' it.

        # Let's prove something now!
        # `proof_path` will be the current path we are focusing on.
        proof_path = []

        # Get the proof state
        proof_state = json.loads(
            self.request_proof_state(
                theory_kind, theory_index, lemma_name, proof_path
            ).text
        )
        proof_methods = proof_state[PROOF_METHODS]
        # Inspect the ranked methods
        # print(proof_methods)

        # Let's take a look at how we can apply a proof method!

        # Not specifying `proof_path` for this function defaults to `[]`.
        # Thus, we could have ommitted it in this case since `proof_path` = [] currently.
        r = self.apply_method_at_path(
            theory_kind, theory_index, lemma_name, 1, proof_path
        )
        # `apply_method_at_path` applies the method at index `1` at proof path [] of
        # the lemma. The index refers to the method at the corresponding proof
        # methods list. I.e., we choose `proof_methods[1]`

        # This method can fail if the path is invalid or the lemma does not exist
        if r.ok:
            r = json.loads(r.text)
        else:
            raise ValueError("Invalid proof path: " + str(proof_path))

        # The response JSON for applying a proof method looks like this:
        # { 'newTheoryIndex': Int, 'nextProofpath': Optional [ String ] }

        # If not 'None', the 'nextProofpath' list is the path to the next case in
        # the proof Tamarin wants to focus on. Currently, Tamarin simply iterates
        # through case distinctions in order.
        # If 'None', this indicates that the proof for this lemma is finished.

        # The 'newTheoryIndex' is the index to the new theory that contains the
        # new, modified proof state. Internally, Tamarin creates a new theory
        # everytime an existing theory is modified. Thus, we have to deal with this.

        theory_index = r[NEW_THEORY_INDEX]
        proof_path = r[NEXT_PROOF_PATH]

        # Now we can inspect the new proof state and continue

        r = self.request_proof_state(theory_kind, theory_index, lemma_name, proof_path)

        # Proof paths returned by Tamarin are always valid paths
        # If not, that's a bug in the server or client code
        r = json.loads(r.text)

        proof_methods = r[PROOF_METHODS]
        print(proof_methods)

        # Continue the proof from here...

    # def prove_lemma_dfs(
    #     self,
    #     lemma_name: str,
    #     model: EndToEndMLModel,
    #     timeout_minutes=5,
    #     cli_output=False,
    #     log_prefix: str | None = None,
    #     wandb=None,
    #     side: str | None = None,
    #     save_tokens=False,
    # ):
    #     if not cli_output:
    #         print("Starting Proving")
    #     else:
    #         print_banner("Starting Proving")

    #     self.write_log_header(log_prefix)
    #     # We will now take a look at how we can use the client to prove a lemma

    #     r = self.request_server_overview()
    #     r = json.loads(r.text)
    #     theory = r[self.theory_name]
    #     theory_kind = theory[THEORY_KIND]
    #     current_theory_index = theory[THEORY_INDEX]
    #     r = json.loads(
    #         self.request_theory_overview(theory_kind, current_theory_index).text
    #     )

    #     # Setup variables needed for proving
    #     current_proof_path: list[str] = []

    #     steps = 0
    #     pfm_index = []

    #     # Apply proof methods until proof is finished or timeout is reached
    #     timeout_seconds = timeout_minutes * 60
    #     start_time = time.time()
    #     while True:
    #         steps += 1
    #         if time.time() - start_time > timeout_seconds:
    #             print(
    #                 f"Timeout of {timeout_minutes} minutes reached. Stopping proof loop after {steps} steps."
    #             )
    #             summary = {
    #                 "steps_num": steps,
    #                 "time_minutes": (time.time() - start_time) / 60,
    #                 "time_seconds": (time.time() - start_time),
    #                 "proof_status": "Timeout",
    #                 "trace_indices": [],
    #             }
    #             self.write_summary(log_prefix, wandb, summary)
    #             return
    #         # while True:
    #         # Warning: You should always give a proof path when querying for the
    #         # proof state in a hot loop. The reason is that the resulting
    #         # JSON grows exponentially in size in the number of proof steps
    #         # because it contains the whole proof tree.
    #         start_time_step = time.time()
    #         r = self.request_proof_state(
    #             theory_kind,
    #             current_theory_index,
    #             lemma_name,
    #             current_proof_path,
    #             side=side,
    #         )
    #         proof_state = json.loads(r.text)

    #         start_time_model = time.time()
    #         method_index, tokens = model(proof_state)
    #         end_time_model = time.time()
    #         pfm_index.append(method_index)

    #         method_name = proof_state[PROOF_METHODS][method_index][PROOF_METHOD_NAME]
    #         if cli_output:
    #             print(f"\tChosen {method_name}, index {method_index}, steps {steps}.")
    #         # +1 to make the index 1-based
    #         r = self.apply_method_at_path(
    #             theory_kind,
    #             current_theory_index,
    #             lemma_name,
    #             method_index + 1,
    #             current_proof_path,
    #             side=side,
    #         )

    #         r = json.loads(r.text)

    #         # Get the new proof state using the new index
    #         current_theory_index = r[NEW_THEORY_INDEX]

    #         end_time_step = time.time()

    #         self.log_step(
    #             log_prefix,
    #             step_info={
    #                 "steps": steps,
    #                 "method_index": method_index,
    #                 "method_name": method_name,
    #                 "time": f"{end_time_step - start_time_step:.2f}",
    #                 "model_time": f"{end_time_model - start_time_model:.2f}",
    #                 "proof_path": str(current_proof_path),
    #                 "proof_status": proof_state[PROOF_STATUS],
    #                 "methods": str(proof_state[PROOF_METHODS]),
    #                 "tokens": str(tokens) if save_tokens else "",
    #             },
    #         )

    #         if r[NEXT_PROOF_PATH] is None:
    #             break

    #         # We have a proof path to continue the proof from
    #         current_proof_path = r[NEXT_PROOF_PATH]

    #     # Get the final proof state
    #     r = self.request_proof_state(
    #         theory_kind, current_theory_index, lemma_name, side=side
    #     )
    #     proof_state = json.loads(r.text)
    #     print(proof_state[PROOF_STATUS])
    #     end_time = time.time()
    #     print(
    #         f"Proving finished after {steps} steps, {(end_time - start_time) / 60:.2f} minutes, with proof path length {len(current_proof_path)}."
    #     )
    #     print("Getting summary...")
    #     if log_prefix is not None:
    #         with open(log_prefix + "log.csv", "a", encoding="utf-8", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(
    #                 [
    #                     steps + 1,
    #                     "",
    #                     "",
    #                     str(current_proof_path),
    #                     proof_state[PROOF_STATUS],
    #                     "",
    #                 ]
    #             )
    #     if proof_state[PROOF_STATUS] == TRACE_FOUND:
    #         path_to_solution = self.request_proofmethods_for_path(
    #             theory_kind,
    #             current_theory_index,
    #             lemma_name,
    #             current_proof_path,
    #             side=side,
    #         )
    #         if cli_output:
    #             print([i - 1 for i in path_to_solution])
    #     else:
    #         path_to_solution = []
    #     summary = {
    #         "steps_num": steps,
    #         "time_minutes": (end_time - start_time) / 60,
    #         "time_seconds": (end_time - start_time),
    #         "proof_status": proof_state[PROOF_STATUS],
    #         "trace_indices": [i - 1 for i in path_to_solution],
    #     }
    #     self.write_summary(log_prefix, wandb, summary)

    # def prove_lemma_iddfs(
    #     self,
    #     lemma_name: str,
    #     model: EndToEndMLModel,
    #     timeout_minutes=5,
    #     cli_output=False,
    #     log_prefix: str | None = None,
    #     wandb=None,
    #     side=None,
    #     save_tokens=False,
    # ):
    #     if not cli_output:
    #         print("Starting Proving Example")
    #     else:
    #         print_banner("Starting Proving Example")

    #     self.write_log_header(log_prefix)

    #     r = self.request_server_overview()
    #     r = json.loads(r.text)
    #     theory = r[self.theory_name]
    #     theory_kind = theory[THEORY_KIND]
    #     initial_theory_index = theory[THEORY_INDEX]
    #     r = json.loads(
    #         self.request_theory_overview(theory_kind, initial_theory_index).text
    #     )

    #     initial_proof_path: list[str] = []

    #     result, method_indices, steps, time = self._prove_lemma_iddfs(
    #         theory_kind,
    #         initial_theory_index,
    #         lemma_name,
    #         model,
    #         initial_proof_path=initial_proof_path,
    #         side=side,
    #         timeout_minutes=timeout_minutes,
    #         cli_output=cli_output,
    #         log_prefix=log_prefix,
    #         save_tokens=save_tokens,
    #     )

    #     if result == "Timeout":
    #         summary = {
    #             "steps_num": steps,
    #             "time_minutes": timeout_minutes,
    #             "time_seconds": timeout_minutes * 60,
    #             "proof_status": "Timeout",
    #             "trace_indices": [],
    #         }
    #         print(
    #             f"Timeout of {timeout_minutes} minutes reached. Stopping proof loop after {result[2]} steps."
    #         )
    #     elif result == TRACE_FOUND:
    #         assert method_indices is not None

    #         if cli_output:
    #             print(f"Found a solution after {steps} steps.")
    #             print(
    #                 f"Path to solution (method indices): {[i - 1 for i in method_indices]}"
    #             )
    #         summary = {
    #             "steps_num": steps,
    #             "time_minutes": time / 60,
    #             "time_seconds": time,
    #             "proof_status": TRACE_FOUND,
    #             "trace_indices": [i - 1 for i in method_indices],
    #         }
    #     elif result == COMPLETE_PROOF:
    #         print(f"Completed proof after {steps} steps.")
    #         summary = {
    #             "steps_num": steps,
    #             "time_minutes": time / 60,
    #             "time_seconds": time,
    #             "proof_status": COMPLETE_PROOF,
    #             "trace_indices": [],
    #         }
    #     else:
    #         raise ValueError("Unknown result: " + str(result))
    #     self.write_summary(log_prefix, wandb, summary)

    # def _prove_lemma_iddfs(
    #     self,
    #     theory_kind,
    #     initial_theory_index,
    #     lemma_name: str,
    #     model: EndToEndMLModel,
    #     initial_proof_path=None,
    #     side=None,
    #     timeout_minutes=5,
    #     cli_output=False,
    #     log_prefix: str | None = None,
    #     save_tokens=False,
    # ) -> tuple[
    #     Literal["TraceFound", "CompleteProof", "Timeout"],
    #     list[int] | None,
    #     int,
    #     float,
    # ]:
    #     """
    #     Proves a lemma using iterative deepening search.
    #     - theory_kind: 'trace' | 'diff'
    #     - theory_index: Int
    #     - lemma: String
    #     - initial_proof_path: [ String ]
    #     - side: 'LHS' | 'RHS' | None -- Only for diff
    #     - timeout_minutes: Timeout in minutes (default: 5)

    #     Returns a tuple (String, [ (Int, String) ]) where the string is one of
    #     - "NoSolution" -- Explored all paths, no solution, i.e., proved a universal lemma
    #     - "Solution"   -- Found a solution at the path
    #     and the list is the path to the solution with the index of the chosen proof method
    #     at each step.
    #     """

    #     if initial_proof_path is None:
    #         initial_proof_path = []

    #     timeout_seconds = timeout_minutes * 60
    #     start_time = time.time()

    #     def iterative_deepening_search(max_depth=4):
    #         print("Searching with depth " + str(max_depth))
    #         # Poor person's enum:
    #         # ("NoSolution", None, steps) -> Explored all paths, no solution, i.e., proved a universal lemma
    #         # ("Bound", None, steps) -> Hit depth bound, need to go deeper
    #         # ("Timeout", None, steps) -> Hit timeout bound, stop search
    #         # ("Solution", (proof_path, theory_index), steps) -> Found a solution at path for the theory with the index

    #         def process_proof_state(
    #             current_depth, current_theory_index, current_proof_path
    #         ):
    #             # Check for timeout
    #             if time.time() - start_time > timeout_seconds:
    #                 return ("Timeout", None, 1)
    #             if current_depth > max_depth:
    #                 return ("Bound", None, 1)

    #             start_time_step = time.time()
    #             # Get the proof state at the current path
    #             r = self.request_proof_state(
    #                 theory_kind,
    #                 current_theory_index,
    #                 lemma_name,
    #                 current_proof_path,
    #                 side=side,
    #             )
    #             proof_state = json.loads(r.text)
    #             proof_status = proof_state[PROOF_STATUS]

    #             if proof_status == TRACE_FOUND:
    #                 return (
    #                     TRACE_FOUND,
    #                     (current_proof_path, current_theory_index),
    #                     1,
    #                 )
    #             elif proof_status == COMPLETE_PROOF:
    #                 return (COMPLETE_PROOF, None, 1)
    #             elif proof_status in [INCOMPLETE_PROOF, UNFINISHABLE_PROOF]:

    #                 start_time_model = time.time()
    #                 method_index, tokens = model(proof_state)
    #                 end_time_model = time.time()

    #                 method_name = proof_state[PROOF_METHODS][method_index][
    #                     PROOF_METHOD_NAME
    #                 ]
    #                 if cli_output:
    #                     print(f"\tChosen {method_name}, index {method_index}.")
    #                 # +1 because Tamarin expects 1-based indices
    #                 r = self.apply_method_at_path(
    #                     theory_kind,
    #                     current_theory_index,
    #                     lemma_name,
    #                     method_index + 1,
    #                     current_proof_path,
    #                     side=side,
    #                 )
    #                 r = json.loads(r.text)

    #                 # Get the new proof state using the new index
    #                 current_theory_index = r[NEW_THEORY_INDEX]

    #                 end_time_step = time.time()

    #                 self.log_step(
    #                     log_prefix,
    #                     step_info={
    #                         "method_index": method_index,
    #                         "method_name": method_name,
    #                         "time": f"{end_time_step - start_time_step:.2f}",
    #                         "model_time": f"{end_time_model - start_time_model:.2f}",
    #                         "proof_path": str(current_proof_path),
    #                         "proof_status": proof_state[PROOF_STATUS],
    #                         "methods": str(proof_state[PROOF_METHODS]),
    #                         "tokens": str(tokens) if save_tokens else "",
    #                     },
    #                 )

    #                 r = self.request_proof_state(
    #                     theory_kind,
    #                     current_theory_index,
    #                     lemma_name,
    #                     current_proof_path,
    #                     side=side,
    #                 )
    #                 r = json.loads(r.text)

    #                 cases = r[CASES]
    #                 # Replace empty case names with '_' for routing. Tamarin quirk.
    #                 for i, _ in enumerate(cases):
    #                     if cases[i] == "":
    #                         cases[i] = "_"

    #                 sub_results = []
    #                 # Counting this node
    #                 new_size = 1
    #                 for case in cases:
    #                     (res, path, size) = process_proof_state(
    #                         current_depth + 1,
    #                         current_theory_index,
    #                         current_proof_path + [case],
    #                     )
    #                     new_size += size
    #                     # print("New size: " + str(new_size))

    #                     # Check if there is a solution or timeout. Return path.
    #                     if res in (TRACE_FOUND, "Timeout"):
    #                         return (res, path, new_size)

    #                     sub_results.append(res)

    #                 # We didn't return early due to a solution or timeout.
    #                 # Check if bound was hit or if no solution is possible.
    #                 if "Bound" in sub_results:
    #                     return ("Bound", None, new_size)
    #                 elif "Timeout" in sub_results:
    #                     return ("Timeout", None, new_size)
    #                 else:
    #                     return (COMPLETE_PROOF, None, new_size)
    #             else:
    #                 raise ValueError("Unsupported proof status: " + str(proof_status))

    #         # Start search at depth 0 with initial parameters and 1 steps (the root)
    #         (res, path, steps) = process_proof_state(
    #             0,
    #             initial_theory_index,
    #             initial_proof_path,
    #         )
    #         if res == "Bound":
    #             print("Hit bound after exploring " + str(steps) + " nodes")
    #             return iterative_deepening_search(max_depth * 2)
    #         elif res == "Timeout":
    #             print(
    #                 f"Timeout of {timeout_minutes} minutes reached. Stopping search after {steps} nodes."
    #             )
    #             return (res, path, steps)
    #         else:
    #             return (res, path, steps)

    #     # Proving is now just searching via IDDFS
    #     (result, maybe_path, steps) = iterative_deepening_search()
    #     end_time = time.time()

    #     if result == TRACE_FOUND:
    #         # If a solution was found, annotate the path with the index of
    #         # the proof method used
    #         (path, solution_theory_index) = maybe_path  # type: ignore
    #         method_indices = self.request_proofmethods_for_path(
    #             theory_kind, solution_theory_index, lemma_name, path, side=side
    #         )
    #         return (result, method_indices, steps, end_time - start_time)
    #     else:
    #         return (result, None, steps, end_time - start_time)

    def stop_process(self):
        """
        Kills the Tamarin server process.
        """
        if hasattr(self, "process") and self.process.poll() is None:
            pid = self.process.pid
            self.process.terminate()
            self.process.wait()
            # Ensure process really stopped; if still alive try to force-kill
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                except OSError:
                    print(
                        f"Client: Tamarin server {pid} stopped successfully at port {self.port}."
                    )
                    break
                else:
                    try:
                        os.kill(pid, 9)
                        print(
                            f"Client: Force-killed process {pid} at port {self.port}."
                        )
                    except Exception as e:
                        print(
                            f"Client: Failed to force-kill process {pid} with error: {e}"
                        )
                time.sleep(1)

    def __del__(self):
        self.stop_process()

    @staticmethod
    def log_step(log_prefix: str | None, step_info: dict):
        if log_prefix is not None:
            with open(log_prefix + "log.csv", "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step_info.get("steps", ""),
                        step_info.get("method_index", ""),
                        step_info.get("method_name", ""),
                        step_info.get("time", ""),
                        step_info.get("model_time", ""),
                        step_info.get("proof_path", ""),
                        step_info.get("proof_status", ""),
                        step_info.get("methods", ""),
                        step_info.get("tokens", ""),
                    ]
                )

    @staticmethod
    def write_summary(log_prefix: str | None, wandb, summary: dict):
        if log_prefix is not None:
            with open(
                log_prefix + "summary.json", "w", encoding="utf-8", newline=""
            ) as f:
                f.write(json.dumps(summary, indent=4))
        if wandb is not None and log_prefix is not None:
            wandb.log({f"{'_'.join(log_prefix.split('/')[-2:])}summary": summary})
        print("Summary written.")

    def write_log_header(self, log_prefix: str | None):
        if log_prefix is not None:
            with open(log_prefix + "log.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "method_index",
                        "method_name",
                        "time",
                        "model_time",
                        "proof_path",
                        "proof_status",
                        "methods",
                        "tokens",
                    ]
                )

    def __enter__(self):
        """
        Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and ensure cleanup.
        """
        self.stop_process()
