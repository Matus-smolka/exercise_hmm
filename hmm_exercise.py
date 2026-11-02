import math
from collections.abc import Mapping
import json
from typing import Any, Dict

def log2(p):
    if p != 0  :
        return math.log2(p)
    if p == 0 :
        return math.inf

def transform_dict(d, func):

    if not isinstance(d, Mapping):
        raise TypeError("Input must be a dictionary or mapping type.")

    new_dict = {}
    for key, value in d.items():
        if isinstance(value, Mapping):
            # Recursively process nested dictionary
            new_dict[key] = transform_dict(value, func)
        else:
            try:
                new_dict[key] = func(value)
            except Exception as e:
                # Handle cases where func cannot be applied
                print(f"Warning: Could not transform value at key '{key}': {e}")
                new_dict[key] = value  # Keep original value if transformation fails
    return new_dict

class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.sates = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e =e
        self.log_pi = transform_dict(pi,log2)
        self.log_a = transform_dict(a,log2)
        self.log_e = transform_dict(e,log2)

    def joint_probability(self, x, z):
        # Convert to list if string
        if isinstance(x, str):
            x = list(x)
        if isinstance(z, str):
            z = list(z)

        # Check length match
        if len(x) != len(z):
            raise ValueError("Sequence and path must be the same length.")

        # --- First step ---
        first_state = z[0]
        first_symbol = x[0]

        try:
            prob_pi = self.pi[first_state]
        except KeyError:
            raise ValueError(f"State '{first_state}' not found in pi.")

        try:
            prob_e = self.e[first_state][first_symbol]
        except KeyError:
            raise ValueError(f"Symbol '{first_symbol}' not found in e for state '{first_state}'.")

        # Initial probability
        P = prob_pi * prob_e

        # --- Subsequent steps ---
        for i in range(1, len(x)):
            PS = z[i - 1]  # Previous state
            CS = z[i]  # Current state
            CE = x[i]  # Current emission

            try:
                A = self.a[PS][CS]
            except KeyError:
                raise ValueError(f"Transition from '{PS}' to '{CS}' not found in a.")

            try:
                E = self.e[CS][CE]
            except KeyError:
                raise ValueError(f"Emission '{CE}' not found in e for state '{CS}'.")

            # Multiply into cumulative probability
            P *= (A * E)

        return P

    def log_joint(self, x, z):

        # Convert to list if string
        if isinstance(x, str):
            x = list(x)
        if isinstance(z, str):
            z = list(z)

        # Check length match
        if len(x) != len(z):
            raise ValueError("Sequence and path must be the same length.")

        # --- First step ---
        first_state = z[0]
        first_symbol = x[0]

        try:
            log_pi_val = self.log_pi[first_state]
        except KeyError:
            raise ValueError(f"State '{first_state}' not found in log_pi.")

        try:
            log_e_val = self.log_e[first_state][first_symbol]
        except KeyError:
            raise ValueError(f"Symbol '{first_symbol}' not found in log_e for state '{first_state}'.")

        # Initial log probability
        P = log_pi_val + log_e_val

        # --- Subsequent steps ---
        for i in range(1, len(x)):
            PS = z[i - 1]  # Previous state
            CS = z[i]  # Current state
            CE = x[i]  # Current emission

            try:
                log_a_val = self.log_a[PS][CS]
            except KeyError:
                raise ValueError(f"Transition from '{PS}' to '{CS}' not found in log_a.")

            try:
                log_e_val = self.log_e[CS][CE]
            except KeyError:
                raise ValueError(f"Emission '{CE}' not found in log_e for state '{CS}'.")

            # Add log probabilities (multiplication in normal space)
            P += (log_a_val + log_e_val)

        return P

    class HMM:
        def __init__(self, states, alphabet, pi, a, e):
            self.states = states
            self.alphabet = alphabet
            self.pi = pi
            self.a = a
            self.e = e

            # Assuming transform_dict and log2 are defined elsewhere
            self.log_pi = transform_dict(pi, log2)
            self.log_a = transform_dict(a, log2)
            self.log_e = transform_dict(e, log2)

        def viterbi(self, x):
            """
            Viterbi algorithm in log2 space (correct addition version).

            Parameters:
            x : list or str
                Sequence of observed symbols.

            Returns:
            tuple : (best_path, best_log_prob)
            """
            # Convert to list if string
            if isinstance(x, str):
                x = list(x)

            n = len(x)
            if n == 0:
                raise ValueError("Sequence cannot be empty.")

            # v[i][state] = best log probability for state at position i
            v = [{} for _ in range(n)]
            # b[i][state] = best previous state for state at position i
            b = [{} for _ in range(n)]

            # --- Initialization ---
            first_emission = x[0]
            for S in self.states:
                try:
                    v[0][S] = self.log_pi[S] + self.log_e[S][first_emission]
                except KeyError:
                    raise ValueError(f"Missing emission '{first_emission}' for state '{S}' in log_e.")
                b[0][S] = None  # No previous state at first position

            # --- Recursion ---
            for i in range(1, n):
                current_emission = x[i]
                for CS in self.states:  # Current state
                    best_prob = None
                    best_prev_state = None
                    for PS in self.states:  # Previous state
                        try:
                            prob = v[i - 1][PS] + self.log_a[PS][CS] + self.log_e[CS][current_emission]
                        except KeyError:
                            raise ValueError(
                                f"Missing transition {PS}->{CS} or emission '{current_emission}' for state '{CS}'.")
                        if best_prob is None or prob > best_prob:
                            best_prob = prob
                            best_prev_state = PS
                    v[i][CS] = best_prob
                    b[i][CS] = best_prev_state

            # --- Termination ---
            best_final_state = max(v[n - 1], key=v[n - 1].get)
            best_log_prob = v[n - 1][best_final_state]

            # --- Backtrace ---
            best_path = [best_final_state]
            for i in range(n - 1, 0, -1):
                best_path.insert(0, b[i][best_path[0]])

            return best_path, best_log_prob

def read_model(path) -> "HMM":

    try:
        with open(path, 'r', encoding='utf-8') as file:
            data: Dict[str, Any] = json.load(file)

        # Validate required keys
        required_keys = ["states", "alphabet", "pi", "a", "e"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in JSON file.")

        # Create and return HMM instance
        return HMM(
            states=data["states"],
            alphabet=data["alphabet"],
            pi=data["pi"],
            a=data["a"],
            e=data["e"]
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading HMM from JSON: {e}")



