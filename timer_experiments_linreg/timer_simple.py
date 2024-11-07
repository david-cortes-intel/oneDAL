import numpy as np
import sys
from sklearn.datasets import make_regression
import time
from sklearnex.linear_model import LinearRegression
import re

def gen_random_uniform(
    m: int, n: int, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    X = rng.random(size=(m,n))
    y = rng.random(size=m)
    return X, y

def gen_random_normal_wide(
    m: int, n: int, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    sd_X = rng.standard_gamma(10)
    sd_y = rng.standard_gamma(5)
    mean_X = rng.standard_normal()
    mean_y = rng.standard_normal()
    X = rng.standard_normal(size=(m,n)) * sd_X + mean_X
    y = rng.standard_normal(size=m) * sd_y + mean_y
    return X, y

def gen_random_nonpsd(
    m: int, n: int, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    X = rng.random(size=(m,n))
    X[:,1] = X[:,0]
    y = rng.random(size=m)
    return X, y

def gen_random_with_patterns(
    m: int, n: int, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=m,
        n_features=n,
        n_informative=min(1, int(0.75*n)),
        random_state=seed,
    )

def print_time(name, time_start, time_end):
    print(f"time {name}:", (time_end - time_start) / 1e6, flush=True)

def run_timer(
    m: int,
    n: int,
    batch_size: int,
    nthreads: int,
    method: str,
    repeated_calls_full: int = 1,
    repeated_calls_fit: int = 1,
    same_seed: bool = True,
    seed: int = 123
):
    batch_size = min(batch_size, m)
    LinearRegression().get_hyperparameters("fit").cpu_macro_block = batch_size

    for repetition in range(repeated_calls_full):
        seed = seed + (0 if same_seed else repetition)
        if method == "uniform":
            st_gen = time.time_ns()
            X, y = gen_random_uniform(m, n, seed)
            end_gen = time.time_ns()
        elif method == "normal":
            st_gen = time.time_ns()
            X, y = gen_random_normal_wide(m, n, seed)
            end_gen = time.time_ns()
        elif method == "make_regression":
            st_gen = time.time_ns()
            X, y = gen_random_with_patterns(m, n, seed)
            end_gen = time.time_ns()
        elif method == "nonpsd":
            st_gen = time.time_ns()
            X, y = gen_random_nonpsd(m, n, seed)
            end_gen = time.time_ns()
        else:
            raise ValueError()
        print_time("gen", st_gen, end_gen)

        for repetition in range(repeated_calls_fit):
            st_fit = time.time_ns()
            model = LinearRegression(n_jobs=nthreads).fit(X, y)
            end_fit = time.time_ns()
            print_time("fit", st_fit, end_fit)

def parse_args():
    def extract_arg(name: str) -> str:
        return [
            arg.split("=")[1]
            for arg in sys.argv
            if arg.startswith(f"--{name}=")
            or arg.startswith(f"-{name}=")
            or arg.startswith(f"{name}=")
        ][0]
    
    return {
        "m": int(extract_arg("m")),
        "n": int(extract_arg("n")),
        "batch_size": int(extract_arg("batch_size")),
        "nthreads": int(extract_arg("nthreads")),
        "repeated_calls_full": int(extract_arg("repeated_calls_full")),
        "repeated_calls_fit": int(extract_arg("repeated_calls_fit")),
        "same_seed" : bool(extract_arg("same_seed")),
        "method" : extract_arg("method"),
    }

if __name__ == "__main__":
    args = parse_args()
    run_timer(**args)
