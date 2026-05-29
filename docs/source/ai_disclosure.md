# AI Assistance Disclosure

This tool was written with the assistance of AI coding agents (ChatGPT).

We used AI for the following scripts:

* `markov_random_field_zarr.py` and `markov_random_field_zarr_parallel.py`
    * an intial version was written `markov_random_field.py` by hand
    * the first version was then optimized (for runtime and memory) by AI leading to the aforementioned scripts
* `metrics/`
    * several metrics were written by AI
* `pixel_scoring_dask.py`
    * an intial version was written by hand
    * the first version was then optimized (for runtime and memory) by AI leading to the aforementioned scripts
* `Dockerfile`
    * Dockerfile was intially written by AI and optimized by hand
* AI added to many scripts docstrings and type hints

Correctness was validated by equal comparison of the output of the different implementations. Humans defined the validation criteria and verified the results.

