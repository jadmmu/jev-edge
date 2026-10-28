# JEV-on-Pi benchmark methodology

The first validation run targets the local StartLux typed-decision API at `http://127.0.0.1:8090/v1/systemone` and the Q8 model configured behind it. The fixed dataset contains three support-ticket states. Each request asks one choice, one yes/no (`noul`), and one ordinal score question.

The runner records the dataset question ID, exact request body, UTC timestamp, HTTP status, response headers, complete response body, client wall-clock latency, model identity, usage counts, and endpoint metadata. Metadata is collected with `GET /health` and `GET /v1/models` before measured POST requests. Warm-up observations are retained with `is_warmup: true` and excluded from summary statistics.

The validation run had no warm-up trials. Future experiments use `--warmup-trials` and `--repeats` to separate warm-up requests from measured trials. The three observations are a connectivity and schema check, not an accuracy estimate. Confidence values are model outputs and are not empirical accuracy.

Measured summaries report median, linearly interpolated p90 and p95, mean, population standard deviation, measured trial count, and failure rate. Probability validation checks answer keys, nonnegative values bounded by one, and sums within a fixed tolerance of one.

The planned 20-question run is reproducible with the configuration in `experiments/configs/benchmark_q8_20.json` and equivalent runner parameters: `python experiments/benchmark_runner.py --dataset experiments/datasets/benchmark_20.jsonl --endpoint http://127.0.0.1:8090/v1/systemone --repeats 1 --warmup-trials 1 --jsonl results/raw/q8_fixed_20.jsonl --csv results/q8_fixed_20.csv --summary results/tables/q8_fixed_20_summary.json`. This command is documented for the next run and was not executed during quality control.

Raw JSONL observations are stored under `results/raw/`. A flat CSV and a JSON summary accompany each run for inspection and later aggregation. The dataset and runner are under `experiments/` so later Q8/Q4 comparisons can reuse the same request records. The 20-question configuration is `experiments/configs/benchmark_q8_20.json`; it is defined but has not been run.
