# Initial validation environment

The saved validation run was executed on Windows 11 using the repository virtual environment at `.venv/`.

Recorded Python facts:

- Python 3.11.9
- Virtual environment created with `python -m venv C:\Users\boing\jev-edge\.venv`
- `include-system-site-packages = false`
- The runner used Python standard-library HTTP and serialization modules.

Recorded service facts:

- llama.cpp service base URL: `http://127.0.0.1:8081`
- StartLux typed-decision API: `http://127.0.0.1:8090/v1/systemone`
- Model identity returned by all three validation responses: `startlux-q8`
- Validation endpoint: `POST /v1/systemone`
- Validation request count: 3 measured requests, no warm-ups

The virtual environment package snapshot used during quality control was:

`annotated-doc==0.0.5`, `anyio==4.15.1`, `click==8.5.0`, `colorama==0.4.6`, `filelock==4.2.4`, `fsspec==2026.9.0`, `h11==0.16.0`, `hf-xet==1.7.0`, `httpcore2==2.16.0`, `httpx2==2.16.0`, `huggingface_hub==2.3.0`, `idna==3.20`, `Jinja2==3.1.6`, `markdown-it-py==4.2.0`, `MarkupSafe==3.1.6`, `mpmath==1.3.0`, `networkx==3.6.1`, `numpy==2.4.6`, `packaging==26.3`, `Pygments==2.21.0`, `PyYAML==6.0.3`, `regex==2026.9.29`, `safetensors==0.8.0`, `sympy==1.14.0`, `tokenizers==0.23.3`, `torch==2.14.1`, `tqdm==4.70.1`, `transformers==5.20.2`, `truststore==0.10.5`, `typer==0.27.3`, and `typing_extensions==4.16.0`.

The repository, saved response headers, and process inventory do not contain the llama.cpp executable command line or flags used for the initial run. Context size, batch size, thread count, GPU offload, and flash-attention settings are therefore unknown for that run. They must be captured from the launch command or server startup log before a performance comparison. No unsupported defaults are inferred here.
