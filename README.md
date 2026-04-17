# AI4LT

- Tongle(Tommy) Shen

DSC 291 class repo. AI for Learning Theory. Aim at exploring how to use agentic skills for theory works. 

## Environment

This repo uses a dedicated Conda environment named `ai4lt`.

Create it with:

```bash
conda env create -f environment.yml
```

Activate it before running any repo workflow:

```bash
conda activate ai4lt
```

Install the `pdflatex` toolchain once after activation:

```bash
bash scripts/setup_pdflatex.sh
```

This installs TinyTeX under `.tools/.TinyTeX`, installs the repo's required LaTeX
packages, and adds an activation hook so future `conda activate ai4lt` sessions
automatically expose the correct `pdflatex` binary.

Quick verification:

```bash
python --version
pdflatex --version
```

## Dependencies

The tracked Conda dependencies live in `environment.yml`. Current core tools:

- `python=3.11`
- `numpy` for numerical work
- `matplotlib` for plots and experiment figures
- `ipykernel` for notebook support
- `pypdf` for extracting assignment and slide text into reusable statement files
- `pip`

Non-Conda repo tools:

- `pdflatex`, installed via `bash scripts/setup_pdflatex.sh`
- TinyTeX, stored under `.tools/.TinyTeX`

When new tools are installed for this repo, record them in `environment.yml` or the setup script and reflect the change here.

## Workflow Note

Local skills in `AGENTS.md` are expected to verify that the `ai4lt` environment is active and that `pdflatex` is available before doing homework or LaTeX work. If `pdflatex` is missing, run `bash scripts/setup_pdflatex.sh` and then continue.

## LaTeX Build

Default compiler:

```bash
pdflatex -interaction=nonstopmode -halt-on-error hw1.tex
```
