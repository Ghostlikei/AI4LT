#!/usr/bin/env bash

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ai4lt" ]]; then
  echo "Activate the Conda environment 'ai4lt' first." >&2
  exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is not available. Run 'bash scripts/setup_pdflatex.sh' first." >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/compile_tex_to_pdf.sh path/to/file.tex" >&2
  exit 1
fi

tex_path="$1"

if [[ ! -f "${tex_path}" ]]; then
  echo "TeX file not found: ${tex_path}" >&2
  exit 1
fi

tex_abs="$(realpath "${tex_path}")"
tex_dir="$(dirname "${tex_abs}")"
tex_file="$(basename "${tex_abs}")"
pdf_file="${tex_file%.tex}.pdf"

if [[ "${tex_file}" == "${pdf_file}" ]]; then
  echo "Expected a .tex file, got: ${tex_file}" >&2
  exit 1
fi

(
  cd "${tex_dir}"
  pdflatex -interaction=nonstopmode -halt-on-error "${tex_file}"
  pdflatex -interaction=nonstopmode -halt-on-error "${tex_file}"
)

echo "Compiled PDF: ${tex_dir}/${pdf_file}"
