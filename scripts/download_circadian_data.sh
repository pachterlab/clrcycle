#!/usr/bin/env bash
set -euo pipefail

# Download the two public GEO files required by the circadian workflow.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${root_dir}/data"
mkdir -p "${data_dir}"

curl -L --fail --retry 3 \
  -o "${data_dir}/GSE54650_series_matrix.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54650/matrix/GSE54650_series_matrix.txt.gz"
curl -L --fail --retry 3 \
  -o "${data_dir}/GPL6246.annot.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6246/annot/GPL6246.annot.gz"
