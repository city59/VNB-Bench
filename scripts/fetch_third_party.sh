#!/bin/bash
# Fetch the official model code into third_party/ at pinned commits, then verify the files.
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() {  # fetch <dest> <repo> <commit> <src>:<dst> ...
  dest=$1 repo=$2 commit=$3; shift 3
  # core.autocrlf=false is stored in the clone, so the checkout keeps upstream line endings for the hash check
  git clone --quiet -c core.autocrlf=false "$repo" "$tmp/$dest"
  git -C "$tmp/$dest" checkout --quiet "$commit"
  mkdir -p "third_party/$dest"
  for pair in "$@"; do
    src=${pair%%:*} dst=${pair##*:}
    rm -rf "third_party/$dest/$dst"
    mkdir -p "third_party/$dest/$(dirname "$dst")"
    cp -r "$tmp/$dest/$src" "third_party/$dest/$dst"
  done
  [ -f "$tmp/$dest/LICENSE" ] && cp "$tmp/$dest/LICENSE" "third_party/$dest/LICENSE"
  echo "fetched $dest"
}

fetch mbstr https://github.com/yuanenming/mb-str 6ead831 src/models:models
fetch pbat https://github.com/TiliaceaeSU/PBAT 91860ec src/models:models
fetch missl https://github.com/qianlima-lab/MISSL 44a9847 src/models:models
fetch mbht https://github.com/yuh-yang/MBHT-KDD22 4a2f3cf \
  recbole/model/sequential_recommender/mbht.py:mbht.py recbole/model/layers.py:layers.py \
  recbole/model/abstract_recommender.py:abstract_recommender.py
fetch mbgen https://github.com/anananan116/MBGen 268a5d0 \
  model/PBA_transformer.py:PBA_transformer.py trainer/evaluation.py:evaluation.py

(cd third_party && sha256sum -c SHA256SUMS)
