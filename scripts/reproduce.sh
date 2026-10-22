#!/usr/bin/env bash
set -euo pipefail

python -m kp_classifier.data.export_mysql
python -m kp_classifier.data.split
python -m kp_classifier.data.tokenizer

python -m kp_classifier.models.baseline --eval-test

for seed in 42 1 2; do
  python -m kp_classifier.train --config configs/hybrid_final.yaml --name "maxmean_ls01_s${seed}" --set "seed=${seed}"
done
python -m kp_classifier.train --config configs/hybrid_final.yaml --name words_mm_ls_s42 --set model.use_chars=false

python -m kp_classifier.ensemble --name ens_mm_ls_cal \
  --models maxmean_ls01_s42 maxmean_ls01_s1 maxmean_ls01_s2 --calibrate --eval-test

python -m kp_classifier.evaluate mcnemar baseline_tfidf ens_mm_ls_cal --split test
python -m kp_classifier.robustness --models baseline_tfidf words_mm_ls_s42 maxmean_ls01_s42 ens_mm_ls_cal \
  --split test --out-dir reports/robustness_test
python -m kp_classifier.evaluate compare --split test
