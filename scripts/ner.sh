python run.py \

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1   # run from repo root
--config_file resume.bert-base-uncased.ner.yaml \
--base ner \