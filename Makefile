.PHONY: help install run tune test clean

help:
	@echo "install  install dependencies"
	@echo "run      full pipeline: data, snapshots, CV, holdout, figures"
	@echo "tune     as above, plus the XGBoost grid search"
	@echo "test     run the test suite (no network)"

install:
	pip install -r requirements.txt

run:
	python -m src.pipeline

tune:
	python -m src.pipeline --tune

test:
	python -m pytest -q

clean:
	rm -rf data/processed/* models/*.joblib reports/*.md reports/figures/*.png reports/*.csv
