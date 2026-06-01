.PHONY: run train test mlflow clean

run:
	python -m src.data.ingest
	python -m src.data.validate
	python -m src.features.rfm
	python -m src.features.churn_features
	python -m src.models.clv_model
	python -m src.models.churn_model
	python -m app.export_powerbi

train:
	python -m src.models.clv_model
	python -m src.models.churn_model

test:
	pytest tests/ -v

mlflow:
	mlflow ui --port 5000

clean:
	rm -f data/processed/*.parquet data/processed/*.csv data/processed/*.xlsx
	rm -f models/*.joblib models/*.png
	@echo "Cleaned processed data and model artefacts."
