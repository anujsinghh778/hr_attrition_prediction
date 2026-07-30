.PHONY: setup data train test run docker-build docker-run all

setup:
	pip install -r requirements.txt

data:
	python src/data_prep.py

train:
	python src/data_prep.py
	python src/train_model.py
	python src/generate_pdf.py

test:
	pytest tests/

run:
	streamlit run dashboard/streamlit_app.py

docker-build:
	docker build -t hr-attrition-app .

docker-run:
	docker run -p 8501:8501 hr-attrition-app

all: setup train test run
