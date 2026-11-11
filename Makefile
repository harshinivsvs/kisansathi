.PHONY: help setup test run ingest eval docker-build docker-up

PYTHON = python

help:
	@echo "KisanSathi Automation Commands"
	@echo "  make setup        - Install dependencies"
	@echo "  make ingest       - Ingest ICAR and university agronomy advisories"
	@echo "  make run          - Launch FastAPI advisory backend"
	@echo "  make test         - Run test suite"
	@echo "  make eval         - Run RAG benchmark and evaluation harness"
	@echo "  make docker-up    - Start services with Docker Compose"

setup:
	pip install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest_data.py

run:
	$(PYTHON) -m uvicorn backend.main:app --reload --port 8000

test:
	pytest tests/ -v

eval:
	$(PYTHON) eval/run_eval.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d
