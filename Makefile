PY=python
RUN=uvicorn neo.api.app:create_app --factory --host 0.0.0.0 --port 8000

.PHONY: install dev run test lint type format docker compose

install:
	$(PY) -m pip install -e .[dev]

dev: install
	uvicorn neo.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

run:
	$(RUN)

test:
	pytest -q

lint:
	ruff check src/neo

type:
	mypy src/neo

format:
	ruff check --fix src/neo || true
	black src/neo

docker:
	docker build -t neo-api:latest .

compose:
	docker compose up --build
