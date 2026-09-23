.PHONY: setup lint format test run docker-build docker-run clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-dev.txt
	python -m pre_commit install

lint:
	python -m black --check src deployment flows tests
	python -m isort --check-only src deployment flows tests
	python -m flake8 src deployment flows tests

format:
	python -m black src deployment flows tests
	python -m isort src deployment flows tests

test:
	python -m pytest tests/ -q

run:
	python -m uvicorn deployment.main:app --reload

docker-build:
	docker build -t predictive-maintenance:latest -f deployment/Dockerfile .

docker-run:
	docker run -p 8000:8000 predictive-maintenance:latest

clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(path) for path in Path('.').rglob('__pycache__')]"


# ------------------------
# 🧹 Code Quality with Pre-commit
# ------------------------

# Install pre-commit and set up git hook
precommit-init:
	python -m pip install pre-commit
	python -m pre_commit install
	echo "Pre-commit installed and configured"

# Run all pre-commit hooks manually
precommit-run:
	python -m pre_commit run --all-files
