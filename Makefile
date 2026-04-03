.PHONY: install test test-fast test-semantic test-integration lint typecheck coverage clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not slow and not integration"

test-semantic:
	pytest tests/ -v --tb=short -m "semantic"

test-integration:
	pytest tests/ -v --tb=short -m "integration"

lint:
	ruff check src/ tests/ pytest_semantic/
	ruff format --check src/ tests/ pytest_semantic/

typecheck:
	mypy src/ pytest_semantic/

coverage:
	pytest tests/ --cov=src --cov=pytest_semantic --cov-report=html --cov-report=term-missing

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +