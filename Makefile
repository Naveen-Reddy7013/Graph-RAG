# Makefile for GraphRAG Book Summarizer Pipeline

.PHONY: install run test verify clean

# Detect if we are running on Windows or Unix/macOS to select correct paths
ifeq ($(OS),Windows_NT)
    PYTHON = .venv/Scripts/python
    PYTEST = .venv/Scripts/pytest
else
    PYTHON = .venv/bin/python
    PYTEST = .venv/bin/pytest
endif

# 1. Installs the virtual environment and packages using uv
install:
	@echo "Creating virtual environment using uv..."
	uv venv .venv
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt

# 2. Runs the ingestion and query pipeline with a default query
run:
	@echo "Running GraphRAG pipeline..."
	$(PYTHON) run.py --query "How is Tony Stark connected to Pepper Potts?" --mode local --max-hops 2 --output-format json

# 3. Runs unit and integration test suites
test:
	@echo "Running tests..."
	$(PYTEST) tests/

# 4. Executes output verification
verify:
	@echo "Running schema and database path verification..."
	$(PYTHON) verify.py

# 5. Cleans up caches and environment
clean:
	@echo "Cleaning up temporary files..."
	rm -rf .venv output/ src/__pycache__ src/agents/__pycache__ tests/__pycache__ .pytest_cache
