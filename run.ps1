# PowerShell Run Script for GraphRAG Book Summarizer Pipeline

# Exit immediately if any command fails
$ErrorActionPreference = "Stop"

# Write headers in a distinct cyan color
function Write-Header ($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

# 1. Environment creation and package installation
Write-Header "Step 1: Checking and Setting Up Virtual Environment"
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment using uv..."
    uv venv .venv
} else {
    Write-Host "Virtual environment .venv already exists."
}

Write-Host "Installing dependencies using uv..."
# uv pip install automatically targets the local .venv in the folder
uv pip install -r requirements.txt

# 2. Run unit and integration tests
Write-Header "Step 2: Running Unit & Integration Tests"
& .venv/Scripts/pytest tests/

# 3. Run the ingestion and local search query
Write-Header "Step 3: Executing Ingestion & Local Search Query"
Write-Host "Query: 'How is Tony Stark connected to Pepper Potts?'"
& .venv/Scripts/python run.py --query "How is Tony Stark connected to Pepper Potts?" --mode local --max-hops 2 --output-format json

# 4. Verify JSON schema output correctness
Write-Header "Step 4: Running Verification checks"
& .venv/Scripts/python verify.py

Write-Header "PIPELINE COMPLETED SUCCESSFULLY!"
