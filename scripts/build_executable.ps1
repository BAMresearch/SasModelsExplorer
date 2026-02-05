Write-Host "Building SasModelsExplorer executable..."

if (Test-Path ".venv\\Scripts\\Activate.ps1") {
    . ".venv\\Scripts\\Activate.ps1"
}

python "scripts\\build_executable.py"
