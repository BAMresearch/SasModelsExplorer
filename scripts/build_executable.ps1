Write-Host "Building SasModelsExplorer executable..."

$python = "python"
if (Test-Path ".venv\\Scripts\\python.exe") {
    $python = ".venv\\Scripts\\python.exe"
}

& $python -m tox -e standalone

if ($LASTEXITCODE -ne 0) {
    Write-Host "If tox is missing, install it with: python -m pip install tox"
}

exit $LASTEXITCODE
