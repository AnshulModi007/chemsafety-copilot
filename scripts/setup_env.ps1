# One-time environment setup for ChemSafety Copilot.
# Run from the project root: scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Redirect heavy caches to C:\ai-cache (D: is space-constrained, C: has more headroom)
$CacheRoot = "C:\ai-cache"
New-Item -ItemType Directory -Force -Path "$CacheRoot\hf-cache", "$CacheRoot\pip-cache", "$CacheRoot\ollama-models" | Out-Null

# Persistent (system-level) env vars, since Ollama runs as its own background service
setx OLLAMA_MODELS "$CacheRoot\ollama-models" | Out-Null
setx HF_HOME "$CacheRoot\hf-cache" | Out-Null
Write-Output "Set persistent OLLAMA_MODELS and HF_HOME to $CacheRoot (restart Ollama / open a new terminal for this to take effect)."

# .env for the Python process (mirrors the above + adds pip cache dir)
if (-not (Test-Path "$ProjectRoot\.env")) {
    Copy-Item "$ProjectRoot\.env.example" "$ProjectRoot\.env"
    Write-Output "Created .env from .env.example"
}

# venv on D: (project directory), packages themselves are small text/metadata plus
# torch/etc. under site-packages -- keep an eye on this if D: fills up again
if (-not (Test-Path "$ProjectRoot\.venv")) {
    python -m venv "$ProjectRoot\.venv"
    Write-Output "Created venv at $ProjectRoot\.venv"
}

& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$ProjectRoot\.venv\Scripts\pip.exe" install -r "$ProjectRoot\requirements.txt"

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Warning "Ollama not found. Install it manually (winget install Ollama.Ollama or https://ollama.com/download), then re-run this script."
    exit 1
}

Write-Output "Ollama found: $($ollama.Source)"
Write-Output "Pulling llama3.1:8b-instruct-q4_K_M (this will use OLLAMA_MODELS=$env:OLLAMA_MODELS for THIS session; a fresh terminal/service restart is needed for the setx value to apply)..."
$env:OLLAMA_MODELS = "$CacheRoot\ollama-models"
ollama pull llama3.1:8b-instruct-q4_K_M

Write-Output "Setup complete."
