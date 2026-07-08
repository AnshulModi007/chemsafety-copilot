# One-time environment setup for ChemSafety Copilot.
# Run from the project root: scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Redirect heavy caches to C:\ai-cache (D: is space-constrained, C: has more headroom)
$CacheRoot = "C:\ai-cache"
New-Item -ItemType Directory -Force -Path "$CacheRoot\hf-cache", "$CacheRoot\pip-cache" | Out-Null

# Persistent (system-level) env var for the HuggingFace cache (embeddings + reranker downloads)
setx HF_HOME "$CacheRoot\hf-cache" | Out-Null
Write-Output "Set persistent HF_HOME to $CacheRoot\hf-cache (open a new terminal for this to take effect)."

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

if (-not $env:GROQ_API_KEY) {
    Write-Warning "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys and add it to .env before running the app."
}

Write-Output "Setup complete."
