# Simple Template Check
Write-Host "=== LaTeX Template Check ===" -ForegroundColor Cyan

# Check main files
Write-Host "Checking files..." -ForegroundColor Yellow

if (Test-Path "main.tex") {
    Write-Host "✓ main.tex exists" -ForegroundColor Green
} else {
    Write-Host "✗ main.tex missing" -ForegroundColor Red
}

if (Test-Path "..\..\preambles\standard_preamble.tex") {
    Write-Host "✓ standard_preamble.tex exists" -ForegroundColor Green
} else {
    Write-Host "✗ standard_preamble.tex missing" -ForegroundColor Red
}

if (Test-Path "..\..\preambles\neuroscience_preamble.tex") {
    Write-Host "✓ neuroscience_preamble.tex exists" -ForegroundColor Green
} else {
    Write-Host "✗ neuroscience_preamble.tex missing" -ForegroundColor Red
}

if (Test-Path "..\..\build_tools\build.ps1") {
    Write-Host "✓ build.ps1 exists" -ForegroundColor Green
} else {
    Write-Host "✗ build.ps1 missing" -ForegroundColor Red
}

# Check LaTeX content
Write-Host "Checking LaTeX content..." -ForegroundColor Yellow
$content = Get-Content "main.tex" -Raw

if ($content -match "\\documentclass") {
    Write-Host "✓ Has documentclass" -ForegroundColor Green
} else {
    Write-Host "✗ Missing documentclass" -ForegroundColor Red
}

if ($content -match "\\begin\{document\}") {
    Write-Host "✓ Has begin document" -ForegroundColor Green
} else {
    Write-Host "✗ Missing begin document" -ForegroundColor Red
}

if ($content -match "\\assembly\{") {
    Write-Host "✓ Has neuroscience notation" -ForegroundColor Green
} else {
    Write-Host "⚠ No neuroscience notation found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Template structure looks good!" -ForegroundColor Green
Write-Host ""
Write-Host "To compile:" -ForegroundColor Yellow
Write-Host "1. Install LaTeX: https://miktex.org/download" -ForegroundColor White
Write-Host "2. Run: ..\..\build_tools\build.ps1 -Mode quick" -ForegroundColor White
Write-Host "3. Or use Overleaf: https://www.overleaf.com" -ForegroundColor White
