# Template Validation Script
# Checks LaTeX template structure and syntax without requiring compilation

Write-Host "=== LaTeX Template Validation ===" -ForegroundColor Cyan
Write-Host ""

# Check if main template exists
Write-Host "Checking main template..." -ForegroundColor Yellow
if (Test-Path "main.tex") {
    Write-Host "✓ main.tex exists" -ForegroundColor Green
} else {
    Write-Host "✗ main.tex missing" -ForegroundColor Red
    exit 1
}

# Check preamble files
Write-Host "Checking preamble files..." -ForegroundColor Yellow
$preambles = @(
    "..\..\preambles\standard_preamble.tex",
    "..\..\preambles\neuroscience_preamble.tex"
)

foreach ($preamble in $preambles) {
    if (Test-Path $preamble) {
        Write-Host "✓ $preamble exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $preamble missing" -ForegroundColor Red
    }
}

# Check bibliography
Write-Host "Checking bibliography..." -ForegroundColor Yellow
$bib_path = "..\..\..\..\..\research\papers\_shared_assets\bibliography\references.bib"
if (Test-Path $bib_path) {
    Write-Host "✓ references.bib exists" -ForegroundColor Green
} else {
    Write-Host "✗ references.bib missing" -ForegroundColor Red
}

# Check build tools
Write-Host "Checking build tools..." -ForegroundColor Yellow
if (Test-Path "..\..\build_tools\build.ps1") {
    Write-Host "✓ build.ps1 exists" -ForegroundColor Green
} else {
    Write-Host "✗ build.ps1 missing" -ForegroundColor Red
}

# Check LaTeX syntax in main.tex
Write-Host "Checking LaTeX syntax..." -ForegroundColor Yellow
$content = Get-Content "main.tex" -Raw

# Check for common LaTeX issues
$issues = @()

if ($content -notmatch "\\documentclass") {
    $issues += "Missing \\documentclass"
}

if ($content -notmatch "\\begin\{document\}") {
    $issues += "Missing \\begin{document}"
}

if ($content -notmatch "\\end\{document\}") {
    $issues += "Missing \\end{document}"
}

if ($content -match "\\input\{[^}]+\}") {
    Write-Host "✓ Contains \\input commands" -ForegroundColor Green
} else {
    $issues += "No \\input commands found"
}

if ($issues.Count -eq 0) {
    Write-Host "✓ LaTeX syntax looks good" -ForegroundColor Green
} else {
    Write-Host "✗ LaTeX syntax issues:" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "  - $issue" -ForegroundColor Red
    }
}

# Check neuroscience notation
Write-Host "Checking neuroscience notation..." -ForegroundColor Yellow
$neuro_commands = @(
    "\\assembly\{",
    "\\neurons\{", 
    "\\activation\{",
    "\\TopK",
    "\\kactive"
)

$neuro_found = 0
foreach ($cmd in $neuro_commands) {
    if ($content -match [regex]::Escape($cmd)) {
        $neuro_found++
    }
}

if ($neuro_found -gt 0) {
    Write-Host "✓ Found $neuro_found neuroscience commands" -ForegroundColor Green
} else {
    Write-Host "⚠ No neuroscience commands found in main.tex" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=== Validation Summary ===" -ForegroundColor Cyan

if ($issues.Count -eq 0) {
    Write-Host "✅ Template structure is valid!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Install LaTeX (MiKTeX or TeX Live)" -ForegroundColor White
    Write-Host "2. Run: ..\..\build_tools\build.ps1 -Mode quick" -ForegroundColor White
    Write-Host "3. Or use Overleaf: https://www.overleaf.com" -ForegroundColor White
} else {
    Write-Host "❌ Template has issues that need fixing" -ForegroundColor Red
}

Write-Host ""
Write-Host "Files in this directory:" -ForegroundColor Yellow
Get-ChildItem -Name | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }