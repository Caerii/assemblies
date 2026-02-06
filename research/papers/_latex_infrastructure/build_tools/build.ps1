# PowerShell Build Script for LaTeX Papers
# For Windows users without make installed
# Usage: .\build.ps1 -Mode [quick|full|clean|watch]

param(
    [string]$Mode = "quick",
    [string]$Paper = "main"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$LATEX = "pdflatex"
$BIBTEX = "bibtex"
$LATEX_FLAGS = "-interaction=nonstopmode", "-file-line-error", "-synctex=1"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-SubHeader {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Invoke-LaTeX {
    Write-SubHeader "Running pdflatex..."
    & $LATEX $LATEX_FLAGS "$Paper.tex"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LaTeX compilation failed!" -ForegroundColor Red
        return $false
    }
    return $true
}

function Invoke-BibTeX {
    Write-SubHeader "Running bibtex..."
    & $BIBTEX $Paper
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BibTeX failed (this may be okay if no citations)" -ForegroundColor Yellow
    }
}

# ============================================================================
# BUILD MODES
# ============================================================================

function Build-Quick {
    Write-Header "Quick Build (no bibliography)"
    
    if (Invoke-LaTeX) {
        Write-Host ""
        Write-Host "Build complete: $Paper.pdf" -ForegroundColor Green
        Write-Host "Use '.\build.ps1 -Mode full' for bibliography" -ForegroundColor Gray
    }
}

function Build-Full {
    Write-Header "Full Build (with bibliography)"
    
    Write-SubHeader "First pass..."
    if (-not (Invoke-LaTeX)) { return }
    
    Invoke-BibTeX
    
    Write-SubHeader "Second pass..."
    if (-not (Invoke-LaTeX)) { return }
    
    Write-SubHeader "Third pass (resolving references)..."
    if (-not (Invoke-LaTeX)) { return }
    
    Write-Host ""
    Write-Host "Full build complete: $Paper.pdf" -ForegroundColor Green
}

function Build-Watch {
    Write-Header "Watch Mode (auto-rebuild on save)"
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    
    # Check if latexmk is available
    if (Get-Command latexmk -ErrorAction SilentlyContinue) {
        & latexmk -pvc -pdf -pdflatex="$LATEX $($LATEX_FLAGS -join ' ')" "$Paper.tex"
    } else {
        Write-Host "latexmk not found. Using basic file watcher..." -ForegroundColor Yellow
        Write-Host "Install latexmk for better watch mode: tlmgr install latexmk" -ForegroundColor Gray
        Write-Host ""
        
        $lastWrite = (Get-Item "$Paper.tex").LastWriteTime
        
        while ($true) {
            Start-Sleep -Seconds 1
            $currentWrite = (Get-Item "$Paper.tex").LastWriteTime
            
            if ($currentWrite -ne $lastWrite) {
                Write-Host "File changed, rebuilding..." -ForegroundColor Cyan
                Build-Quick
                $lastWrite = $currentWrite
            }
        }
    }
}

function Clean-Auxiliary {
    Write-Header "Cleaning auxiliary files"
    
    $extensions = @(
        "*.aux", "*.bbl", "*.blg", "*.log", "*.out", "*.toc",
        "*.lof", "*.lot", "*.fls", "*.fdb_latexmk", "*.synctex.gz",
        "*.nav", "*.snm", "*.vrb", "*.idx", "*.ind", "*.ilg",
        "*.glo", "*.gls", "*.glg", "*.acn", "*.acr", "*.alg"
    )
    
    foreach ($ext in $extensions) {
        Remove-Item $ext -ErrorAction SilentlyContinue
    }
    
    Write-Host "Auxiliary files removed" -ForegroundColor Green
}

function Clean-All {
    Clean-Auxiliary
    
    Write-SubHeader "Removing PDF..."
    Remove-Item "$Paper.pdf" -ErrorAction SilentlyContinue
    
    Write-Host "All generated files removed" -ForegroundColor Green
}

function Show-Help {
    Write-Host ""
    Write-Host "LaTeX Build Script for Windows" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [-Mode <mode>] [-Paper <name>]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Modes:" -ForegroundColor Yellow
    Write-Host "  quick      - Fast compile without bibliography (default)"
    Write-Host "  full       - Full compile with bibliography"
    Write-Host "  watch      - Auto-rebuild on file changes"
    Write-Host "  clean      - Remove auxiliary files (keep PDF)"
    Write-Host "  cleanall   - Remove all generated files including PDF"
    Write-Host "  help       - Show this help message"
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -Paper     - Main paper file without .tex (default: main)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1"
    Write-Host "  .\build.ps1 -Mode full"
    Write-Host "  .\build.ps1 -Mode watch -Paper manuscript"
    Write-Host ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

switch ($Mode.ToLower()) {
    "quick"    { Build-Quick }
    "full"     { Build-Full }
    "watch"    { Build-Watch }
    "clean"    { Clean-Auxiliary }
    "cleanall" { Clean-All }
    "help"     { Show-Help }
    default {
        Write-Host "Unknown mode: $Mode" -ForegroundColor Red
        Write-Host "Use '.\build.ps1 -Mode help' for usage information"
    }
}
