# Universal Makefile for LaTeX Papers
# Copy this to your paper directory and modify PAPER variable

# ============================================================================
# CONFIGURATION
# ============================================================================

# Main paper file (without .tex extension)
PAPER = main

# LaTeX compiler
LATEX = pdflatex

# BibTeX compiler
BIBTEX = bibtex

# LaTeX flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error -synctex=1

# ============================================================================
# MAIN TARGETS
# ============================================================================

.PHONY: all quick full watch clean cleanall help

# Default target
all: quick

# Quick build (no bibliography, single pass)
# Use this during active writing for fast iteration
quick:
	@echo "=== Quick Build (no bibliography) ==="
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	@echo "=== Build complete: $(PAPER).pdf ==="

# Full build (with bibliography and cross-references)
# Use this before submission or when bibliography changes
full:
	@echo "=== Full Build (with bibliography) ==="
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	@echo "--- Running BibTeX ---"
	$(BIBTEX) $(PAPER)
	@echo "--- Second pass ---"
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	@echo "--- Third pass (resolving references) ---"
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	@echo "=== Build complete: $(PAPER).pdf ==="

# Watch mode (requires latexmk)
# Automatically rebuilds on file changes
watch:
	@echo "=== Watch Mode (auto-rebuild on save) ==="
	@echo "Press Ctrl+C to stop"
	latexmk -pvc -pdf -pdflatex="$(LATEX) $(LATEX_FLAGS)" $(PAPER).tex

# ============================================================================
# CLEANING TARGETS
# ============================================================================

# Clean auxiliary files (keeps PDF)
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
	      *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb \
	      *.idx *.ind *.ilg *.glo *.gls *.glg *.acn *.acr *.alg
	@echo "=== Clean complete ==="

# Clean everything including PDF
cleanall: clean
	@echo "=== Removing PDF ==="
	rm -f $(PAPER).pdf
	@echo "=== Complete clean ==="

# ============================================================================
# UTILITY TARGETS
# ============================================================================

# View PDF (platform-specific)
view: quick
	@echo "=== Opening PDF ==="
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PAPER).pdf; \
	elif command -v open > /dev/null; then \
		open $(PAPER).pdf; \
	elif command -v start > /dev/null; then \
		start $(PAPER).pdf; \
	else \
		echo "Cannot detect PDF viewer. Please open $(PAPER).pdf manually."; \
	fi

# Count words (approximate)
wordcount:
	@echo "=== Word Count (approximate) ==="
	@detex $(PAPER).tex | wc -w
	@echo "Note: This is an approximation. For accurate counts, use specific tools."

# Check for common LaTeX issues
lint:
	@echo "=== Checking for common issues ==="
	@echo "Checking for overfull hboxes..."
	@grep -n "Overfull" $(PAPER).log || echo "No overfull hboxes found"
	@echo "Checking for undefined references..."
	@grep -n "undefined" $(PAPER).log || echo "No undefined references"
	@echo "Checking for citation warnings..."
	@grep -n "Citation" $(PAPER).log || echo "No citation warnings"

# Help message
help:
	@echo "Available targets:"
	@echo "  make quick      - Fast compile without bibliography (default)"
	@echo "  make full       - Full compile with bibliography"
	@echo "  make watch      - Auto-rebuild on file changes (requires latexmk)"
	@echo "  make clean      - Remove auxiliary files (keep PDF)"
	@echo "  make cleanall   - Remove all generated files including PDF"
	@echo "  make view       - Compile and open PDF"
	@echo "  make wordcount  - Count approximate words"
	@echo "  make lint       - Check for common issues"
	@echo "  make help       - Show this help message"

# ============================================================================
# PLATFORM-SPECIFIC NOTES
# ============================================================================

# Windows (PowerShell):
#   Use: .\build.ps1 instead of make
#   Or install make via: choco install make
#
# macOS/Linux:
#   This Makefile works out of the box
#
# Alternative: Use VSCode with LaTeX Workshop extension
#   Provides GUI buttons and automatic compilation
