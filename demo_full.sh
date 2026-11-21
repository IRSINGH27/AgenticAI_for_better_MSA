#!/bin/bash
# =============================================================================
# MSA AI Pipeline - Full Demonstration
# =============================================================================
# This demo showcases the AI-powered iterative MSA pipeline with RAG learning.
# Pipeline learns from each experiment stored in /home/user/AI/knowledge/
# =============================================================================

set -e

cd "$(dirname "$0")"
PIPELINE_DIR=$(pwd)
KNOWLEDGE_DIR="$PIPELINE_DIR/knowledge"
LOGS_DIR="$PIPELINE_DIR/logs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}>>>${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

pause() {
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# =============================================================================
# STEP 0: Environment check and knowledge reset
# =============================================================================
print_header "MSA AI Pipeline - Full Demonstration"

echo "Project location: $PIPELINE_DIR"
echo "Knowledge base:   $KNOWLEDGE_DIR"
echo "  Three-layer storage:"
echo "    Layer 1 (TOON, text, LLM-friendly):"
echo "      - experiments.toon      (all experiment records)"
echo "      - strategy_stats.toon   (per-strategy success rates)"
echo "    Layer 2 (JSON, legacy / config):"
echo "      - experiments.jsonl     (legacy JSONL, fallback reader)"
echo "      - strategy_stats.json   (legacy JSON)"
echo "    Layer 3 (Sparse NumPy, fast search):"
echo "      - tfidf_index.npz        (sparse CSR matrix)"
echo "      - tfidf_index_vocab.json (vocab + experiment_ids)"
echo "      - user_feedback.json     (user ratings)"
echo ""
echo "Test files:"
ls -la test*.fasta 2>/dev/null | awk '{print "  -", $NF, "(" $5 " bytes)"}'
echo ""

read -p "Reset knowledge base for fresh demo? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Clearing all knowledge and logs..."
    rm -f "$KNOWLEDGE_DIR"/experiments.jsonl
    rm -f "$KNOWLEDGE_DIR"/experiments.toon
    rm -f "$KNOWLEDGE_DIR"/strategy_stats.json
    rm -f "$KNOWLEDGE_DIR"/strategy_stats.toon
    rm -f "$KNOWLEDGE_DIR"/tfidf_index.npz
    rm -f "$KNOWLEDGE_DIR"/tfidf_index_vocab.json
    rm -f "$KNOWLEDGE_DIR"/user_feedback.json
    rm -f "$KNOWLEDGE_DIR"/external_knowledge.json
    rm -f "$LOGS_DIR"/*.jsonl
    echo "  Cleared."
else
    print_info "Continuing with existing knowledge base..."
fi

echo ""
print_info "Verifying environment..."
if ! conda run -n seqTools mafft --version > /dev/null 2>&1; then
    print_warn "MAFFT not found in seqTools env"
    exit 1
fi
if ! conda run -n seqTools muscle --version > /dev/null 2>&1; then
    print_warn "MUSCLE not found in seqTools env"
fi
if ! conda run -n seqTools trimal --version > /dev/null 2>&1; then
    print_warn "trimal not found in seqTools env"
fi
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    print_warn "Ollama not running - LLM features will fail"
fi
echo "  Environment ready."

pause

# =============================================================================
# STEP 1: Run on real UniProt data (dnaA/sigma factor proteins)
# =============================================================================
print_header "STEP 1: Align Real UniRef Protein Sequences"
echo "File: test_uniref_dnaA_sigma.fasta"
echo "  - 50 protein sequences (RNA polymerase sigma-H factor)"
echo "  - Real biological data from UniProt"
echo "  - Expected: AI explores strategies, possibly switches tools, may trim"
echo ""
print_step "Running pipeline (max 5 iterations)..."
echo ""

python -u run_pipeline.py test_uniref_dnaA_sigma.fasta -n 5 --tolerance 0.005 --patience 2 2>&1 | tee /tmp/pipeline_step1.log

pause

# =============================================================================
# STEP 2: Run on the same file again - show learning/duplicate detection
# =============================================================================
print_header "STEP 2: Re-run Same File - Show Duplicate Detection & Learning"
echo "Running same file again. The knowledge base should now skip this as duplicate."
echo "Or if RAG retrieves the prior experience, the AI starts smarter."
echo ""
print_step "Running pipeline again on same file..."
echo ""

python -u run_pipeline.py test_uniref_dnaA_sigma.fasta -n 4 2>&1 | tee /tmp/pipeline_step2.log

pause

# =============================================================================
# STEP 3: Run on diverse DNA sequences - show tool switching
# =============================================================================
print_header "STEP 3: Diverse DNA Sequences - Test Tool Switching"
echo "File: test_variable.fasta (8 diverse DNA sequences, high gap fraction)"
echo "Expected: MAFFT fails, AI auto-switches to MUSCLE, then trims"
echo ""
print_step "Running pipeline..."
echo ""

python -u run_pipeline.py test_variable.fasta -n 5 2>&1 | tee /tmp/pipeline_step3.log

pause

# =============================================================================
# STEP 4: Dashboard - show what was learned
# =============================================================================
print_header "STEP 4: Learning Dashboard - Knowledge Base Contents"
echo "Displaying accumulated knowledge..."
echo ""

print_step "Knowledge base files:"
ls -la "$KNOWLEDGE_DIR"
echo ""

print_step "Experiment count (TOON):"
if [ -f "$KNOWLEDGE_DIR/experiments.toon" ]; then
    wc -l "$KNOWLEDGE_DIR/experiments.toon"
    echo ""
    echo "Last experiment (TOON, human-readable):"
    awk '/^- experiment_id:/ {p=1} p' "$KNOWLEDGE_DIR/experiments.toon" | head -20
fi
echo ""

print_step "Strategy performance:"
python scripts/dashboard.py --strategies

pause

# =============================================================================
# STEP 5: Show dashboard summary
# =============================================================================
print_header "STEP 5: Full Dashboard Summary"
python scripts/dashboard.py --dashboard

pause

# =============================================================================
# STEP 6: Learning timeline
# =============================================================================
print_header "STEP 6: Learning Timeline (chronological progress)"
python scripts/dashboard.py --timeline

pause

# =============================================================================
# STEP 7: Cross-validation
# =============================================================================
print_header "STEP 7: Cross-Validation - How Well Did We Learn?"
python scripts/dashboard.py --validate

pause

# =============================================================================
# STEP 8: Detailed report
# =============================================================================
print_header "STEP 8: Detailed Report"
python scripts/dashboard.py --report

pause

# =============================================================================
# STEP 8b: Storage format inspection (TOON + sparse NumPy)
# =============================================================================
print_header "STEP 8b: Storage Format - Three-Layer Architecture"

print_step "Format info (TOON / JSONL / sparse layout):"
python scripts/dashboard.py --format-info

pause

print_step "Sparse index details (matrix shape, vocab, density):"
python scripts/dashboard.py --sparse-info

pause

print_step "TOON context preview (what the LLM actually sees):"
python scripts/dashboard.py --toon-context

pause

print_step "Format comparison (sizes side by side):"
python scripts/dashboard.py --compare-formats

pause

# =============================================================================
# STEP 9: Export knowledge base
# =============================================================================
print_header "STEP 9: Export Knowledge Base (Share With Others)"
EXPORT_FILE="demo_export_$(date +%Y%m%d_%H%M%S).zip"
python scripts/dashboard.py --export "$EXPORT_FILE"
echo ""
print_info "Export created: $PIPELINE_DIR/$EXPORT_FILE"

pause

# =============================================================================
# STEP 10: Provide feedback
# =============================================================================
print_header "STEP 10: Provide User Feedback"
echo "Rate the most recent experiment to help the AI learn preferences."
echo ""

if [ -f "$KNOWLEDGE_DIR/experiments.jsonl" ]; then
    LAST_ID=$(tail -1 "$KNOWLEDGE_DIR/experiments.jsonl" | python -c "import json,sys; print(json.loads(sys.stdin.read())['experiment_id'])")
    echo "Last experiment: $LAST_ID"
    echo ""
    read -p "Rate this experiment (1-5, or 0 to skip): " RATING
    if [[ "$RATING" =~ ^[1-5]$ ]]; then
        python scripts/dashboard.py --feedback "$LAST_ID" --rating "$RATING" 2>/dev/null || \
        echo "(feedback system may not support this exact syntax - skip)"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
print_header "DEMO COMPLETE - Summary"

echo "What we did:"
echo "  1. Aligned 50 real UniProt protein sequences"
echo "  2. Re-ran to test duplicate detection / RAG learning"
echo "  3. Ran on diverse DNA to test tool switching (MAFFT -> MUSCLE)"
echo "  4. Inspected knowledge base contents (TOON format)"
echo "  5. Reviewed strategy performance and learning timeline"
echo "  6. Cross-validated learned strategies"
echo "  7. Exported knowledge base for sharing"
echo "  8. Inspected the three-layer storage (TOON / JSONL / sparse)"
echo ""

echo "Knowledge base location:"
echo "  $KNOWLEDGE_DIR"
echo ""
echo "Files in knowledge base:"
ls -la "$KNOWLEDGE_DIR" 2>/dev/null | tail -n +2 | awk '{print "  -", $NF}'

echo ""
echo "Pipeline log location: $LOGS_DIR"
echo ""
echo "Useful follow-up commands:"
echo "  python run_pipeline.py <fasta> -n 10     # Run on your own data"
echo "  python scripts/dashboard.py --dashboard  # View learning summary"
echo "  python scripts/dashboard.py --strategies # View per-strategy success"
echo "  python scripts/dashboard.py --timeline   # View learning history"
echo "  python scripts/dashboard.py --export x.zip  # Share knowledge"
echo ""
echo "Storage format tools (TOON / sparse):"
echo "  python scripts/dashboard.py --format-info    # Show storage layout"
echo "  python scripts/dashboard.py --sparse-info    # Sparse index details"
echo "  python scripts/dashboard.py --toon-context   # Preview LLM context"
echo "  python scripts/dashboard.py --compare-formats # Compare JSONL vs TOON vs sparse"
echo "  python scripts/dashboard.py --inspect        # List all experiments"
echo "  python scripts/dashboard.py --inspect <id>   # Inspect specific experiment"
echo ""
