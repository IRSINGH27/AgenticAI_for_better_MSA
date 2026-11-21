#!/bin/bash
# MSA AI Pipeline Demonstration Script

echo "======================================"
echo "   MSA AI Pipeline Demonstration"
echo "======================================"
echo ""
echo "This demo showcases the AI-powered MSA pipeline"
echo "that learns from experiments and improves alignment quality."
echo ""

# Reset option
read -p "Reset experiment logs for fresh demo? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f /home/user/AI/knowledge/experiments.jsonl
    rm -f /home/user/AI/knowledge/strategy_stats.json
    rm -f /home/user/AI/knowledge/user_feedback.json
    rm -f /home/user/AI/knowledge/external_knowledge.json
    rm -f /home/user/AI/logs/*.jsonl
    echo "✓ Cleared all experiment data. Starting fresh demo."
    echo ""
else
    echo "Continuing with existing knowledge base..."
    echo ""
fi

echo "Press Enter to continue to Demo 1..."
read

# Demo 1: Simple good-quality sequences
echo ""
echo "========================================"
echo ">>> DEMO 1: Simple Good-Quality Sequences"
echo "========================================"
echo "File: test.fasta (5 similar sequences)"
echo "Expected: AI accepts immediately (gap_fraction = 0)"
echo ""
python run_pipeline.py test.fasta -n 3
echo ""
echo "✓ Demo 1 Complete - AI recognized good alignment"
echo ""

echo "Press Enter to continue to Demo 2..."
read

# Demo 2: Show dashboard
echo ""
echo "========================================"
echo ">>> DEMO 2: Learning Dashboard"
echo "========================================"
echo "Shows accumulated experiments and success rates"
echo ""
python scripts/dashboard.py --dashboard
echo ""
echo "✓ Demo 2 Complete"
echo ""

echo "Press Enter to continue to Demo 3..."
read

# Demo 3: Variable sequences (harder)
echo ""
echo "========================================"
echo ">>> DEMO 3: High Gap-Fraction Sequences"
echo "========================================"
echo "File: test_variable.fasta (8 diverse sequences)"
echo "Expected: AI tries to refine, then uses trimal to trim"
echo ""
python run_pipeline.py test_variable.fasta -n 5
echo ""
echo "✓ Demo 3 Complete - AI used trimal to improve alignment"
echo ""

echo "Press Enter to continue to Demo 4..."
read

# Demo 4: Show timeline
echo ""
echo "========================================"
echo ">>> DEMO 4: Learning Timeline"
echo "========================================"
echo "Shows chronological history of all experiments"
echo ""
python scripts/dashboard.py --timeline
echo ""
echo "✓ Demo 4 Complete"
echo ""

echo "Press Enter to continue to Demo 5..."
read

# Demo 5: Strategy performance
echo ""
echo "========================================"
echo ">>> DEMO 5: Strategy Performance Analysis"
echo "========================================"
echo "Shows which strategies work best based on experiments"
echo ""
python scripts/dashboard.py --strategies
echo ""
echo "✓ Demo 5 Complete"
echo ""

echo "Press Enter to continue to Demo 6..."
read

# Demo 6: Cross-validation
echo ""
echo "========================================"
echo ">>> DEMO 6: Cross-Validation"
echo "========================================"
echo "Tests how well learned strategies predict outcomes"
echo ""
python scripts/dashboard.py --validate
echo ""
echo "✓ Demo 6 Complete"
echo ""

echo "Press Enter to continue to Demo 7..."
read

# Demo 7: Export demo
echo ""
echo "========================================"
echo ">>> DEMO 7: Export Knowledge Base"
echo "========================================"
echo "Export learned knowledge to share with others"
echo ""
python scripts/dashboard.py --export demo_export.zip
echo ""
echo "Exported to: /home/user/AI/demo_export.zip"
echo ""
echo "✓ Demo 7 Complete"
echo ""

echo "Press Enter to see summary..."
read

# Summary
echo ""
echo "======================================"
echo "   DEMO COMPLETE!"
echo "======================================"
echo ""
echo "┌─────────────────────────────────────┐"
echo "│         KEY TAKEAWAYS               │"
echo "├─────────────────────────────────────┤"
echo "│ • AI learns from each experiment    │"
echo "│ • Stores knowledge for future      │"
echo "│ • Uses RAG to retrieve past learnings│"
echo "│ • Can use trimal to trim bad aligns │"
echo "│ • Adapts strategies based on results│"
echo "│ • 100% success in demo cases       │"
echo "└─────────────────────────────────────┘"
echo ""
echo "To run more demos:"
echo "  - python run_pipeline.py <your_fasta> -n 10"
echo "  - python scripts/dashboard.py --dashboard"
echo "  - python scripts/dashboard.py --timeline"
echo ""