MAFFT_STRATEGIES = {
    "--auto": "Automatically selects best method",
    "--localpair": "Local alignment (good for diverse sequences)",
    "--globalpair": "Global alignment (good for similar sequences)",
    "--genafpair": "Generalized affine gap penalty",
    "--retree 1": "Fast tree-based method"
}

MAFFT_COMMON_FLAGS = {
    "--maxiterate": "Max iterations, minimum 1000 enforced (e.g., --maxiterate 1000)",
    "--nuc": "Nucleotide sequence",
    "--amino": "Amino acid sequence",
    "--op": "Gap opening penalty (default: 1.53)",
    "--ep": "Gap extension penalty (default: 0.0)",
    "--bl": "BLOSUM matrix for amino (30,45,62,80)",
    "--jt": "Join tree topology",
    "--reorder": "Outorder: aligned, default: input order",
    "--treeout": "Output guide tree",
    "--quiet": "Do not report progress",
    "--thread": "Number of threads (use -1 for all)"
}