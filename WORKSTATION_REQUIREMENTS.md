# System Requirements - AI-Powered MSA Pipeline Workstation

**Created:** August 10, 2026
**Purpose:** Lab workstation for running iterative MSA pipeline with ESM-2, Ollama LLM, and bioinformatics tools
**OS:** Windows 11 + WSL2 (Ubuntu 22.04)

---

## Pipeline Hardware Demands

| Component | What Runs | VRAM/RAM Needed | CPU Intensity |
|-----------|-----------|-----------------|---------------|
| ESM-2 650M | Protein embeddings + alignment | 2-4GB VRAM | Low (GPU-bound) |
| Ollama Llama3 8B | LLM controller decisions | 6-7GB VRAM (Q4_K_M) | Low (GPU-bound) |
| MAFFT/MUSCLE | Multiple sequence alignment | 2-4GB RAM | High (CPU-bound) |
| trimal | Alignment trimming | 1-2GB RAM | Low |
| Python pipeline | Orchestration, TOON, TF-IDF | 2-4GB RAM | Medium |
| **Total** | | **~8-11GB VRAM, ~8-12GB RAM** | |

---

## GPU Comparison - 40 Series vs 50 Series

### TSV Format (copy to spreadsheet)

```
GPU	VRAM	Memory Type	Bandwidth	TDP	Price (INR)	ESM-2 Speed	Llama3 8B Speed	Fits ESM-2 + Llama3?	Best For
RTX 4060	8GB	GDDR6	272 GB/s	115W	₹28,000-32,000	~2-3s/seq	~15-20 tok/s	Tight (8GB of 8GB)	Budget minimum
RTX 4060 Ti 8GB	8GB	GDDR6X	288 GB/s	160W	₹35,000-40,000	~2-3s/seq	~18-22 tok/s	Tight (8GB of 8GB)	Budget + better perf
RTX 4060 Ti 16GB	16GB	GDDR6X	288 GB/s	160W	₹42,000-48,000	~1-2s/seq	~20-25 tok/s	Comfortable (10-11GB of 16GB)	Best value 40-series
RTX 4070	12GB	GDDR6X	504 GB/s	200W	₹50,000-58,000	~1-2s/seq	~25-30 tok/s	OK (10-11GB of 12GB)	Mid-range 40-series
RTX 4070 Super	12GB	GDDR6X	504 GB/s	220W	₹55,000-65,000	~1-2s/seq	~28-35 tok/s	OK (10-11GB of 12GB)	Better mid-range
RTX 4070 Ti	12GB	GDDR6X	504 GB/s	285W	₹60,000-70,000	~1-2s/seq	~30-38 tok/s	OK (10-11GB of 12GB)	High-end 40-series
RTX 4070 Ti Super	16GB	GDDR6X	672 GB/s	285W	₹70,000-80,000	~0.5-1s/seq	~85-90 tok/s	Comfortable (10-11GB of 16GB)	Best 40-series for AI
RTX 4080	16GB	GDDR6X	717 GB/s	320W	₹90,000-1,00,000	~0.5-1s/seq	~90-100 tok/s	Comfortable (10-11GB of 16GB)	High-end but overpriced
RTX 4090	24GB	GDDR6X	1,008 GB/s	450W	₹1,60,000-1,80,000	~0.3-0.5s/seq	~120-140 tok/s	Very comfortable (10-11GB of 24GB)	Overkill for this pipeline
RTX 5070	12GB	GDDR7	672 GB/s	250W	₹58,000-68,000	~1-2s/seq	~80-90 tok/s	Tight (10-11GB of 12GB)	New gen, good value
RTX 5070 Ti	16GB	GDDR7	896 GB/s	300W	₹82,000-95,000	~0.5-1s/seq	~108 tok/s	Comfortable (10-11GB of 16GB)	Recommended for this pipeline
RTX 5080	16GB	GDDR7	960 GB/s	360W	₹1,10,000-1,30,000	~0.3-0.5s/seq	~120-130 tok/s	Comfortable (10-11GB of 16GB)	High-end 50-series
RTX 5090	32GB	GDDR7	1,792 GB/s	575W	₹2,50,000+	~0.2-0.3s/seq	~200+ tok/s	Very comfortable (10-11GB of 32GB)	Massive overkill
```

---

## Workstation Tiers

### Tier A: Budget (RTX 4060 Ti 16GB)
**Total: ₹95,000 - ₹1,10,000**

| Component | Spec | Price (₹) |
|-----------|------|-----------|
| CPU | AMD Ryzen 5 5600 (6C/12T) | 12,000 |
| Motherboard | B550M (Gigabyte/MSI) | 8,000 |
| RAM | 32GB DDR4 3200MHz (2x16GB) | 6,500 |
| GPU | RTX 4060 Ti 16GB (Zotac/Gigabyte) | 45,000 |
| Storage | 1TB NVMe SSD ( WD SN770) | 6,500 |
| PSU | 650W 80+ Gold (Corsair/Cooler Master) | 5,500 |
| Cabinet | ATX Mid Tower (Corsair 4000D) | 4,000 |
| Monitor | 24" FHD IPS (LG/Samsung) | 10,000 |
| KB+Mouse | Basic wired | 1,500 |
| **Total** | | **₹99,000** |

**Performance:**
- ESM-2: ~1-2s per sequence
- Llama3 8B Q4_K_M: ~20-25 tok/s
- Full pipeline iteration: ~45-90s
- VRAM: 16GB (comfortable for ESM-2 + Llama3)

---

### Tier B: Recommended (RTX 5070 Ti 16GB)
**Total: ₹1,40,000 - ₹1,60,000**

| Component | Spec | Price (₹) |
|-----------|------|-----------|
| CPU | AMD Ryzen 7 5700X (8C/16T) | 18,000 |
| Motherboard | B550M (Gigabyte/MSI) | 9,000 |
| RAM | 64GB DDR4 3200MHz (2x32GB) | 13,000 |
| GPU | RTX 5070 Ti 16GB (Zotac/Gigabyte) | 88,000 |
| Storage | 1TB NVMe SSD (WD SN770) | 6,500 |
| PSU | 750W 80+ Gold (Corsair RM750e) | 7,000 |
| Cabinet | ATX Mid Tower (Corsair 4000D) | 4,000 |
| Monitor | 27" QHD IPS (LG 27GP850) | 15,000 |
| KB+Mouse | Basic wired | 1,500 |
| **Total** | | **₹1,62,000** |

**Performance:**
- ESM-2: ~0.5-1s per sequence
- Llama3 8B Q4_K_M: ~108 tok/s
- Full pipeline iteration: ~20-40s
- VRAM: 16GB GDDR7 (comfortable headroom)

---

### Tier C: Ideal (RTX 5070 Ti + AM5 + DDR5)
**Total: ₹1,80,000 - ₹2,00,000**

| Component | Spec | Price (₹) |
|-----------|------|-----------|
| CPU | AMD Ryzen 7 7700X (8C/16T, AM5) | 25,000 |
| Motherboard | B650 (Gigabyte/MSI, AM5) | 16,000 |
| RAM | 64GB DDR5 5600MHz (2x32GB) | 16,000 |
| GPU | RTX 5070 Ti 16GB (Zotac/Gigabyte) | 88,000 |
| Storage | 2TB NVMe SSD (WD SN770) | 12,000 |
| PSU | 750W 80+ Gold (Corsair RM750e) | 7,000 |
| Cabinet | ATX Mid Tower (Corsair 4000D) | 5,000 |
| Monitor | 27" QHD IPS (LG 27GP850) | 15,000 |
| KB+Mouse | Basic wired | 1,500 |
| **Total** | | **₹1,85,500** |

**Performance:**
- ESM-2: ~0.5-1s per sequence
- Llama3 8B Q8_0 (full quality): ~100+ tok/s
- Full pipeline iteration: ~15-30s
- VRAM: 16GB GDDR7 + DDR5 bandwidth boost

---

### Tier D: Future-Proof (RTX 5090 32GB + Threadripper)
**Total: ₹4,50,000 - ₹5,50,000**

| Component | Spec | Price (₹) |
|-----------|------|-----------|
| CPU | AMD Threadripper 7960X (24C/48T, sTR5) | 1,20,000 |
| Motherboard | TRX50 (ASUS Pro WS) | 55,000 |
| RAM | 256GB DDR5 5200MHz ECC (4x64GB) | 80,000 |
| GPU | RTX 5090 32GB GDDR7 (MSI/ASUS) | 2,80,000 |
| Storage | 4TB NVMe SSD (Samsung 990 Pro) | 35,000 |
| PSU | 1200W 80+ Titanium (Corsair AX1200i) | 18,000 |
| Cabinet | Full Tower (Corsair 7000D) | 12,000 |
| Monitor | 32" 4K IPS (LG 32UK950) | 35,000 |
| KB+Mouse | Mechanical KB + Wireless Mouse | 5,000 |
| UPS | Online 3kVA (APC Smart-UPS) | 35,000 |
| **Total** | | **₹6,75,000** |

**Performance:**
- ESM-2: ~0.2-0.3s per sequence
- Llama3 70B Q4_K_M: ~40-50 tok/s (fits in 32GB VRAM)
- ProtT5-XL: Runs comfortably (16GB model + 16GB headroom)
- MAFFT/MUSCLE: ~2-3x faster (24 cores)
- Full pipeline iteration: ~5-15s
- VRAM: 32GB GDDR7 (massive headroom)

---

## Tier Comparison

| Metric | Tier A (₹99K) | Tier B (₹1.62L) | Tier C (₹1.86L) | Tier D (₹6.75L) |
|--------|---------------|-----------------|-----------------|-----------------|
| GPU | RTX 4060 Ti 16GB | RTX 5070 Ti 16GB | RTX 5070 Ti 16GB | RTX 5090 32GB |
| CPU | Ryzen 5 5600 | Ryzen 7 5700X | Ryzen 7 7700X | Threadripper 7960X |
| Cores | 6C/12T | 8C/16T | 8C/16T | 24C/48T |
| RAM | 32GB DDR4 | 64GB DDR4 | 64GB DDR5 | 256GB DDR5 ECC |
| Storage | 1TB | 1TB | 2TB | 4TB NVMe |
| ESM-2 speed | ~1-2s/seq | ~0.5-1s/seq | ~0.5-1s/seq | ~0.2-0.3s/seq |
| LLM speed | ~20-25 tok/s | ~108 tok/s | ~100+ tok/s | ~40-50 tok/s (70B) |
| Pipeline iter | ~45-90s | ~20-40s | ~15-30s | ~5-15s |
| VRAM headroom | Comfortable | Comfortable | Comfortable | Massive (32GB) |
| Future-proofing | Good | Better | Best (AM5) | Ultimate (5+ years) |
| Max model size | Llama3 8B | Llama3 8B | Llama3 8B | Llama3 70B, ProtT5-XL |

---

## OS Setup: Windows 11 + WSL2

### Why Windows + WSL2
- Lab members familiar with Windows
- WSL2 runs native Linux (Ubuntu 22.04) with full GPU support
- Ollama works natively on Windows (no WSL2 needed for LLM)
- Bioinformatics tools run in WSL2 Linux environment
- Best of both worlds

### Installation Steps

#### 1. Windows 11 Setup
- Install Windows 11 Pro (Hyper-V support required)
- Enable Hyper-V: `bcdedit /set hypervisorlaunchtype auto`
- Install NVIDIA GPU drivers (Studio Driver recommended for stability)
- Install Ollama for Windows: https://ollama.com/download
- Pull Llama3 model: `ollama pull llama3`

#### 2. WSL2 + Ubuntu Setup
```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04
# Restart computer
# Set up Ubuntu username/password
```

#### 3. WSL2 GPU Support
```bash
# In WSL2 Ubuntu terminal
# Install NVIDIA CUDA toolkit for WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

#### 4. Conda Environment Setup (in WSL2)
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart terminal

# Create AI environment
conda create -n AI python=3.11 -y
conda activate AI
pip install numpy scipy torch transformers scikit-learn

# Create seqTools environment
conda create -n seqTools -c bioconda -c conda-forge mafft muscle trimAl -y
```

#### 5. Pipeline Setup (in WSL2)
```bash
# Clone/copy pipeline to WSL2 filesystem (e.g., ~/AI/)
cd ~/AI
# Install remaining deps
conda activate AI
pip install -r requirements.txt  # if exists

# Test pipeline
python run_pipeline.py --help
```

### Performance Notes
- WSL2 GPU overhead: ~3-5% (negligible for this workload)
- File I/O: Keep pipeline files on WSL2 filesystem (`/home/`), NOT Windows mount (`/mnt/c/`)
- Ollama: Can run on Windows side, WSL2 connects via `http://localhost:11434`
- Memory: WSL2 shares RAM with Windows, allocate via `.wslconfig`

### WSL2 Memory Allocation
Create `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
memory=32GB
swap=8GB
processors=8
```

---

## Recommended Workstation (Tier B)

**Final configuration for lab purchase:**

| Component | Model | Price (₹) | Vendor |
|-----------|-------|-----------|--------|
| CPU | AMD Ryzen 7 5700X | 18,000 | MDComputers, PrimeABGB |
| Motherboard | Gigabyte B550M DS3H | 9,000 | MDComputers, Vedant |
| RAM | Corsair Vengeance LPX 64GB DDR4 3200MHz | 13,000 | Amazon, MDComputers |
| GPU | Zotac RTX 5070 Ti 16GB Trinity | 88,000 | MDComputers, PrimeABGB |
| Storage | WD Black SN770 1TB NVMe | 6,500 | Amazon, Vedant |
| PSU | Corsair RM750e 750W 80+ Gold | 7,000 | MDComputers |
| Cabinet | Corsair 4000D Airflow | 4,000 | Amazon, Vedant |
| Monitor | LG 27GP850-B 27" QHD IPS | 15,000 | Amazon |
| KB+Mouse | Logitech MK270 Wireless | 1,500 | Amazon |
| **Total** | | **₹1,62,000** | |

### Where to Buy (India)
- **MDComputers** (mdcomputers.in) - Best GPU prices, reliable
- **PrimeABGB** (primeabgb.com) - Good stock, reliable
- **Vedant Computers** (vedantcomputers.com) - Competitive pricing
- **Amazon India** - Convenience, easy returns
- **Flipkart** - Occasionally good deals

### Warranty Notes
- GPU: 3 years (AIB partner warranty)
- CPU: 3 years (AMD)
- Motherboard: 3 years
- RAM: Lifetime (Corsair)
- SSD: 5 years (WD)
- PSU: 7-10 years (Corsair)

---

## Quick Reference

### Minimum Requirements (Tier A)
- GPU: 16GB VRAM (RTX 4060 Ti 16GB)
- RAM: 32GB DDR4
- Storage: 1TB NVMe SSD
- CPU: 6 cores
- Budget: ~₹1,00,000

### Recommended Requirements (Tier B)
- GPU: 16GB VRAM (RTX 5070 Ti)
- RAM: 64GB DDR4
- Storage: 1TB NVMe SSD
- CPU: 8 cores
- Budget: ~₹1,60,000

### Ideal Requirements (Tier C)
- GPU: 16GB+ VRAM (RTX 5070 Ti)
- RAM: 64GB DDR5
- Storage: 2TB NVMe SSD
- CPU: 8+ cores (AM5 platform)
- Budget: ~₹1,85,000

### Future-Proof Requirements (Tier D)
- GPU: 32GB VRAM (RTX 5090)
- RAM: 256GB DDR5 ECC
- Storage: 4TB NVMe SSD
- CPU: 24+ cores (Threadripper)
- Budget: ~₹6,75,000
- Use case: Large models (70B LLM, ProtT5-XL, ESM-2 3B), multi-user, 5+ year lifespan

---

## Future-Proofing Considerations

### What "Future-Proof" Means for This Pipeline

| Timeline | Expected Change | Hardware Impact |
|----------|-----------------|-----------------|
| 2026 Q4 | ESM-2 3B model integration | Needs 8GB+ VRAM for embeddings |
| 2027 Q1 | ProtT5-XL for alignment scoring | Needs 16GB+ VRAM |
| 2027 Q2 | Llama3 70B for better decisions | Needs 32GB+ VRAM (Q4 quantized) |
| 2027 H2 | Multi-sequence ESM-2 batching | Needs 24GB+ VRAM |
| 2028+ | Next-gen pLMs (ESM-3, ProtT5 v2) | Likely 32GB+ VRAM |

### Upgrade Paths

| Component | Current (Tier B) | Upgrade Option | Cost |
|-----------|------------------|----------------|------|
| GPU | RTX 5070 Ti 16GB | RTX 5090 32GB (when price drops) | ~₹1,50,000 |
| RAM | 64GB DDR4 | 128GB DDR4 | ~₹15,000 |
| Storage | 1TB NVMe | 2TB + 2TB RAID | ~₹15,000 |
| CPU | Ryzen 7 5700X | Ryzen 9 5950X (same socket) | ~₹25,000 |

### Multi-GPU Consideration

For labs running multiple simultaneous workloads:

| Setup | GPU Config | Use Case | Price |
|-------|------------|----------|-------|
| Single GPU | 1x RTX 5070 Ti | Single user, single pipeline | ~₹88,000 |
| Dual GPU | 2x RTX 5070 Ti | Two pipelines OR ESM-2 + LLM separate | ~₹1,76,000 |
| Mixed GPU | RTX 5090 + RTX 5070 Ti | Primary heavy work + secondary light work | ~₹3,68,000 |

**Note:** Multi-GPU requires:
- Motherboard with 2x PCIe x16 slots
- PSU 1000W+ for dual GPU
- Cabinet with good airflow
- Ollama can run on one GPU, ESM-2 on the other (separate CUDA devices)
