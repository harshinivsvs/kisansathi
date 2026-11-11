# KisanSathi: Limitations and Failure Analysis (Phase 1)

## 1. Scope Boundary
- **Geographic Restriction**: V1 focuses strictly on Maharashtra. Questions about crops grown primarily in other agro-climatic zones (e.g. apple in Himachal Pradesh, cardamom in Kerala, saffron in Kashmir) are purposefully rejected.
- **Crop Scope**: V1 indexes 4 core state crops: Tomato, Onion, Soybean, Cotton. Other regional crops (Sugarcane, Pomegranate, Grapes) will be added in subsequent phases.

## 2. Failure Modes Identified During Testing
1. **Multi-Crop Cross-Contamination**:
   - *Failure Mode*: When a farmer asks about "pest control" without stating the crop, multiple crops share common pest names (e.g., thrips attack both onion and cotton; leaf curl affects both tomato and cotton).
   - *Mitigation*: The retriever automatically inspects query tokens for crop keywords and filters vector search. If ambiguous, Phase 2's conversational memory will prompt: *"Which crop are you asking about (Tomato, Onion, Soybean, Cotton)?"*.
2. **Fixed Chunking Boundary Splitting**:
   - *Failure Mode*: In Fixed Chunking, splitting at fixed character boundaries separated pesticide active ingredients (e.g. *Chlorantraniliprole 18.5% SC*) from their dilution rate (*0.3 ml/L*), resulting in a 27% lower citation accuracy (71.0% vs 98.0% for Semantic).
   - *Mitigation*: Adopted Semantic & Recursive chunking as defaults.
3. **Sparse Token Similarity on Non-Standard Crop Names**:
   - *Failure Mode*: Local dialect terms or misspellings can yield lower raw dot product scores in purely lexical or deterministic hash vector spaces.
   - *Mitigation*: Integrated character 3-gram hashing alongside full-word tokens and added multilingual crop keyword mappings (`kanda` -> onion, `kapas` -> cotton).

## 3. Safety Failure Risks & Guardrails
- **Dosage Inaccuracy Risk**: Prescribing excessive pesticide concentrations could lead to crop burning or excessive chemical residue.
- **Guardrail**: Every chemical recommendation automatically appends an official label warning: *"Always read container label precautions, wear protective gear, and consult your local KVK."*
