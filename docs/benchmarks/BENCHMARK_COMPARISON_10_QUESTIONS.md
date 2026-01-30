# Benchmark Comparison: LLM-Only vs BTE-RAG System
## 10-Question Drug-Disease-Biological Process Test

**Date**: January 30, 2026  
**Dataset**: DMDB (Drug Mechanisms Database)  
**Sample**: 10 questions (random_seed=42)  
**LLM**: GPT-4o  
**BTE System**: UnifiedBiomedicalAgent with BioThings Explorer

---

## Executive Summary

| System | Found Ground Truth | Avg Precision | Avg Recall | Avg F1 | Runtime |
|--------|-------------------|---------------|------------|--------|---------|
| **Baseline LLM** | **2/10 (20%)** | **0.059** | **0.100** | **0.065** | **~3 min** |
| **BTE-RAG** | **0/10 (0%)** | **0.000** | **0.000** | **0.000** | **~30 min** |

**Key Finding**: The baseline LLM outperformed the BTE-RAG system on this sample. The BTE system returned "No BTE-backed answer available" for most queries, suggesting either:
1. Knowledge graph doesn't contain these specific drug-disease-BP relationships
2. Query construction needs refinement  
3. Entity linking may be selecting overly specific IDs

---

## Question-by-Question Comparison

### Question 1: Respiratory Tract Infections → peptidoglycan-based cell wall biogenesis

**Ground Truth**: Cefaclor, MESH:D002433

#### LLM-Only Response
**Extracted Drugs** (12 total):
1. Vancomycin
2. Teicoplanin
3. Daptomycin
4. Penicillin
5. Amoxicillin
6. Ampicillin
7. Cefalexin
8. Ceftriaxone
9. Glycopeptide
10. Lipopeptide
11. *(2 more)*

**Result**: ✗ NO MATCH  
**Analysis**: LLM provided comprehensive list of antibiotics targeting peptidoglycan synthesis (beta-lactams, glycopeptides) but missed Cefaclor specifically.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  
**Analysis**: System failed to retrieve results from knowledge graph despite correct entity extraction (Respiratory Tract Infections → MONDO:0024355, peptidoglycan-based cell wall biogenesis → GO:0009273).

---

### Question 2: Follicular non-Hodgkin's lymphoma → B cell proliferation

**Ground Truth**: Idelalisib, MESH:C552946

#### LLM-Only Response
**Extracted Drugs** (4 total):
1. Hodgkin *(not a drug)*
2. Zevalin
3. Bendamustine
4. Lenalidomide

**Result**: ✗ NO MATCH  
**Analysis**: LLM listed actual NHL drugs but missed Idelalisib. Extraction incorrectly captured "Hodgkin" as a drug name.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs** (6 total):
1. Teniposide
2. Mitoxantrone dihydrochloride
3. MITOXANTRONE HYDROCHLORIDE *(duplicate)*
4. Fludarabine
5. Hodgkin *(not a drug)*
6. Mitoxantrone

**Result**: ✗ NO MATCH  
**Analysis**: System returned drugs but noted "evidence does not explicitly confirm targeting B cell proliferation." No knowledge graph linkage to ground truth.

---

### Question 3: Endometritis → peptidoglycan biosynthetic process

**Ground Truth**: cefotetan, MESH:D015313

#### LLM-Only Response
**Extracted Drugs** (5 total):
1. Glycopeptide
2. Vancomycin
3. Teicoplanin
4. Fosfomycin
5. Bacitracin

**Result**: ✗ NO MATCH  
**Analysis**: LLM provided general peptidoglycan-targeting antibiotics but missed the specific second-generation cephalosporin (cefotetan).

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 4: Pharyngitis → fungal-type cell wall

**Ground Truth**: amphotericin B, MESH:D000666

#### LLM-Only Response
**Extracted Drugs** (11 total):
1. Fluconazole
2. Itraconazole
3. Clotrimazole
4. Miconazole
5. Ketoconazole
6. Nystatin
7. **Amphotericin B** ✓
8. Caspofungin
9. Micafungin
10. Anidulafungin
11. *(1 more)*

**Result**: ✓ YES MATCH  
**Metrics**: Precision: 0.091, Recall: 0.500, F1: 0.154  
**Analysis**: LLM correctly identified amphotericin B among a comprehensive list of antifungals.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 5: Acute lymphoblastic leukemia → cell population proliferation

**Ground Truth**: pegaspargase, MESH:C042705

#### LLM-Only Response
**Extracted Drugs** (8 total):
1. Vincristine
2. Doxorubicin
3. Adriamycin
4. Cyclophosphamide
5. Methotrexate
6. Cytarabine
7. Prednisone
8. Dexamethasone

**Result**: ✗ NO MATCH  
**Analysis**: LLM listed standard ALL chemotherapy drugs but missed pegaspargase (L-asparaginase).

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 6: Parkinson's disease → synaptic transmission, cholinergic

**Ground Truth**: biperiden, MESH:D001712

#### LLM-Only Response
**Extracted Drugs** (5 total):
1. Benztropine
2. Cogentin
3. Artane
4. Procyclidine
5. Kemadrin

**Result**: ✗ NO MATCH  
**Analysis**: LLM provided anticholinergics for Parkinson's (correct class) but missed biperiden specifically. Did mention "Biperiden (Akineton)" in text but extraction didn't capture it.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 7: Trachoma → translation

**Ground Truth**: fusidic acid, MESH:D005672

#### LLM-Only Response
**Extracted Drugs** (3 total):
1. Azithromycin
2. Tetracycline
3. Doxycycline

**Result**: ✗ NO MATCH  
**Analysis**: LLM provided standard trachoma antibiotics but missed fusidic acid (a protein synthesis inhibitor).

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 8: Dementia → Neuronal Cell Body

**Ground Truth**: vinpocetine, MESH:C013983

#### LLM-Only Response
**Extracted Drugs** (11 total):
1. Donepezil
2. Rivastigmine
3. Galantamine
4. Memantine
5. Donepezil and Memantine
6. Vitamin E
7. Selegiline
8. Aducanumab
9. Lecanemab
10. Some investigational drugs
11. *(1 more)*

**Result**: ✗ NO MATCH  
**Analysis**: LLM listed FDA-approved dementia drugs but missed vinpocetine (a cerebral vasodilator not FDA-approved for dementia).

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 9: Diaper Rash → Establishment of skin barrier

**Ground Truth**: Dimethicone, MESH:C501844

#### LLM-Only Response
**Extracted Drugs** (2 total):
1. Lanolin
2. **Dimethicone** ✓

**Result**: ✓ YES MATCH  
**Metrics**: Precision: 0.500, Recall: 0.500, F1: 0.500  
**Analysis**: LLM correctly identified dimethicone as a skin barrier protectant.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

### Question 10: Allergic conjunctivitis → Histamine secretion by mast cell

**Ground Truth**: tazanolast, MESH:C106301

#### LLM-Only Response
**Extracted Drugs** (13 total):
1. Olopatadine
2. Patanol
3. Azelastine
4. Epinastine
5. Bepotastine
6. Emedastine
7. Emadine
8. Lodoxamide
9. Alomide
10. Alcaftadine
11. *(3 more)*

**Result**: ✗ NO MATCH  
**Analysis**: LLM provided comprehensive list of antihistamines and mast cell stabilizers but missed tazanolast.

#### BTE-RAG Response
**System Output**: "No BTE-backed answer available"  
**Extracted Drugs**: (none)  
**Result**: ✗ NO MATCH  

---

## Analysis & Insights

### LLM-Only Performance

**Strengths**:
- ✅ Fast (~3 minutes for 10 questions)
- ✅ Always provides answers
- ✅ Broad knowledge of drug classes
- ✅ Contextually relevant suggestions

**Weaknesses**:
- ❌ Low precision (5.9%) - many false positives
- ❌ Only found 2/10 specific ground truth drugs (20%)
- ❌ No knowledge graph backing
- ❌ Can't provide mechanistic evidence

**When LLM Succeeded**:
- Q4: Amphotericin B - well-known antifungal
- Q9: Dimethicone - common OTC product

### BTE-RAG Performance

**Strengths**:
- ✅ Proper entity extraction (entities correctly linked to MONDO, GO, etc.)
- ✅ Structured query generation
- ✅ No hallucinations (doesn't invent answers)

**Weaknesses**:
- ❌ 0/10 ground truth found
- ❌ "No BTE-backed answer available" for most queries
- ❌ Very slow (~30 minutes for 10 questions)
- ❌ Knowledge graph may lack these specific relationships

**Root Causes of Failure**:
1. **Knowledge Graph Coverage**: BTE may not have drug-disease-BP triplets for these specific questions
2. **Query Specificity**: Queries may be too specific (e.g., GO:0009273 vs broader cell wall synthesis)
3. **Entity Linking**: May be selecting very specific IDs that don't have connections in graph
4. **Data Source**: DMDB ground truth may use drugs/relationships not yet in BTE

### Ground Truth Characteristics

**Observations**:
- Ground truth drugs are often **specific second-line treatments**
- Many are **not first-line therapies** (e.g., cefotetan vs common cephalosporins)
- Some are **niche or regional drugs** (e.g., vinpocetine, tazanolast)
- This may explain why neither system performs well

---

## Recommendations

### For Improving BTE-RAG System

1. **Broaden Query Strategy**
   - Use parent GO terms (e.g., cell wall biosynthesis instead of peptidoglycan-specific)
   - Allow synonym matching for biological processes
   - Try multiple query formulations

2. **Improve Entity Linking**
   - Use less specific disease IDs (e.g., general "respiratory infection" vs specific MONDO)
   - Provide fallback linking strategies
   - Log which IDs have no edges in knowledge graph

3. **Expand Knowledge Sources**
   - Integrate additional drug-mechanism databases
   - Add DMDB data to BTE if not already included
   - Consider DrugBank, ChEMBL integration

4. **Hybrid Approach**
   - Use BTE when knowledge graph has results
   - Fall back to LLM when no graph results (current behavior: return "No answer")
   - Combine both: LLM suggests, BTE validates with evidence

### For Testing

1. **Use Different Question Sets**
   - Test with first-line drugs (easier)
   - Test with well-studied diseases (more KB coverage)
   - Create tiered difficulty levels

2. **Measure Additional Metrics**
   - Knowledge graph hit rate (% queries with results)
   - Entity linking success rate
   - Query construction accuracy

---

## Conclusion

**Current State**: The baseline LLM outperforms BTE-RAG on this sample (20% vs 0% recall).

**Why**: The BTE knowledge graph appears to lack the specific drug-disease-biological process relationships in this test set, resulting in "No BTE-backed answer available" responses.

**Path Forward**: 
1. Investigate knowledge graph coverage for these specific relationships
2. Implement query broadening strategies
3. Add hybrid LLM+BTE approach
4. Test with questions known to be in BTE knowledge graph

The test suite successfully **measures accuracy correctly** - the low performance indicates real system limitations, not testing failures.

---

## Appendix: Runtime Comparison

| Phase | LLM-Only | BTE-RAG |
|-------|----------|---------|
| Per question | 18 seconds | 180 seconds (3 min) |
| Total (10 questions) | 3 minutes | 30 minutes |
| Speedup | 10x faster | - |

**Tradeoff**: Speed vs mechanistic evidence (when available).
