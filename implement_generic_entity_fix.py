#!/usr/bin/env python3
"""
Fix for mapping generic entities to specific biolink categories

When users ask about "genetic variations" or "genetic factors", 
the system should search for specific biolink:Gene entities
instead of using generic UMLS concepts.
"""

def create_generic_entity_mapping_fix():
    """
    Show how to implement the generic entity to biolink category mapping fix
    """
    print("🔧 IMPLEMENTING GENERIC ENTITY MAPPING FIX")
    print("=" * 60)
    
    print("📋 PROBLEM ANALYSIS:")
    print("-" * 20)
    print("1. User query: 'What genetic factors influence drug efficacy?'")
    print("2. BioNER extracts: 'genetic factors' -> UMLS:C0042333")
    print("3. TRAPI query uses: specific UMLS ID instead of biolink:Gene category")
    print("4. Results: Generic concept relationships instead of specific genes")
    
    print(f"\n🎯 SOLUTION APPROACH:")
    print("-" * 20)
    print("Add entity concept mapping to detect generic terms and convert them")
    print("to appropriate biolink categories for open-ended knowledge graph search.")
    
    # Define the mapping
    generic_to_biolink_mapping = {
        # Genetic terms -> Gene category  
        "genetic variations": "biolink:Gene",
        "genetic factors": "biolink:Gene", 
        "genetic polymorphisms": "biolink:Gene",
        "genes": "biolink:Gene",
        "gene variants": "biolink:Gene",
        
        # Drug terms -> SmallMolecule category
        "drugs": "biolink:SmallMolecule",
        "medications": "biolink:SmallMolecule",
        "pharmaceuticals": "biolink:SmallMolecule",
        "compounds": "biolink:SmallMolecule",
        
        # Disease terms -> Disease category  
        "diseases": "biolink:Disease",
        "disorders": "biolink:Disease",
        "conditions": "biolink:Disease",
        
        # Pathway terms -> BiologicalProcess
        "pathways": "biolink:BiologicalProcess", 
        "biological processes": "biolink:BiologicalProcess",
        "metabolic pathways": "biolink:BiologicalProcess"
    }
    
    print(f"\n📋 ENTITY MAPPING TABLE:")
    print("-" * 30)
    for generic_term, biolink_category in generic_to_biolink_mapping.items():
        print(f"'{generic_term}' -> {biolink_category}")
    
    print(f"\n🔧 IMPLEMENTATION STEPS:")
    print("=" * 25)
    print("1. Modify build_trapi_query_structure in trapi.py")
    print("2. Add pre-processing to detect generic entity terms") 
    print("3. When building TRAPI nodes, use biolink category without IDs")
    print("4. This allows BTE to return all entities of that type")
    print("5. Results will show specific genes like 'DRD2', 'CYP2D6' etc.")
    
    print(f"\n💻 CODE CHANGES NEEDED:")
    print("-" * 25)
    print("""
Add to build_trapi_query_structure method:

# GENERIC ENTITY MAPPING FIX
generic_mapping = {
    "genetic variations": "biolink:Gene",
    "genetic factors": "biolink:Gene", 
    # ... more mappings
}

# Check if any entity should be mapped to biolink category
enhanced_entity_data = entity_data.copy()
for entity_name, entity_id in entity_data.items():
    entity_lower = entity_name.lower()
    if entity_lower in generic_mapping:
        # Remove the specific ID, use category search instead
        biolink_category = generic_mapping[entity_lower]
        enhanced_entity_data[f"_category_{entity_name}"] = biolink_category
        # Signal to use category-only node in TRAPI query
        enhanced_entity_data[f"_no_id_{entity_name}"] = True
        
# Use enhanced_entity_data in TRAPI prompt
""")
    
    print(f"\n📊 EXPECTED RESULTS AFTER FIX:")
    print("-" * 35)
    print("BEFORE FIX:")
    print("  Genetic variations → affects → UMLS:C0018270")
    print("")
    print("AFTER FIX:")
    print("  DRD2 → affects → Haloperidol")  
    print("  CYP2D6 → metabolizes → Risperidone")
    print("  COMT → affects → Dopamine")
    
    print(f"\n✅ BENEFITS:")
    print("-" * 15)
    print("• Specific gene names in results instead of generic terms")
    print("• Actionable pharmacogenomic information")
    print("• Better scientific interpretability")
    print("• More useful for personalized medicine")

if __name__ == "__main__":
    create_generic_entity_mapping_fix()