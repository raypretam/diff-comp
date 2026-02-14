"""
Visualization of the difference between standard and graph-aware compound encoding.

This example shows how dependency information is encoded in a simple nested compound.
"""

# Example Sentence:
# रामः सीतायाः पुस्तकालयस्य गच्छति
# Rama goes to Sita's library
#
# Compound structure:
#   सीतायाः-पुस्तकालयस्य (Sita's library - possessive compound)
#   └─ सीतायाः (Sita's)
#   └─ पुस्तकालयस्य (library)

# Token indices and labels:
# Token 0: रामः        → Label: "ROOT:root"          (sentence root, not a compound)
# Token 1: सीतायाः     → Label: "+1:Tatpurusha"     (head is token 2, possessive relation)
# Token 2: पुस्तकालयस्य → Label: "ROOT:Comp_root"    (compound root)
# Token 3: गच्छति      → Label: "ROOT:root"          (verb, sentence root)

# ============================================================================
# STANDARD COMPOUND-AWARE (ORIGINAL - what you tried)
# ============================================================================

"""
Step 1: Identify compounds from labels
  Compound A: tokens [1, 2] (सीतायाः पुस्तकालयस्य)
  
Step 2: Pool tokens within each compound
  compound_A_features = mean(token_1_features, token_2_features)
  
Step 3: Run diffusion on compound features
  INDEPENDENTLY - no connection between compounds
  
Step 4: Broadcast predictions back to tokens
  token_1_prediction = compound_A_prediction
  token_2_prediction = compound_A_prediction
  
RESULT:
  ✅ All tokens in compound A get same label (internal consistency)
  ❌ No understanding that token 1 depends on token 2
  ❌ No learning of Tatpurusha relation pattern
  ❌ Cannot enforce structural constraints across tokens
"""

# ============================================================================
# GRAPH-AWARE COMPOUND ENCODER (NEW - what I implemented)
# ============================================================================

"""
Step 1: Identify compounds (same as before)
  Compound A: tokens [1, 2]
  
Step 2: Pool tokens within compounds (same as before)
  compound_A_features = mean(token_1_features, token_2_features)
  
Step 3: **NEW** - Extract dependency structure from labels
  Parse label "+1:Tatpurusha" for token 1:
    - Distance: +1 means head is at position 1+1=2
    - Token 1 belongs to compound A
    - Token 2 belongs to compound A (same compound in this case)
    - **But in general, could be different compounds!**
    - Relation type: "Tatpurusha"
  
  Build dependency graph:
    In this case, it's within the same compound, but consider:
    
    Example with nested compounds:
      Compound B (outer): tokens [1, 2, 3]
      Compound A (inner): tokens [1, 2]
      
      Graph: Compound B → Compound A (Compound A is nested in B)
      
  Result: Adjacency matrix [num_compounds, num_compounds]
          Relation types [num_compounds, num_compounds]
  
Step 4: **NEW** - Graph Neural Network message passing
  For num_gnn_layers iterations:
    For each compound:
      # Find parent compounds (incoming edges)
      parents = get_parents(compound, adjacency_matrix)
      
      messages = []
      for parent in parents:
        # Get relation embedding (learned!)
        relation_type = relation_types[parent, compound]
        rel_emb = relation_embeddings[relation_type]  # e.g., "Tatpurusha" embedding
        
        # Compute message: parent features + relation type
        message = message_network([parent_features, rel_emb])
        
        # Attention weight (how important is this parent?)
        attention = attention_network([compound_features, parent_features])
        
        messages.append((message, attention))
      
      # Aggregate messages
      aggregated = weighted_sum(messages)  # using attention weights
      
      # Update compound features
      compound_features = update_network([compound_features, aggregated])
  
  RESULT: Compound features now encode:
    ✅ Information from parent compounds
    ✅ Relation-specific patterns (Tatpurusha vs Dvandva differ)
    ✅ Hierarchical structure (nested compound information)
    
Step 5: Run diffusion on graph-enhanced compound features
  Now the diffusion process operates on richer representations!
  
Step 6: Broadcast predictions (same as before)
  token_predictions = compound_predictions
  
FINAL RESULT:
  ✅ Internal compound consistency (same as before)
  ✅ **NEW**: Inter-compound dependency structure learned
  ✅ **NEW**: Relation-specific patterns learned
  ✅ **NEW**: Hierarchical constraints enforced
"""

# ============================================================================
# CONCRETE EXAMPLE: Why Graph Encoding Matters
# ============================================================================

"""
Consider this nested compound:
  राम-सीता-विवाह-समारोह
  Rama-Sita-wedding-ceremony
  
Structure:
  Level 1: राम-सीता (Rama-Sita) - Dvandva (coordinative)
  Level 2: [राम-सीता]-विवाह (their wedding) - Tatpurusha (possessive)
  Level 3: [[राम-सीता]-विवाह]-समारोह (wedding ceremony) - Karmadhara (appositional)

Token labels:
  राम     (0): "+1:Dvandva"         (head: token 1)
  सीता    (1): "+1:Tatpurusha"      (head: token 2)
  विवाह   (2): "+1:Karmadhara"      (head: token 3)
  समारोह  (3): "ROOT:Comp_root"     (root of compound)

Compounds identified:
  Compound A: tokens [0] (राम)
  Compound B: tokens [1] (सीता)  
  Compound C: tokens [2] (विवाह)
  Compound D: tokens [3] (समारोह)

Dependency graph:
  B → A (Dvandva)
  C → B (Tatpurusha)  
  D → C (Karmadhara)
  
WITHOUT graph encoding:
  ❌ Each compound processed independently
  ❌ Model doesn't know B depends on A
  ❌ Cannot learn that Dvandva is different from Tatpurusha
  ❌ Cannot enforce that if B has label X, A must be compatible
  
WITH graph encoding:
  ✅ Compound B receives message from A with Dvandva embedding
  ✅ Compound C receives message from B with Tatpurusha embedding
  ✅ Model learns different patterns for different relation types
  ✅ Hierarchical information flows: D → C → B → A
  ✅ Structural constraints: if C predicts Tatpurusha, B and A features influence this
"""

# ============================================================================
# IMPACT ON EXACT MATCH
# ============================================================================

"""
Original compound-aware (no graph encoding):
  Scenario: Model predicts 3 out of 4 compounds correctly
  Problem: The one wrong compound breaks EM for the entire sequence
  Why: No structural constraints, so errors are independent
  
  Example error:
    Predicted:  राम (+1:Dvandva)  ✓
                सीता (+1:Bahuvrihi)  ✗ (should be Tatpurusha)
                विवाह (+1:Karmadhara)  ✓
                समारोह (ROOT:Comp_root)  ✓
    
    EM = 0 (one error breaks everything)

Graph-aware compound encoder:
  Advantage: Structural consistency
  
  Example:
    - Model learns: "If B points to A with Dvandva, both should be similar status"
    - Model learns: "If C points to B with Tatpurusha, C is possessed by B"
    - GNN propagates this information
    
  Result: Errors are correlated (less likely to make inconsistent predictions)
  
  If model makes error, it's more likely to be:
    - Entire subtree wrong (still EM=0)
    - BUT: Fewer isolated errors
    
  More importantly: FEWER ERRORS overall because:
    - Relation embeddings provide strong signal
    - Parent compound features guide child predictions
    - Hierarchical constraints reduce search space
"""

print(__doc__)
