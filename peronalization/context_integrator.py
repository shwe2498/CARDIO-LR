from gnn.subgraph_extractor import SubgraphExtractor

class ContextIntegrator:
    def __init__(self):
        self.subgraph_extractor = SubgraphExtractor()
    
    def integrate_patient_context(self, query_entities, patient_entities):
        """Integrate patient context into the query processing"""
        # Combine query and patient entities
        all_entities = query_entities.copy()
        
        # Add patient entities to the list
        for entity in patient_entities:
            kg_entity = entity['kg_entity']
            all_entities.append({
                'id': kg_entity['cui'],
                'name': kg_entity['name'],
                'type': kg_entity['type']
            })
        
        # Extract personalized subgraph
        personalized_subgraph = self.subgraph_extractor.extract_subgraph(all_entities)
        
        # Add patient-specific relationships
        self.add_patient_relationships(personalized_subgraph, patient_entities)
        
        return personalized_subgraph
    
    def add_patient_relationships(self, subgraph, patient_entities):
        """Add patient-specific relationships to the subgraph"""
        # This would be enhanced with medical logic in a real system
        # For demo: add "has_condition" and "taking_medication" relationships
        
        patient_node = "PATIENT_001"  # Represent the patient
        
        # Add patient node
        subgraph.add_node(patient_node, name="Current Patient", type="Patient")
        
        # Add relationships to conditions and medications
        for entity in patient_entities:
            kg_entity = entity['kg_entity']
            patient_entity = entity['patient_entity']
            
            if kg_entity['type'] == 'Condition':
                subgraph.add_edge(
                    patient_node, 
                    kg_entity['cui'], 
                    relation="has_condition",
                    source="patient_context"
                )
            elif kg_entity['type'] == 'Medication':
                subgraph.add_edge(
                    patient_node, 
                    kg_entity['cui'], 
                    relation="taking_medication",
                    source="patient_context"
                )
        
        return subgraph