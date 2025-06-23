import os
import networkx as nx
from config import DATA_PROCESSED_KG_DIR

class KnowledgeIntegrator:
    def __init__(self):
        self.umls_graph = None
        self.drugbank_graph = None
        self.snomed_graph = None
        self.integrated_graph = nx.DiGraph()
    
    def load_graphs(self):
        print("Loading knowledge graphs...")
        graphs_to_load = []
        
        # Load UMLS graph if available
        umls_path = os.path.join(DATA_PROCESSED_KG_DIR, 'umls_cardio_graph.pkl')
        if os.path.exists(umls_path):
            print("Loading UMLS graph...")
            self.umls_graph = nx.read_gpickle(umls_path)
            graphs_to_load.append(self.umls_graph)
            print(f"UMLS graph loaded with {len(self.umls_graph.nodes)} nodes and {len(self.umls_graph.edges)} edges")
        else:
            print("UMLS graph not found.")
            
        # Load DrugBank graph if available
        drugbank_path = os.path.join(DATA_PROCESSED_KG_DIR, 'drugbank_graph.pkl')
        if os.path.exists(drugbank_path):
            print("Loading DrugBank graph...")
            self.drugbank_graph = nx.read_gpickle(drugbank_path)
            graphs_to_load.append(self.drugbank_graph)
            print(f"DrugBank graph loaded with {len(self.drugbank_graph.nodes)} nodes and {len(self.drugbank_graph.edges)} edges")
        else:
            print("DrugBank graph not found.")
            
        # Load SNOMED graph if available
        snomed_path = os.path.join(DATA_PROCESSED_KG_DIR, 'snomed_graph_2025.pkl')
        if os.path.exists(snomed_path):
            print("Loading SNOMED graph...")
            self.snomed_graph = nx.read_gpickle(snomed_path)
            graphs_to_load.append(self.snomed_graph)
            print(f"SNOMED graph loaded with {len(self.snomed_graph.nodes)} nodes and {len(self.snomed_graph.edges)} edges")
        else:
            print("SNOMED graph not found.")
        
        return graphs_to_load
    
    def integrate_graphs(self):
        print("Integrating knowledge graphs...")
        
        # Get available graphs
        graphs_to_integrate = self.load_graphs()
        
        if not graphs_to_integrate:
            print("No knowledge graphs found to integrate. Exiting.")
            return None
            
        # Combine all available graphs
        if len(graphs_to_integrate) > 1:
            self.integrated_graph = nx.compose_all(graphs_to_integrate)
        else:
            self.integrated_graph = graphs_to_integrate[0].copy()
            print("Only one graph available, no integration needed.")
        
        # Add cross-references between ontologies
        print("Adding cross-ontology mappings...")
        added_edges = 0
        
        # Get nodes with valid names
        valid_nodes = {}
        for node, data in self.integrated_graph.nodes(data=True):
            if 'name' in data and data['name'] is not None:
                try:
                    # Convert name to string if it's not already
                    name_str = str(data['name']).lower()
                    valid_nodes[node] = name_str
                except (TypeError, ValueError):
                    # Skip nodes with problematic names
                    continue
        
        print(f"Found {len(valid_nodes)} nodes with valid names for cross-mapping")
        
        # Option to skip cross-mapping for large graphs
        if len(valid_nodes) > 1000000:
            print("Warning: Large number of nodes detected. Cross-mapping may be very time-consuming.")
            print("Skipping detailed cross-mapping due to performance considerations.")
            print("Saving integrated graph without detailed cross-mappings...")
            output_path = os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
            nx.write_gpickle(self.integrated_graph, output_path)
            print(f"Integrated graph saved with {len(self.integrated_graph.nodes)} nodes and {len(self.integrated_graph.edges)} edges")
            return self.integrated_graph
        
        # Process in smaller batches for better performance
        processed = 0
        batch_size = 1000  # Reduced batch size
        node_items = list(valid_nodes.items())
        total_nodes = len(node_items)
        
        for i in range(0, total_nodes, batch_size):
            batch = node_items[i:i+batch_size]
            for node, name in batch:
                # Find matching nodes across ontologies (more efficiently)
                matches = [
                    other_node for other_node, other_name in valid_nodes.items()
                    if node != other_node and name == other_name
                ]
                
                # Add equivalence relationships
                for match in matches:
                    if not self.integrated_graph.has_edge(node, match):
                        self.integrated_graph.add_edge(node, match, relation='same_as', source='cross_ontology')
                        self.integrated_graph.add_edge(match, node, relation='same_as', source='cross_ontology')
                        added_edges += 1
            
            processed += len(batch)
            print(f"Processed {processed}/{total_nodes} nodes, added {added_edges} cross-ontology mappings so far")
        
        print(f"Added {added_edges} cross-ontology mappings in total")
        
        # Save integrated graph
        output_path = os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        print(f"Saving integrated graph to {output_path}...")
        nx.write_gpickle(self.integrated_graph, output_path)
        print(f"Integrated graph saved with {len(self.integrated_graph.nodes)} nodes and {len(self.integrated_graph.edges)} edges")
        return self.integrated_graph

if __name__ == "__main__":
    integrator = KnowledgeIntegrator()
    integrator.integrate_graphs()