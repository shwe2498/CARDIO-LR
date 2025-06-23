import networkx as nx
from config import DATA_PROCESSED_KG_DIR

class TraceabilityLogger:
    def __init__(self):
        self.trace_data = {
            'query': '',
            'vector_retrieval': [],
            'symbolic_retrieval': [],
            'subgraph_nodes': [],
            'subgraph_edges': [],
            'generation_context': '',
            'answer': ''
        }
        self.integrated_kg = nx.read_gpickle(
            os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        )
    
    def log_retrieval(self, query, vector_results, symbolic_results):
        self.trace_data['query'] = query
        self.trace_data['vector_retrieval'] = vector_results[:3]  # Top 3
        self.trace_data['symbolic_retrieval'] = symbolic_results
    
    def log_subgraph(self, subgraph):
        self.trace_data['subgraph_nodes'] = [
            (node, self.integrated_kg.nodes[node]) 
            for node in subgraph.nodes()
        ]
        self.trace_data['subgraph_edges'] = [
            (u, v, self.integrated_kg.edges[u, v]) 
            for u, v in subgraph.edges()
        ]
    
    def log_generation(self, context, answer):
        self.trace_data['generation_context'] = context
        self.trace_data['answer'] = answer
    
    def generate_explanation(self):
        explanation = f"## Clinical Reasoning Report\n\n"
        explanation += f"**Question:** {self.trace_data['query']}\n\n"
        
        # Show evidence sources
        explanation += "### Evidence Sources\n"
        
        # Retrieved documents
        explanation += "**Relevant Medical Literature:**\n"
        for i, doc in enumerate(self.trace_data['vector_retrieval']):
            summary = doc.split('\n')[0][:150] + '...' if len(doc) > 150 else doc
            explanation += f"{i+1}. {summary}\n"
        
        # Knowledge graph concepts
        explanation += "\n**Medical Concepts Considered:**\n"
        unique_concepts = set()
        for node, data in self.trace_data['subgraph_nodes']:
            if 'name' in data and data['name'] not in unique_concepts:
                explanation += f"- {data['name']} ({data.get('type', 'Concept')})\n"
                unique_concepts.add(data['name'])
        
        # Clinical relationships
        explanation += "\n**Clinical Relationships Used:**\n"
        for u, v, data in self.trace_data['subgraph_edges'][:10]:  # Show top 10
            u_name = self.integrated_kg.nodes[u].get('name', u)
            v_name = self.integrated_kg.nodes[v].get('name', v)
            rel_type = data.get('relation', 'related_to')
            explanation += f"- {u_name} → **{rel_type}** → {v_name}\n"
        
        # Add source attribution
        explanation += "\n*This response was generated based on current medical literature and " \
                       "knowledge graphs including UMLS, SNOMED CT, and DrugBank.*"
        
        return explanation