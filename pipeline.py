import os
from retrieval.hybrid_retriever import HybridRetriever
from gnn.subgraph_extractor import SubgraphExtractor
from generation.biomed_generator import BiomedGenerator
from generation.answer_validator import AnswerValidator
from generation.explainability import TraceabilityLogger
from personalization.patient_context import PatientContextProcessor
from personalization.context_integrator import ContextIntegrator
from config import DATA_PROCESSED_KG_DIR

class CardiologyLightRAG:
    def __init__(self):
        print("Initializing Cardiology LightRAG system...")
        self.retriever = HybridRetriever()
        self.subgraph_extractor = SubgraphExtractor()
        self.generator = BiomedGenerator()
        self.validator = AnswerValidator()
        self.context_processor = PatientContextProcessor()
        self.context_integrator = ContextIntegrator()
        self.trace_logger = TraceabilityLogger()
        self.kg = nx.read_gpickle(
            os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        )
    
    def process_query(self, query, patient_context=None):
        """Process a clinical query with optional patient context"""
        # Step 1: Process patient context
        parsed_context = None
        if patient_context:
            print("Processing patient context...")
            parsed_context = self.context_processor.parse_context(patient_context)
        
        # Step 2: Hybrid retrieval
        print("Performing hybrid retrieval...")
        vector_results, symbolic_results = self.retriever.hybrid_retrieve(query)
        self.trace_logger.log_retrieval(query, vector_results, symbolic_results)
        
        # Step 3: Personalized subgraph extraction
        print("Extracting knowledge subgraph...")
        if parsed_context:
            subgraph = self.context_integrator.integrate_patient_context(
                symbolic_results, parsed_context
            )
        else:
            subgraph = self.subgraph_extractor.extract_subgraph(symbolic_results)
        
        subgraph_text = self.subgraph_extractor.subgraph_to_text(subgraph)
        self.trace_logger.log_subgraph(subgraph)
        
        # Step 4: Generate context
        context = f"## Retrieved Medical Knowledge\n"
        context += f"**Relevant Documents:**\n"
        for i, doc in enumerate(vector_results[:2]):
            context += f"{i+1}. {doc[:200]}...\n\n"
        
        context += f"\n**Clinical Knowledge Graph:**\n{subgraph_text}"
        
        if patient_context:
            context += f"\n\n**Patient Context:** {patient_context}"
        
        # Step 5: Generate answer
        print("Generating clinical answer...")
        answer = self.generator.generate_answer(context, query)
        
        # Step 6: Validate answer
        print("Validating clinical accuracy...")
        is_valid, validation_msg = self.validator.validate_answer(context, answer)
        if not is_valid:
            answer = f"{answer}\n\n*Validation Note: {validation_msg}*"
        
        self.trace_logger.log_generation(context, answer)
        
        # Step 7: Generate explanation
        explanation = self.trace_logger.generate_explanation()
        
        return answer, explanation

# Example usage
if __name__ == "__main__":
    system = CardiologyLightRAG()
    
    # Example cardiology query
    query = "What are the first-line treatments for stable angina in diabetic patients?"
    patient_context = "65-year-old male with type 2 diabetes, hypertension, and aspirin allergy"
    
    print(f"\nQuestion: {query}")
    if patient_context:
        print(f"Patient Context: {patient_context}")
    
    answer, explanation = system.process_query(query, patient_context)
    
    print("\nClinical Answer:")
    print(answer)
    
    print("\nExplanation:")
    print(explanation)