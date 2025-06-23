import time

class MockCardiologyLightRAG:
    def __init__(self):
        print("Initializing Mock Cardiology LightRAG system...")
        # No complex dependencies initialized here
        time.sleep(2)  # Simulate loading time
        print("System ready!")
    
    def process_query(self, query, patient_context=None):
        """Process a clinical query with mocked responses"""
        print("Processing query:", query)
        if patient_context:
            print("Patient context:", patient_context)
        
        time.sleep(2)  # Simulate processing time
        
        # Sample answers based on common cardiology queries
        if "angina" in query.lower():
            answer = """First-line treatments for stable angina include:
1. Medical therapy:
   - Beta-blockers (e.g., metoprolol, atenolol)
   - Calcium channel blockers (e.g., amlodipine, diltiazem)
   - Nitrates (e.g., isosorbide mononitrate)
   - Antiplatelet therapy (e.g., aspirin)

2. Lifestyle modifications:
   - Regular physical activity
   - Smoking cessation
   - Weight management
   - Stress reduction

For patients with diabetes (as mentioned in context), special considerations include:
- ACE inhibitors may be preferred
- Careful monitoring of glucose levels when using beta-blockers
- Alternative antiplatelet therapy if aspirin allergy present"""
            
            explanation = """The system analyzed the query about stable angina treatments with consideration for the patient context of diabetes and hypertension.

Retrieved knowledge included:
- Clinical guidelines from ACC/AHA on stable angina management
- Research on beta-blocker usage in diabetic patients
- Drug interaction data between antihypertensives and antianginal medications
- Subgraph analysis of treatment pathways specific to cardiovascular conditions with metabolic comorbidities

The system identified diabetes as an important factor in treatment selection, noting that some beta-blockers may affect glucose control, and hypertension management should be coordinated with angina treatment."""

        elif "heart failure" in query.lower():
            answer = """Treatment for heart failure includes:
1. Medications:
   - ACE inhibitors or ARBs
   - Beta-blockers
   - Diuretics
   - Aldosterone antagonists
   - SGLT2 inhibitors (particularly beneficial in diabetic patients)

2. Lifestyle measures:
   - Sodium restriction
   - Fluid management
   - Regular monitoring of weight
   - Physical activity as tolerated"""
            
            explanation = """The system analyzed the query regarding heart failure treatment with consideration of the patient context.

The reasoning process included:
- Analysis of GDMT (Guideline-Directed Medical Therapy) for heart failure
- Evaluation of medication classes with proven mortality benefits
- Consideration of comorbid conditions mentioned in patient context
- Risk assessment for medication interactions"""
        
        else:
            answer = """Based on your query, I would recommend consulting with a cardiologist for personalized medical advice.

General cardiology care principles include:
1. Regular monitoring of blood pressure and heart rate
2. Medication adherence
3. Healthy lifestyle including diet and exercise
4. Regular follow-up with healthcare providers"""
            
            explanation = """The system processed your query but could not generate a specific clinical answer due to one of the following reasons:
- The query may be outside the knowledge domain of cardiology
- Insufficient information in the knowledge base
- The question may require personalized clinical judgment

The system prioritizes safety and accuracy in clinical information."""
        
        return answer, explanation

# Example usage
if __name__ == "__main__":
    system = MockCardiologyLightRAG()
    
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
