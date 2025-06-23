import json
import pandas as pd
from .metrics import evaluate_answer
from config import BIOASQ_PATH, MEDQUAD_PATH, DATA_PROCESSED_KG_DIR
from pipeline import CardiologyLightRAG

class Evaluator:
    def __init__(self):
        self.system = CardiologyLightRAG()
        self.kg = nx.read_gpickle(
            os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
        )
    
    def load_bioasq_data(self):
        """Load BioASQ dataset"""
        with open(BIOASQ_PATH) as f:
            data = json.load(f)
        return data['questions']
    
    def load_medquad_cardio(self):
        """Load cardiology subset from MedQuAD"""
        df = pd.read_csv(MEDQUAD_PATH)
        cardio_df = df[df['topic'] == 'Heart Diseases']
        return [
            {'question': row['question'], 'answer': row['answer']}
            for _, row in cardio_df.iterrows()
        ]
    
    def evaluate(self, dataset_name, patient_context=None):
        """Evaluate system on a specific dataset"""
        if dataset_name == 'bioasq':
            data = self.load_bioasq_data()
        elif dataset_name == 'medquad':
            data = self.load_medquad_cardio()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        results = []
        for item in data:
            question = item['body'] if 'body' in item else item['question']
            gold_answer = item['exact_answer'] if 'exact_answer' in item else item['answer']
            
            # Handle different answer formats
            if isinstance(gold_answer, list):
                gold_answer = " ".join(gold_answer)
            
            # Get system prediction
            answer, _ = self.system.process_query(question, patient_context)
            
            # Evaluate
            metrics = evaluate_answer(answer, gold_answer)
            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'pred_answer': answer,
                **metrics
            })
        
        return results
    
    def calculate_knowledge_coverage(self, results):
        """Calculate how well the system utilizes knowledge graph"""
        coverage_scores = []
        for result in results:
            pred_entities = set()
            gold_entities = set()
            
            # Extract entities from answers (simplified)
            for node in self.kg.nodes(data=True):
                if 'name' in node[1] and node[1]['name'] in result['pred_answer']:
                    pred_entities.add(node[1]['name'])
                if 'name' in node[1] and node[1]['name'] in result['gold_answer']:
                    gold_entities.add(node[1]['name'])
            
            # Calculate coverage score
            if len(gold_entities) > 0:
                coverage = len(pred_entities & gold_entities) / len(gold_entities)
                coverage_scores.append(coverage)
        
        return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
    
    def run_full_evaluation(self):
        """Run comprehensive evaluation"""
        print("Evaluating on BioASQ dataset...")
        bioasq_results = self.evaluate('bioasq')
        
        print("Evaluating on MedQuAD Cardiology subset...")
        medquad_results = self.evaluate('medquad')
        
        # Calculate overall metrics
        metrics = {
            'bioasq': self.calculate_metrics(bioasq_results),
            'medquad': self.calculate_metrics(medquad_results)
        }
        
        # Add knowledge coverage
        metrics['bioasq']['knowledge_coverage'] = self.calculate_knowledge_coverage(bioasq_results)
        metrics['medquad']['knowledge_coverage'] = self.calculate_knowledge_coverage(medquad_results)
        
        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump({
                'metrics': metrics,
                'bioasq_details': bioasq_results,
                'medquad_details': medquad_results
            }, f, indent=2)
        
        return metrics
    
    def calculate_metrics(self, results):
        """Calculate average metrics from results"""
        return {
            'em': sum(r['em'] for r in results) / len(results),
            'f1': sum(r['f1'] for r in results) / len(results),
            'rouge': sum(r['rouge'] for r in results) / len(results)
        }

if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.run_full_evaluation()
    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2))