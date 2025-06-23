from transformers import pipeline
import torch

class AnswerValidator:
    def __init__(self):
        self.contradiction_detector = pipeline(
            "text-classification", 
            model="morit/english_xlmr_mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.medical_fact_checker = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def check_contradiction(self, context, answer):
        """Check if answer contradicts the context"""
        result = self.contradiction_detector(
            f"{context} [SEP] {answer}",
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        return result[0]['label'] == 'contradiction'
    
    def verify_medical_fact(self, statement):
        """Verify a medical fact using knowledge-intensive approach"""
        response = self.medical_fact_checker(
            f"Verify this medical statement: {statement}",
            max_length=100
        )
        verification = response[0]['generated_text'].lower()
        return "true" in verification or "correct" in verification or "accurate" in verification
    
    def validate_answer(self, context, answer):
        """Comprehensive validation of clinical answer"""
        # Split answer into individual statements
        statements = [s.strip() for s in answer.split('.') if s.strip()]
        
        # Check for contradictions
        if self.check_contradiction(context, answer):
            return False, "Answer contradicts source context"
        
        # Verify key medical facts
        for statement in statements[:3]:  # Check first 3 statements
            if not self.verify_medical_fact(statement):
                return False, f"Unverified medical claim: {statement}"
        
        return True, "Answer validated"