from rouge_score import rouge_scorer
import numpy as np

def calculate_em(pred, gold):
    """Calculate Exact Match (case-insensitive)"""
    return int(pred.strip().lower() == gold.strip().lower())

def calculate_f1(pred, gold):
    """Calculate token-level F1 score"""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(gold_tokens)
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1

def calculate_rouge(pred, gold):
    """Calculate ROUGE-L score"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gold, pred)
    return scores['rougeL'].fmeasure

def calculate_medical_accuracy(pred, gold):
    """Medical-specific accuracy metric (placeholder)"""
    # In real implementation, this would use clinical NLP models
    # For now, use keyword matching for demo
    medical_keywords = ['heart', 'cardio', 'angina', 'stroke', 'attack', 
                        'blood pressure', 'cholesterol', 'aortic', 'valve']
    
    pred_score = sum(1 for kw in medical_keywords if kw in pred.lower())
    gold_score = sum(1 for kw in medical_keywords if kw in gold.lower())
    
    if gold_score == 0:
        return 0.0
    
    return min(pred_score / gold_score, 1.0)

def evaluate_answer(pred, gold):
    """Comprehensive evaluation of answer quality"""
    return {
        'em': calculate_em(pred, gold),
        'f1': calculate_f1(pred, gold),
        'rouge': calculate_rouge(pred, gold),
        'medical_acc': calculate_medical_accuracy(pred, gold)
    }