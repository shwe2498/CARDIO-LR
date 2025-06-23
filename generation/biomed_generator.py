# generation/biomed_generator.py
from transformers import BioGptTokenizer, BioGptForCausalLM

class BiomedGenerator:
    def __init__(self):
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    
    def generate_answer(self, context, question, max_length=200):
        input_text = f"Clinical Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input text from answer
        return answer.replace(input_text, "").strip()

if __name__ == "__main__":
    generator = BiomedGenerator()
    context = "Angina is chest pain caused by reduced blood flow to the heart."
    question = "What are the first-line treatments for angina?"
    print(generator.generate_answer(context, question))