import csv
import json
import requests
import random
import re
from typing import List, Dict, Any
from config import Config

class QAGenerator:
    def __init__(self):
        llm_config = Config.get_question_generation_config()
        self.model = llm_config["model"]
        self.base_url = llm_config["base_url"]
        self.temperature = llm_config["temperature"]
    
    def call_llm_api(self, prompt: str) -> str:
        api_url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return ""
    
    def parse_qa_response(self, generated_text: str) -> List[Dict[str, str]]:
        qa_pairs = []
        sections = re.split(r'\n\s*\d+\.|\n\s*\(\d+\)|\n\s*\d+\)', generated_text)
        
        if sections and not ('question:' in sections[0].lower() or 'answer:' in sections[0].lower()):
            sections = sections[1:]
        
        for section in sections:
            if not section.strip():
                continue
                
            parts = re.split(r'\n\s*Answer:', section, flags=re.IGNORECASE)
            
            if len(parts) < 2:
                parts = re.split(r'\n\s*A:', section, flags=re.IGNORECASE)
            
            if len(parts) < 2:
                qa_pairs.append({"question": section.strip(), "answer": ""})
                continue
                
            question = parts[0].replace("Question:", "", 1).replace("Q:", "", 1).strip()
            answer = parts[1].strip()
            
            qa_pairs.append({"question": question, "answer": answer})
        
        return qa_pairs
    
    def generate_factual_questions(self, abstract_text: str, accession: str) -> List[Dict[str, str]]:
        prompt = f"""Generate 2 factual questions that can be answered directly from the abstract below. Make questions specific and avoid vague references like "this study", "the study", "this research. do not add any text before or after question".

Abstract (Accession: {accession}): {abstract_text}

Format:
1. Question: [Specific factual question]
   Answer: [Exact text chunk from abstract]
2. Question: [Another specific factual question] 
   Answer: [Exact text chunk from abstract]

Do not include any introductory or concluding text."""

        generated_text = self.call_llm_api(prompt)
        return self.parse_qa_response(generated_text)
    
    def generate_analytical_questions(self, abstract_text: str, accession: str) -> List[Dict[str, str]]:
        prompt = f"""Generate 2 analytical questions that require interpretation of the abstract below. Make questions specific and avoid vague references.

Abstract (Accession: {accession}): {abstract_text}

Format:
1. Question: [Analytical question with specific context]
   Answer: [Relevant text chunk from abstract]
2. Question: [Another analytical question]
   Answer: [Relevant text chunk from abstract]

Do not include any introductory or concluding text."""

        generated_text = self.call_llm_api(prompt)
        return self.parse_qa_response(generated_text)
    
    def generate_comparative_questions(self, abstract_texts: List[str], accessions: List[str]) -> List[Dict[str, str]]:
        combined_context = "\n\n".join([f"Abstract {i+1} (Accession: {accessions[i]}): {text}" for i, text in enumerate(abstract_texts)])
        
        prompt = f"""Generate 2 comparison questions that require reading across the abstracts below. Use specific study references.

{combined_context}

Format:
1. Question: [Comparison question with specific context]
   Answer: [Combined relevant text chunks]
2. Question: [Another comparison question]
   Answer: [Combined relevant text chunks]

Do not include any introductory or concluding text."""

        generated_text = self.call_llm_api(prompt)
        return self.parse_qa_response(generated_text)
    
    def generate_unanswerable_questions(self, abstract_texts: List[str], accessions: List[str]) -> List[Dict[str, str]]:
        combined_context = "\n\n".join([f"Abstract {i+1} (Accession: {accessions[i]}): {text}" for i, text in enumerate(abstract_texts)])
        
        prompt = f"""Generate 2 plausible questions that CANNOT be answered from the abstracts below.

{combined_context}

Format:
1. Question: [Specific question that cannot be answered]
   Answer: Information not available in provided abstracts.
2. Question: [Another unanswerable question]
   Answer: Information not available in provided abstracts.

Do not include any introductory or concluding text."""

        generated_text = self.call_llm_api(prompt)
        return self.parse_qa_response(generated_text)
    
    def generate_from_file(self, input_file: str, output_file: str, num_questions: int = 400):
        from data_processor import DataProcessor
        
        processor = DataProcessor()
        abstracts = processor.load_abstracts_from_file(input_file)
        valid_abstracts = [abstract for abstract in abstracts if abstract['abstract'].strip()]
        
        print(f"Total valid abstracts: {len(valid_abstracts)}")
        
        all_questions = []
        question_counter = 1
        
        # question distribution
        question_types = [
            ("factual", Config.QUESTION_TYPES["factual"]),
            ("analytical", Config.QUESTION_TYPES["analytical"]), 
            ("comparative", Config.QUESTION_TYPES["comparative"]),
            ("unanswerable", Config.QUESTION_TYPES["unanswerable"])
        ]
        
        for question_type, count in question_types:
            print(f"Generating {question_type} questions...")
            
            iterations = count // 2
            
            for iteration in range(iterations):
                if question_type in ["factual", "analytical"]:
                    selected_abstract = random.choice(valid_abstracts)
                    
                    if question_type == "factual":
                        qa_pairs = self.generate_factual_questions(
                            selected_abstract['abstract'], 
                            selected_abstract['accession']
                        )
                    else:
                        qa_pairs = self.generate_analytical_questions(
                            selected_abstract['abstract'], 
                            selected_abstract['accession']
                        )
                    
                    context = selected_abstract['abstract']
                    
                else:
                    selected_abstracts = random.sample(valid_abstracts, min(3, len(valid_abstracts)))
                    abstract_texts = [abstract['abstract'] for abstract in selected_abstracts]
                    accessions = [abstract['accession'] for abstract in selected_abstracts]
                    
                    if question_type == "comparative":
                        qa_pairs = self.generate_comparative_questions(abstract_texts, accessions)
                    else:
                        qa_pairs = self.generate_unanswerable_questions(abstract_texts, accessions)
                    
                    context = "\n\n".join([f"Abstract {i+1}: {abstract_texts[i]}" for i in range(len(abstract_texts))])
                
                for qa_pair in qa_pairs:
                    # Extract primary accession for question ID
                    if question_type in ["factual", "analytical"]:
                        primary_accession = selected_abstract['accession']
                    else:
                        # For comparative and unanswerable, use the first accession
                        primary_accession = accessions[0]
                    
                    question_entry = {
                        "question_id": f"Q{question_counter:03d}_{primary_accession}",
                        "question": qa_pair["question"],
                        "context": context,
                        "expected_answer": qa_pair["answer"],
                        "question_type": question_type
                    }
                    all_questions.append(question_entry)
                    question_counter += 1
        
        dataset = {
            "total_questions": len(all_questions),
            "question_types": {
                "factual": len([q for q in all_questions if q["question_type"] == "factual"]),
                "analytical": len([q for q in all_questions if q["question_type"] == "analytical"]),
                "comparative": len([q for q in all_questions if q["question_type"] == "comparative"]),
                "unanswerable": len([q for q in all_questions if q["question_type"] == "unanswerable"])
            },
            "questions": all_questions
        }
        
        processor.save_dataset(dataset, output_file)
        print(f"Generated {len(all_questions)} questions saved to {output_file}")
        
        return dataset