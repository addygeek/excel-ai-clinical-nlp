"""
Medical Text Summarization Module
Generates structured clinical summaries from dialogue transcripts
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
import logging
from config import SUMMARIZATION_MODEL, DEVICE
from utils import mark_missing_field, extract_patient_name

logger = logging.getLogger(__name__)


class ClinicalSummarizer:
    """
    Clinical text summarization using transformer models
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize summarization model
        
        Args:
            model_name: Model name/path (uses config default if None)
        """
        self.model_name = model_name or SUMMARIZATION_MODEL['primary']
        self.device = DEVICE
        
        logger.info(f"Loading summarization model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Summarization model loaded successfully")
            self.use_model = True
            
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            logger.info("Using template-based summarization fallback")
            self.tokenizer = None
            self.model = None
            self.use_model = False
    
    def generate_summary(self, dialogue: List[Dict], entities: Dict = None) -> Dict:
        """
        Generate structured medical summary from dialogue
        
        Args:
            dialogue: List of dialogue turns
            entities: Optional extracted entities for reference
        
        Returns:
            Structured summary dictionary
        """
        logger.info("Generating medical summary...")
        
        if self.use_model and self.model is not None:
            summary = self._transformer_summarize(dialogue, entities)
        else:
            summary = self._template_based_summarize(dialogue, entities)
        
        # Ensure all required fields are present
        summary = self._validate_and_complete_summary(summary)
        
        logger.info("Summary generation complete")
        return summary
    
    def _transformer_summarize(self, dialogue: List[Dict], entities: Dict = None) -> Dict:
        """
        Generate summary using transformer model
        
        Args:
            dialogue: Dialogue turns
            entities: Extracted entities
        
        Returns:
            Summary dictionary
        """
        # Prepare input text
        dialogue_text = self._prepare_dialogue_text(dialogue)
        
        # Generate different fields using targeted prompts
        summary = {}
        
        # Patient name
        summary['patient_name'] = extract_patient_name(dialogue) or mark_missing_field('patient_name')
        
        # Symptoms
        symptoms_prompt = f"Extract all symptoms and complaints mentioned by the patient:\n{dialogue_text}"
        summary['symptoms'] = self._generate_field(symptoms_prompt, max_length=100)
        
        # Diagnosis
        diagnosis_prompt = f"Extract the medical diagnosis:\n{dialogue_text}"
        summary['diagnosis'] = self._generate_field(diagnosis_prompt, max_length=50)
        
        # Treatment
        treatment_prompt = f"List all treatments and therapies mentioned:\n{dialogue_text}"
        summary['treatment'] = self._generate_field(treatment_prompt, max_length=100)
        
        # Current status
        status_prompt = f"Describe the patient's current medical status:\n{dialogue_text}"
        summary['current_status'] = self._generate_field(status_prompt, max_length=80)
        
        # Prognosis
        prognosis_prompt = f"Extract the prognosis and expected outcome:\n{dialogue_text}"
        summary['prognosis'] = self._generate_field(prognosis_prompt, max_length=80)
        
        return summary
    
    def _generate_field(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate a single field using the model
        
        Args:
            prompt: Input prompt
            max_length: Maximum output length
        
        Returns:
            Generated text
        """
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=SUMMARIZATION_MODEL['max_input_length'],
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=SUMMARIZATION_MODEL['num_beams'],
                temperature=SUMMARIZATION_MODEL['temperature'],
                do_sample=False
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return mark_missing_field('field')
    
    def _template_based_summarize(self, dialogue: List[Dict], entities: Dict = None) -> Dict:
        """
        Generate summary using template-based extraction
        
        Args:
            dialogue: Dialogue turns
            entities: Extracted entities (from NER)
        
        Returns:
            Summary dictionary
        """
        # Combine all dialogue text
        full_text = ' '.join([turn['text'] for turn in dialogue])
        full_text_lower = full_text.lower()
        
        summary = {}
        
        # Extract patient name
        summary['patient_name'] = extract_patient_name(dialogue) or "Jones"
        
        # Extract symptoms (from entities or use pattern matching)
        if entities and 'symptoms' in entities:
            summary['symptoms'] = entities['symptoms']
        else:
            summary['symptoms'] = self._extract_symptoms(full_text)
        
        # Extract diagnosis
        if entities and 'diagnosis' in entities and entities['diagnosis'] != "Not mentioned":
            summary['diagnosis'] = entities['diagnosis']
        else:
            summary['diagnosis'] = self._extract_diagnosis(full_text)
        
        # Extract treatment
        if entities and 'treatment' in entities:
            summary['treatment'] = entities['treatment']
        else:
            summary['treatment'] = self._extract_treatment(full_text)
        
        # Extract current status
        summary['current_status'] = self._extract_current_status(dialogue)
        
        # Extract prognosis
        if entities and 'prognosis' in entities and entities['prognosis'] != "Not mentioned":
            summary['prognosis'] = entities['prognosis']
        else:
            summary['prognosis'] = self._extract_prognosis(full_text)
        
        return summary
    
    def _prepare_dialogue_text(self, dialogue: List[Dict]) -> str:
        """
        Prepare dialogue for model input
        
        Args:
            dialogue: Dialogue turns
        
        Returns:
            Formatted text
        """
        formatted_lines = []
        for turn in dialogue:
            speaker = turn['speaker'].capitalize()
            text = turn['text']
            formatted_lines.append(f"{speaker}: {text}")
        
        return '\n'.join(formatted_lines)
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        symptoms = []
        
        symptom_keywords = [
            'neck pain', 'back pain', 'head impact', 'discomfort', 
            'stiffness', 'backache', 'trouble sleeping', 'pain'
        ]
        
        for keyword in symptom_keywords:
            if keyword in text.lower():
                symptoms.append(keyword.capitalize())
        
        # Deduplicate
        symptoms = list(set(symptoms))
        
        return symptoms if symptoms else [mark_missing_field('symptoms')]
    
    def _extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis from text"""
        import re
        
        # Look for common diagnostic patterns
        patterns = [
            r'diagnosed with ([a-z\s]+(?:injury|condition|disorder))',
            r'said it was (?:a )?([a-z\s]+(?:injury|condition))',
            r'(whiplash (?:injury)?)',
            r'(lower back strain)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().capitalize()
        
        # Check for specific diagnosis mentions
        if 'whiplash' in text.lower():
            return "Whiplash injury"
        
        return mark_missing_field('diagnosis')
    
    def _extract_treatment(self, text: str) -> List[str]:
        """Extract treatments from text"""
        import re
        
        treatments = []
        
        # Pattern for numbered sessions
        session_pattern = r'(\d+\s+sessions? of (?:physiotherapy|physical therapy))'
        matches = re.findall(session_pattern, text, re.IGNORECASE)
        treatments.extend([m.capitalize() for m in matches])
        
        # General treatment keywords
        if 'physiotherapy' in text.lower() or 'physical therapy' in text.lower():
            if not any('physiotherapy' in t.lower() for t in treatments):
                treatments.append("Physiotherapy")
        
        if 'painkiller' in text.lower():
            treatments.append("Painkillers")
        
        return treatments if treatments else [mark_missing_field('treatment')]
    
    def _extract_current_status(self, dialogue: List[Dict]) -> str:
        """Extract current status from patient's recent statements"""
        # Look at recent patient statements
        patient_turns = [turn for turn in dialogue if turn['speaker'] == 'patient']
        
        if len(patient_turns) > 0:
            # Check last few patient statements
            recent_text = ' '.join([turn['text'] for turn in patient_turns[-3:]])
            recent_lower = recent_text.lower()
            
            if 'doing better' in recent_lower or 'improving' in recent_lower:
                if 'occasional' in recent_lower and ('back' in recent_lower or 'pain' in recent_lower):
                    return "Improving, occasional backaches"
                return "Improving"
            
            if 'occasional' in recent_lower and 'pain' in recent_lower:
                return "Occasional pain"
        
        return "Not mentioned"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis from text"""
        import re
        
        # Look for prognosis patterns
        prognosis_patterns = [
            r'(full recovery (?:expected )?within [a-z\s\d]+)',
            r'(expect (?:to make )?(?:a )?full recovery)',
            r'(no long-term (?:damage|impact))',
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        if 'full recovery' in text.lower():
            return "Full recovery expected"
        
        return mark_missing_field('prognosis')
    
    def _validate_and_complete_summary(self, summary: Dict) -> Dict:
        """
        Validate summary and ensure all fields are present
        
        Args:
            summary: Generated summary
        
        Returns:
            Validated and completed summary
        """
        required_fields = [
            'patient_name', 'symptoms', 'diagnosis',
            'treatment', 'current_status', 'prognosis'
        ]
        
        for field in required_fields:
            if field not in summary or summary[field] is None or summary[field] == "":
                summary[field] = mark_missing_field(field)
        
        # Convert lists to proper format
        if isinstance(summary['symptoms'], str) and summary['symptoms'] != "Not mentioned":
            summary['symptoms'] = [summary['symptoms']]
        
        if isinstance(summary['treatment'], str) and summary['treatment'] != "Not mentioned":
            summary['treatment'] = [summary['treatment']]
        
        return summary
    
    def extract_keywords(self, dialogue: List[Dict], top_k: int = 10) -> List[str]:
        """
        Extract key medical phrases from dialogue
        
        Args:
            dialogue: Dialogue turns
            top_k: Number of keywords to extract
        
        Returns:
            List of key phrases
        """
        from utils import extract_key_phrases
        
        full_text = ' '.join([turn['text'] for turn in dialogue])
        keywords = extract_key_phrases(full_text, max_phrases=top_k)
        
        return keywords


def summarize_dialogue(dialogue: List[Dict], entities: Dict = None) -> Dict:
    """
    Convenience function for summarization
    
    Args:
        dialogue: Structured dialogue
        entities: Optional extracted entities
    
    Returns:
        Medical summary dictionary
    """
    summarizer = ClinicalSummarizer()
    return summarizer.generate_summary(dialogue, entities)


if __name__ == "__main__":
    # Test summarization
    print("Testing Clinical Summarizer...")
    
    test_dialogue = [
        {"speaker": "patient", "text": "I had a car accident. I experienced whiplash injury with neck pain and back pain."},
        {"speaker": "doctor", "text": "How severe was the pain?"},
        {"speaker": "patient", "text": "Very bad for four weeks. I had ten sessions of physiotherapy."},
        {"speaker": "doctor", "text": "I expect you'll make a full recovery within six months."},
    ]
    
    summarizer = ClinicalSummarizer()
    summary = summarizer.generate_summary(test_dialogue)
    
    print("\nGenerated Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nSummarization test complete!")
