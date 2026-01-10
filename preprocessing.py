"""
Pre-Processing Layer for Clinical NLP System
Handles speaker segmentation, temporal normalization, noise removal, and basic coreference
"""

import re
from typing import List, Dict, Tuple
import logging
from config import NOISE_WORDS, SPEAKER_TAGS, TEMPORAL_PATTERNS

logger = logging.getLogger(__name__)


class ClinicalPreprocessor:
    """
    Pre-processor for clinical dialogue transcripts
    Performs cleaning and normalization while preserving medical meaning
    """
    
    def __init__(self):
        self.noise_words = NOISE_WORDS
        self.speaker_tags = SPEAKER_TAGS
        
    def parse_transcript(self, raw_text: str) -> List[Dict]:
        """
        Parse raw transcript into structured dialogue turns
        
        Args:
            raw_text: Raw transcript text with speaker labels
        
        Returns:
            List of dialogue turns with speaker and text
        """
        logger.info("Parsing transcript...")
        
        # Split by lines
        lines = raw_text.strip().split('\n')
        
        dialogue = []
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Check if line starts with a speaker tag
            speaker = self._detect_speaker(line)
            
            if speaker:
                # Save previous turn if exists
                if current_speaker and current_text:
                    dialogue.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text).strip()
                    })
                    current_text = []
                
                # Extract text after speaker tag
                text = self._extract_text_after_speaker(line)
                current_speaker = speaker
                current_text = [text] if text else []
            else:
                # Continuation of previous speaker
                if current_speaker:
                    current_text.append(line)
        
        # Add last turn
        if current_speaker and current_text:
            dialogue.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text).strip()
            })
        
        logger.info(f"Parsed {len(dialogue)} dialogue turns")
        return dialogue
    
    def _detect_speaker(self, line: str) -> str:
        """
        Detect speaker from line
        
        Args:
            line: Text line
        
        Returns:
            'doctor' or 'patient' or None
        """
        line_lower = line.lower()
        
        # Check for doctor tags
        for tag in self.speaker_tags['doctor']:
            if line_lower.startswith(tag.lower() + ':'):
                return 'doctor'
        
        # Check for patient tags
        for tag in self.speaker_tags['patient']:
            if line_lower.startswith(tag.lower() + ':') or \
               line_lower.startswith(tag.lower() + '.'):
                return 'patient'
        
        return None
    
    def _extract_text_after_speaker(self, line: str) -> str:
        """
        Extract text after speaker label
        
        Args:
            line: Line with speaker label
        
        Returns:
            Text without speaker label
        """
        # Remove speaker tag and colon
        match = re.match(r'^[^:]+:\s*(.*)$', line)
        if match:
            return match.group(1).strip()
        return line
    
    def normalize_temporal_expressions(self, text: str) -> str:
        """
        Normalize temporal expressions to standard formats
        
        Args:
            text: Input text
        
        Returns:
            Text with normalized temporal expressions
        """
        normalized = text
        
        # Normalize common duration patterns
        duration_map = {
            r'\bfour weeks\b': '4 weeks',
            r'\bten sessions\b': '10 sessions',
            r'\bsix months\b': '6 months',
            r'\bone week\b': '1 week',
            r'\bfirst four weeks\b': 'first 4 weeks',
        }
        
        for pattern, replacement in duration_map.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Normalize time expressions
        normalized = re.sub(r'\b12:30 in the afternoon\b', '12:30 PM', normalized)
        normalized = re.sub(r'\baround (\d+:\d+)\b', r'\1', normalized)
        
        return normalized
    
    def remove_noise(self, text: str) -> str:
        """
        Remove filler words and noise while preserving medical content
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        cleaned = text
        
        # Remove filler words
        for noise_word in self.noise_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(noise_word) + r'\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        
        # Remove repeated punctuation
        cleaned = re.sub(r'([.,;:!?])\1+', r'\1', cleaned)
        
        return cleaned.strip()
    
    def resolve_coreferences(self, dialogue: List[Dict]) -> List[Dict]:
        """
        Simple rule-based coreference resolution for medical contexts
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            Dialogue with resolved coreferences
        """
        resolved_dialogue = []
        
        # Track medical entities mentioned
        mentioned_conditions = []
        mentioned_body_parts = []
        mentioned_treatments = []
        
        # Common medical terms to track
        condition_keywords = ['accident', 'injury', 'pain', 'whiplash']
        body_part_keywords = ['neck', 'back', 'head', 'spine']
        treatment_keywords = ['physiotherapy', 'painkillers', 'treatment']
        
        for turn in dialogue:
            text = turn['text']
            resolved_text = text
            
            # Track mentioned entities
            text_lower = text.lower()
            
            for keyword in condition_keywords:
                if keyword in text_lower and keyword not in mentioned_conditions:
                    mentioned_conditions.append(keyword)
            
            for keyword in body_part_keywords:
                if keyword in text_lower and keyword not in mentioned_body_parts:
                    mentioned_body_parts.append(keyword)
            
            for keyword in treatment_keywords:
                if keyword in text_lower and keyword not in mentioned_treatments:
                    mentioned_treatments.append(keyword)
            
            # Resolve simple pronouns
            if mentioned_conditions:
                # Replace "it" when referring to condition
                if re.search(r'\bit\b.*\b(injury|condition|accident)\b', text_lower):
                    resolved_text = re.sub(
                        r'\bit\b',
                        f'the {mentioned_conditions[-1]}',
                        resolved_text,
                        count=1,
                        flags=re.IGNORECASE
                    )
            
            # Resolve "that pain" to specific pain
            if mentioned_body_parts:
                resolved_text = re.sub(
                    r'\bthat pain\b',
                    f'{mentioned_body_parts[-1]} pain',
                    resolved_text,
                    flags=re.IGNORECASE
                )
                
                resolved_text = re.sub(
                    r'\bthe pain\b',
                    f'{mentioned_body_parts[-1]} pain',
                    resolved_text,
                    flags=re.IGNORECASE
                )
            
            resolved_dialogue.append({
                'speaker': turn['speaker'],
                'text': resolved_text,
                'original_text': text
            })
        
        return resolved_dialogue
    
    def process(self, raw_transcript: str) -> Tuple[List[Dict], str]:
        """
        Complete preprocessing pipeline
        
        Args:
            raw_transcript: Raw transcript text
        
        Returns:
            Tuple of (structured dialogue, cleaned full text)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Parse into dialogue turns
        dialogue = self.parse_transcript(raw_transcript)
        
        # Step 2: Process each turn
        processed_dialogue = []
        
        for turn in dialogue:
            text = turn['text']
            
            # Normalize temporal expressions
            text = self.normalize_temporal_expressions(text)
            
            # Remove noise (but keep it light to preserve medical meaning)
            # We'll be conservative here
            # text = self.remove_noise(text)  # Optional - can be too aggressive
            
            processed_dialogue.append({
                'speaker': turn['speaker'],
                'text': text
            })
        
        # Step 3: Resolve coreferences
        processed_dialogue = self.resolve_coreferences(processed_dialogue)
        
        # Step 4: Create cleaned full text
        cleaned_text = self._combine_dialogue(processed_dialogue)
        
        logger.info("Preprocessing complete")
        
        return processed_dialogue, cleaned_text
    
    def _combine_dialogue(self, dialogue: List[Dict]) -> str:
        """
        Combine dialogue turns into single text
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            Combined text
        """
        return ' '.join([turn['text'] for turn in dialogue])
    
    def get_patient_utterances(self, dialogue: List[Dict]) -> List[str]:
        """
        Extract only patient utterances
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            List of patient utterances
        """
        return [turn['text'] for turn in dialogue if turn['speaker'] == 'patient']
    
    def get_doctor_utterances(self, dialogue: List[Dict]) -> List[str]:
        """
        Extract only doctor utterances
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            List of doctor utterances
        """
        return [turn['text'] for turn in dialogue if turn['speaker'] == 'doctor']


def preprocess_transcript(raw_transcript: str) -> Tuple[List[Dict], str]:
    """
    Convenience function for preprocessing
    
    Args:
        raw_transcript: Raw transcript text
    
    Returns:
        Tuple of (structured dialogue, cleaned text)
    """
    preprocessor = ClinicalPreprocessor()
    return preprocessor.process(raw_transcript)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing Clinical Preprocessor...")
    
    test_transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    Physician: I understand you were in a car accident last September.
    
    Patient: Yes, it was on September 1st. The first four weeks were rough. My neck and back pain were really bad.
    """
    
    preprocessor = ClinicalPreprocessor()
    dialogue, cleaned_text = preprocessor.process(test_transcript)
    
    print(f"\nParsed {len(dialogue)} turns:")
    for i, turn in enumerate(dialogue[:3]):
        print(f"{i+1}. {turn['speaker'].upper()}: {turn['text'][:80]}...")
    
    print(f"\nCleaned text length: {len(cleaned_text)} characters")
    print("\nPreprocessing test complete!")
