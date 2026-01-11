"""
Utility functions for the Clinical NLP System
Helper functions for validation, formatting, and common operations
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from config import CONFIDENCE_THRESHOLDS, SAFETY_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """
    Validate JSON data against a schema definition
    
    Args:
        data: JSON data to validate
        schema: Schema definition with expected types
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        for key, expected_type in schema.items():
            if isinstance(expected_type, dict):
                # Nested schema
                if key not in data or not isinstance(data[key], dict):
                    return False
                if not validate_json_schema(data[key], expected_type):
                    return False
            else:
                # Simple type check
                if key not in data:
                    logger.warning(f"Missing key in schema validation: {key}")
                    return False
                if not isinstance(data[key], expected_type):
                    logger.warning(f"Type mismatch for {key}: expected {expected_type}, got {type(data[key])}")
                    return False
        return True
    except Exception as e:
        logger.error(f"Schema validation error: {e}")
        return False


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """
    Remove duplicate entities based on text and type
    
    Args:
        entities: List of entity dictionaries
    
    Returns:
        Deduplicated list of entities
    """
    seen = set()
    deduplicated = []
    
    for entity in entities:
        # Create unique key from text and type
        key = (entity.get('text', '').lower().strip(), entity.get('type', ''))
        
        if key not in seen:
            seen.add(key)
            deduplicated.append(entity)
    
    logger.info(f"Deduplicated entities: {len(entities)} -> {len(deduplicated)}")
    return deduplicated


def normalize_entity_text(text: str) -> str:
    """
    Normalize entity text for consistency
    
    Args:
        text: Raw entity text
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation at the end
    text = re.sub(r'[.,;:!?]+$', '', text)
    
    return text


def filter_low_confidence(items: List[Dict], min_confidence: float = None) -> List[Dict]:
    """
    Filter out items below confidence threshold
    
    Args:
        items: List of items with 'confidence' field
        min_confidence: Minimum confidence threshold (uses config default if None)
    
    Returns:
        Filtered list of high-confidence items
    """
    if min_confidence is None:
        min_confidence = CONFIDENCE_THRESHOLDS['ner_min']
    
    filtered = [item for item in items if item.get('confidence', 0) >= min_confidence]
    
    logger.info(f"Confidence filtering: {len(items)} -> {len(filtered)} items")
    return filtered


def mark_missing_field(field_name: str) -> str:
    """
    Explicitly mark a field as missing/not mentioned
    
    Args:
        field_name: Name of the missing field
    
    Returns:
        Standard missing value marker
    """
    if SAFETY_CONFIG['mark_missing_explicitly']:
        return "Not mentioned"
    return None


def format_json_output(data: Dict, pretty: bool = True) -> str:
    """
    Format dictionary as JSON string
    
    Args:
        data: Dictionary to format
        pretty: Whether to use pretty printing
    
    Returns:
        JSON string
    """
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def save_json_output(data: Dict, filename: str, output_dir: str = "outputs") -> str:
    """
    Save JSON data to file
    
    Args:
        data: Dictionary to save
        filename: Output filename
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved output to: {filepath}")
    return str(filepath)


def extract_patient_name(dialogue: List[Dict]) -> Optional[str]:
    """
    Extract patient name from dialogue
    
    Args:
        dialogue: List of dialogue turns
    
    Returns:
        Patient name if found, None otherwise
    """
    # Look for patterns like "Ms. Jones", "Mr. Smith", etc.
    name_pattern = r'\b(Ms\.?|Mr\.?|Mrs\.?|Dr\.?)\s+([A-Z][a-z]+)\b'
    
    for turn in dialogue:
        text = turn.get('text', '')
        match = re.search(name_pattern, text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
    
    return None


def group_by_speaker(dialogue: List[Dict]) -> Dict[str, List[str]]:
    """
    Group dialogue turns by speaker
    
    Args:
        dialogue: List of dialogue turns with 'speaker' field
    
    Returns:
        Dictionary mapping speaker to list of utterances
    """
    grouped = {}
    
    for turn in dialogue:
        speaker = turn.get('speaker', 'unknown')
        text = turn.get('text', '')
        
        if speaker not in grouped:
            grouped[speaker] = []
        grouped[speaker].append(text)
    
    return grouped


def extract_temporal_expressions(text: str) -> List[str]:
    """
    Extract temporal expressions from text
    
    Args:
        text: Input text
    
    Returns:
        List of temporal expressions
    """
    temporal_exprs = []
    
    # Date patterns
    date_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    ]
    
    # Duration patterns
    duration_patterns = [
        r'\b\d+\s+(?:days?|weeks?|months?|years?|sessions?)\b',
        r'\b(?:first|last|past|next)\s+(?:few|several|\d+)\s+(?:days?|weeks?|months?|years?)\b',
    ]
    
    # Time patterns
    time_patterns = [
        r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        r'\b(?:morning|afternoon|evening|night)\b',
    ]
    
    all_patterns = date_patterns + duration_patterns + time_patterns
    
    for pattern in all_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        temporal_exprs.extend(matches)
    
    return list(set(temporal_exprs))


def calculate_confidence_score(scores: List[float]) -> float:
    """
    Calculate aggregate confidence score from multiple scores
    
    Args:
        scores: List of confidence scores
    
    Returns:
        Aggregate confidence score (mean)
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def clean_medical_text(text: str) -> str:
    """
    Clean medical text while preserving clinical meaning
    
    Args:
        text: Raw medical text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed text
    
    # Trim
    text = text.strip()
    
    return text


def create_timestamped_filename(base_name: str, extension: str = "json") -> str:
    """
    Create filename with timestamp
    
    Args:
        base_name: Base filename
        extension: File extension
    
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def merge_overlapping_entities(entities: List[Dict]) -> List[Dict]:
    """
    Merge entities that overlap in the text
    
    Args:
        entities: List of entities with start_char and end_char
    
    Returns:
        Merged list of non-overlapping entities
    """
    if not entities:
        return []
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: x.get('start_char', 0))
    
    merged = [sorted_entities[0]]
    
    for current in sorted_entities[1:]:
        last = merged[-1]
        
        # Check for overlap
        if current.get('start_char', 0) <= last.get('end_char', 0):
            # Keep the one with higher confidence
            if current.get('confidence', 0) > last.get('confidence', 0):
                merged[-1] = current
        else:
            merged.append(current)
    
    return merged


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key medical phrases from text
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
    
    Returns:
        List of key phrases
    """
    # Simple keyword extraction using common medical patterns
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    key_phrases = []
    medical_keywords = ['pain', 'injury', 'treatment', 'diagnosis', 'therapy', 
                       'medication', 'symptoms', 'recovery', 'examination']
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Check if sentence contains medical keywords
        if any(keyword in sentence.lower() for keyword in medical_keywords):
            # Extract noun phrases (simplified)
            words = sentence.split()
            if len(words) >= 2 and len(words) <= 6:
                key_phrases.append(sentence)
    
    return key_phrases[:max_phrases]


def format_entity_for_display(entity: Dict) -> str:
    """
    Format entity dictionary for human-readable display
    
    Args:
        entity: Entity dictionary
    
    Returns:
        Formatted string
    """
    text = entity.get('text', 'N/A')
    entity_type = entity.get('type', 'UNKNOWN')
    confidence = entity.get('confidence', 0.0)
    
    return f"{text} [{entity_type}] (confidence: {confidence:.2f})"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test deduplication
    test_entities = [
        {"text": "neck pain", "type": "SYMPTOM", "confidence": 0.9},
        {"text": "Neck Pain", "type": "SYMPTOM", "confidence": 0.85},
        {"text": "back pain", "type": "SYMPTOM", "confidence": 0.92}
    ]
    
    deduped = deduplicate_entities(test_entities)
    print(f"\nDeduplication test: {len(test_entities)} -> {len(deduped)}")
    
    # Test temporal extraction
    test_text = "The accident was on September 1st at 12:30 PM. I had ten physiotherapy sessions over four weeks."
    temporal = extract_temporal_expressions(test_text)
    print(f"\nTemporal expressions: {temporal}")
    
    print("\nUtility functions test complete!")
