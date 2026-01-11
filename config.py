"""
Configuration file for Physician Notetaker Clinical NLP System
Centralized configuration for models, entity types, and system parameters
"""

import torch
from pathlib import Path

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically detect and use GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Named Entity Recognition (NER)
NER_MODEL = {
    "primary": "emilyalsentzer/Bio_ClinicalBERT",  # BioClinicalBERT for medical NER
    "fallback": "dmis-lab/biobert-base-cased-v1.2",  # BioBERT alternative
    "max_length": 512,
    "batch_size": 8
}

# Medical Text Summarization
SUMMARIZATION_MODEL = {
    "primary": "google/flan-t5-base",  # FLAN-T5 for clinical summarization
    "alternative": "facebook/bart-base",  # BART as alternative
    "max_input_length": 1024,
    "max_output_length": 256,
    "num_beams": 4,
    "temperature": 0.7
}

# Sentiment Analysis
SENTIMENT_MODEL = {
    "primary": "emilyalsentzer/Bio_ClinicalBERT",  # ClinicalBERT for sentiment
    "max_length": 512,
    "classes": ["Anxious", "Neutral", "Reassured"]
}

# Intent Detection
INTENT_MODEL = {
    "primary": "emilyalsentzer/Bio_ClinicalBERT",  # ClinicalBERT for intent
    "max_length": 512,
    "classes": [
        "Reporting Symptoms",
        "Seeking Reassurance",
        "Expressing Concern",
        "Confirming Recovery",
        "Asking Follow-up"
    ]
}

# SOAP Note Generation
SOAP_MODEL = {
    "primary": "google/flan-t5-base",  # T5 for SOAP generation
    "max_length": 1024
}

# ============================================================================
# CLINICAL ENTITY CATEGORIES
# ============================================================================

ENTITY_TYPES = [
    "SYMPTOM",      # Pain, discomfort, physical sensations
    "DIAGNOSIS",    # Medical conditions, injuries
    "TREATMENT",    # Medications, therapies, procedures
    "PROGNOSIS",    # Expected outcomes, recovery predictions
    "DURATION",     # Time periods, temporal expressions
    "ANATOMY",      # Body parts, anatomical references
    "FACILITY",     # Healthcare facilities, locations
    "MEDICATION",   # Specific drugs and dosages
    "PROCEDURE",    # Medical procedures and tests
]

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Noise words to remove (fillers, hesitations)
NOISE_WORDS = [
    "um", "uh", "uhm", "er", "ah", "like", "you know", "actually",
    "basically", "literally", "sort of", "kind of", "I mean"
]

# Speaker tags
SPEAKER_TAGS = {
    "doctor": ["Physician", "Doctor", "Dr", "Clinician"],
    "patient": ["Patient", "Ms", "Mr", "Mrs"]
}

# Temporal expression patterns
TEMPORAL_PATTERNS = {
    "months": r"(\w+)\s+(months?|weeks?|days?|years?)",
    "dates": r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?",
    "durations": r"(first|last|past|next)\s+(four|ten|\d+)\s+(weeks?|months?|days?|sessions?)"
}

# ============================================================================
# OUTPUT SCHEMA DEFINITIONS
# ============================================================================

# Medical Summary Schema
SUMMARY_SCHEMA = {
    "patient_name": str,
    "symptoms": list,
    "diagnosis": str,
    "treatment": list,
    "current_status": str,
    "prognosis": str
}

# Sentiment Analysis Schema
SENTIMENT_SCHEMA = {
    "overall_sentiment": str,
    "overall_intent": str,
    "utterance_analysis": list
}

# SOAP Note Schema
SOAP_SCHEMA = {
    "Subjective": {
        "chief_complaint": str,
        "history_of_present_illness": str
    },
    "Objective": {
        "physical_exam": str,
        "observations": str
    },
    "Assessment": {
        "diagnosis": str,
        "severity": str
    },
    "Plan": {
        "treatment": str,
        "follow_up": str
    }
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    "ner_min": 0.5,         # Minimum confidence for entity extraction
    "sentiment_min": 0.6,   # Minimum confidence for sentiment classification
    "intent_min": 0.6,      # Minimum confidence for intent classification
}

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
SAMPLE_TRANSCRIPT = PROJECT_ROOT / "sample_transcript.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_CACHE = PROJECT_ROOT / "models_cache"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_CACHE.mkdir(exist_ok=True)

# ============================================================================
# SAFETY SETTINGS
# ============================================================================

# Clinical safety flags
SAFETY_CONFIG = {
    "allow_inference": False,          # Never infer medical facts not in transcript
    "mark_missing_explicitly": True,   # Explicitly mark missing information
    "require_confidence": True,        # Include confidence scores in outputs
    "conservative_extraction": True,   # Prefer precision over recall
}

# Disclaimer text
CLINICAL_DISCLAIMER = """
IMPORTANT CLINICAL SAFETY NOTICE:
This system is a documentation assistant tool only. It does not provide medical 
advice, diagnosis, or treatment recommendations. All outputs must be reviewed 
by qualified healthcare professionals before clinical use. This tool should not 
replace physician judgment or clinical decision-making.
"""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}
