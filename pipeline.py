"""
Main Pipeline Orchestrator for Clinical NLP System
Coordinates all modules to process medical transcripts end-to-end full
"""


import json
import logging
from typing import Dict, Tuple
from pathlib import Path

# Import all modules
from preprocessing import ClinicalPreprocessor
from ner_module import ClinicalNER
from summarization_module import ClinicalSummarizer
from sentiment_module import analyze_patient_sentiment_intent
from soap_generator import SOAPGenerator
from utils import save_json_output, format_json_output
from config import CLINICAL_DISCLAIMER, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalNLPPipeline:
    """
    End-to-end clinical NLP pipeline
    Processes physician-patient transcripts into structured medical documentation
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        logger.info("="*70)
        logger.info("Initializing Clinical NLP Pipeline")
        logger.info("="*70)
        
        self.preprocessor = ClinicalPreprocessor()
        self.ner = ClinicalNER()
        self.summarizer = ClinicalSummarizer()
        self.soap_generator = SOAPGenerator()
        
        logger.info("All pipeline components initialized successfully")
        logger.info(f"Output directory: {OUTPUT_DIR}")
    
    def process_transcript(self, raw_transcript: str, save_outputs: bool = True) -> Dict:
        """
        Process a raw medical transcript through the complete pipeline
        
        Args:
            raw_transcript: Raw transcript text
            save_outputs: Whether to save outputs to files
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING CLINICAL NLP PIPELINE")
        logger.info("="*70)
        
        # ----------------------------------------------------------------
        # STEP 1: Preprocessing
        # ----------------------------------------------------------------
        logger.info("\n[STEP 1/5] Preprocessing transcript...")
        dialogue, cleaned_text = self.preprocessor.process(raw_transcript)
        logger.info(f"✓ Parsed {len(dialogue)} dialogue turns")
        
        # ----------------------------------------------------------------
        # STEP 2: Named Entity Recognition
        # ----------------------------------------------------------------
        logger.info("\n[STEP 2/5] Extracting clinical entities...")
        entities_list = self.ner.extract_entities(cleaned_text, dialogue)
        entities_summary = self.ner.extract_structured_summary(entities_list)
        logger.info(f"✓ Extracted {len(entities_list)} clinical entities")
        
        # ----------------------------------------------------------------
        # STEP 3: Medical Summarization
        # ----------------------------------------------------------------
        logger.info("\n[STEP 3/5] Generating medical summary...")
        medical_summary = self.summarizer.generate_summary(dialogue, entities_summary)
        keywords = self.summarizer.extract_keywords(dialogue)
        logger.info(f"✓ Generated structured summary with {len(keywords)} keywords")
        
        # ----------------------------------------------------------------
        # STEP 4: Sentiment & Intent Analysis
        # ----------------------------------------------------------------
        logger.info("\n[STEP 4/5] Analyzing patient sentiment and intent...")
        sentiment_intent = analyze_patient_sentiment_intent(dialogue)
        logger.info(f"✓ Detected sentiment: {sentiment_intent['overall_sentiment']}")
        logger.info(f"✓ Detected intent: {sentiment_intent['overall_intent']}")
        
        # ----------------------------------------------------------------
        # STEP 5: SOAP Note Generation
        # ----------------------------------------------------------------
        logger.info("\n[STEP 5/5] Generating SOAP note...")
        soap_note = self.soap_generator.generate_soap_note(dialogue, entities_summary, medical_summary)
        logger.info("✓ SOAP note generated successfully")
        
        # ----------------------------------------------------------------
        # Consolidate Results
        # ----------------------------------------------------------------
        logger.info("\n" + "="*70)
        logger.info("CONSOLIDATING RESULTS")
        logger.info("="*70)
        
        results = {
            'medical_summary': medical_summary,
            'extracted_entities': {
                'entity_list': entities_list,
                'entity_summary': entities_summary
            },
            'sentiment_and_intent': sentiment_intent,
            'soap_note': soap_note,
            'keywords': keywords,
            'metadata': {
                'num_dialogue_turns': len(dialogue),
                'num_patient_turns': len([t for t in dialogue if t['speaker'] == 'patient']),
                'num_doctor_turns': len([t for t in dialogue if t['speaker'] == 'doctor']),
                'num_entities_extracted': len(entities_list),
                'clinical_disclaimer': CLINICAL_DISCLAIMER
            }
        }
        
        logger.info("✓ All results consolidated")
        
        # ----------------------------------------------------------------
        # Save Outputs
        # ----------------------------------------------------------------
        if save_outputs:
            self._save_all_outputs(results)
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70 + "\n")
        
        return results
    
    def _save_all_outputs(self, results: Dict):
        """
        Save all outputs to JSON files
        
        Args:
            results: Complete results dictionary
        """
        logger.info("\nSaving outputs to files...")
        
        # Save complete results
        save_json_output(results, 'complete_results.json')
        
        # Save individual components
        save_json_output(results['medical_summary'], 'medical_summary.json')
        save_json_output(results['extracted_entities'], 'extracted_entities.json')
        save_json_output(results['sentiment_and_intent'], 'sentiment_intent.json')
        save_json_output(results['soap_note'], 'soap_note.json')
        
        logger.info("✓ All outputs saved to disk")
    
    def display_results(self, results: Dict):
        """
        Display results in human-readable format
        
        Args:
            results: Pipeline results
        """
        print("\n" + "="*70)
        print("CLINICAL NLP ANALYSIS RESULTS")
        print("="*70)
        
        # Medical Summary
        print("\n📋 MEDICAL SUMMARY")
        print("-"*70)
        summary = results['medical_summary']
        print(f"Patient Name: {summary.get('patient_name', 'N/A')}")
        print(f"Symptoms: {', '.join(summary.get('symptoms', [])) if isinstance(summary.get('symptoms'), list) else summary.get('symptoms')}")
        print(f"Diagnosis: {summary.get('diagnosis', 'N/A')}")
        print(f"Treatment: {', '.join(summary.get('treatment', [])) if isinstance(summary.get('treatment'), list) else summary.get('treatment')}")
        print(f"Current Status: {summary.get('current_status', 'N/A')}")
        print(f"Prognosis: {summary.get('prognosis', 'N/A')}")
        
        # Entities
        print("\n🔍 EXTRACTED ENTITIES")
        print("-"*70)
        entity_summary = results['extracted_entities']['entity_summary']
        for entity_type, entities in entity_summary.items():
            if entities and entities != "Not mentioned":
                if isinstance(entities, list):
                    print(f"{entity_type.upper()}: {', '.join(entities)}")
                else:
                    print(f"{entity_type.upper()}: {entities}")
        
        # Sentiment & Intent
        print("\n💭 SENTIMENT & INTENT ANALYSIS")
        print("-"*70)
        sent_int = results['sentiment_and_intent']
        print(f"Overall Sentiment: {sent_int['overall_sentiment']}")
        print(f"Overall Intent: {sent_int['overall_intent']}")
        
        # SOAP Note
        print("\n📝 SOAP NOTE")
        print("-"*70)
        soap = results['soap_note']
        
        print("\nSUBJECTIVE:")
        print(f"  Chief Complaint: {soap['Subjective']['chief_complaint']}")
        print(f"  HPI: {soap['Subjective']['history_of_present_illness'][:150]}...")
        
        print("\nOBJECTIVE:")
        print(f"  Physical Exam: {soap['Objective']['physical_exam']}")
        
        print("\nASSESSMENT:")
        print(f"  Diagnosis: {soap['Assessment']['diagnosis']}")
        print(f"  Severity: {soap['Assessment']['severity']}")
        
        print("\nPLAN:")
        print(f"  Treatment: {soap['Plan']['treatment']}")
        print(f"  Follow-up: {soap['Plan']['follow_up']}")
        
        # Keywords
        print("\n🔑 KEY MEDICAL PHRASES")
        print("-"*70)
        keywords = results.get('keywords', [])
        for i, keyword in enumerate(keywords[:5], 1):
            print(f"{i}. {keyword}")
        
        print("\n" + "="*70)
        print("⚠️  CLINICAL SAFETY NOTICE")
        print("="*70)
        print("This is a documentation assistant tool only.")
        print("All outputs must be reviewed by qualified healthcare professionals.")
        print("="*70 + "\n")


def process_transcript_file(filepath: str, save_outputs: bool = True) -> Dict:
    """
    Convenience function to process a transcript from a file
    
    Args:
        filepath: Path to transcript file
        save_outputs: Whether to save outputs
    
    Returns:
        Analysis results
    """
    # Read transcript
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_transcript = f.read()
    
    # Process
    pipeline = ClinicalNLPPipeline()
    results = pipeline.process_transcript(raw_transcript, save_outputs)
    
    return results


if __name__ == "__main__":
    # Test the pipeline
    print("\nTesting Clinical NLP Pipeline...")
    print("="*70)
    
    # Sample transcript
    test_transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    Physician: I understand you were in a car accident last September.
    
    Patient: Yes, it was on September 1st. I had whiplash injury. My neck and back pain were really bad for four weeks.
    
    Physician: Did you receive treatment?
    
    Patient: Yes, I had ten sessions of physiotherapy and took painkillers regularly.
    
    Physician: Are you still experiencing pain?
    
    Patient: It's not constant, but I do get occasional backaches.
    
    Physician: Let me do a physical examination.
    
    [Physical Examination Conducted]
    
    Physician: Everything looks good. Full range of motion, no tenderness. I expect you to make a full recovery within six months.
    
    Patient: That's great to hear. Thank you, doctor!
    """
    
    # Run pipeline
    pipeline = ClinicalNLPPipeline()
    results = pipeline.process_transcript(test_transcript, save_outputs=False)
    
    # Display results
    pipeline.display_results(results)
    
    print("\n✅ Pipeline test complete!")
