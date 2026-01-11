"""
Demo Script for Clinical NLP System
Demonstrates the complete pipeline on the sample transcript
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ClinicalNLPPipeline
from config import SAMPLE_TRANSCRIPT, CLINICAL_DISCLAIMER
from utils import format_json_output


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"📌 {title}")
    print("="*80)


def main():
    """Main demo function"""
    
    print_header("PHYSICIAN NOTETAKER - CLINICAL NLP SYSTEM DEMO")
    
    print("\n" + CLINICAL_DISCLAIMER)
    print("\n[Press Enter to continue...]")
    input()
    
    # Load sample transcript
    print_section("LOADING SAMPLE TRANSCRIPT")
    
    try:
        with open(SAMPLE_TRANSCRIPT, 'r', encoding='utf-8') as f:
            raw_transcript = f.read()
        
        print(f"✓ Loaded transcript from: {SAMPLE_TRANSCRIPT}")
        print(f"✓ Transcript length: {len(raw_transcript)} characters")
        
        # Show preview
        print("\n📄 Transcript Preview (first 500 characters):")
        print("-" * 80)
        print(raw_transcript[:500] + "...")
        print("-" * 80)
        
    except FileNotFoundError:
        print(f"❌ Error: Sample transcript not found at {SAMPLE_TRANSCRIPT}")
        return
    
    print("\n[Press Enter to run the pipeline...]")
    input()
    
    # Initialize and run pipeline
    print_section("INITIALIZING CLINICAL NLP PIPELINE")
    
    pipeline = ClinicalNLPPipeline()
    
    print("\n[Press Enter to process transcript...]")
    input()
    
    # Process transcript
    print_section("PROCESSING TRANSCRIPT")
    print("\nRunning complete NLP pipeline...\n")
    
    results = pipeline.process_transcript(raw_transcript, save_outputs=True)
    
    print("\n[Press Enter to view results...]")
    input()
    
    # Display detailed results
    display_detailed_results(results)
    
    # Save demonstration output
    save_demo_output(results)
    
    print_header("DEMO COMPLETE")
    print("\n✅ All outputs have been saved to the 'outputs' directory")
    print("✅ Check the following files:")
    print("   - outputs/complete_results.json")
    print("   - outputs/medical_summary.json")
    print("   - outputs/extracted_entities.json")
    print("   - outputs/sentiment_intent.json")
    print("   - outputs/soap_note.json")
    print("\n" + "="*80 + "\n")


def display_detailed_results(results: Dict):
    """Display detailed results with formatted output"""
    
    # Medical Summary
    print_section("1. MEDICAL SUMMARY")
    summary = results['medical_summary']
    print(format_json_output(summary))
    
    print("\n[Press Enter for next section...]")
    input()
    
    # Extracted Entities
    print_section("2. EXTRACTED CLINICAL ENTITIES")
    entities_summary = results['extracted_entities']['entity_summary']
    print("\n📊 Entity Summary by Type:")
    for entity_type, entities in entities_summary.items():
        if entities and entities != "Not mentioned":
            print(f"\n{entity_type.upper()}:")
            if isinstance(entities, list):
                for entity in entities:
                    print(f"  • {entity}")
            else:
                print(f"  • {entities}")
    
    print(f"\n\n📋 Total Entities Extracted: {len(results['extracted_entities']['entity_list'])}")
    
    # Show top entities with confidence
    print("\n🔝 Top 10 High-Confidence Entities:")
    top_entities = sorted(
        results['extracted_entities']['entity_list'],
        key=lambda x: x.get('confidence', 0),
        reverse=True
    )[:10]
    
    for i, entity in enumerate(top_entities, 1):
        print(f"  {i}. {entity['text']} [{entity['type']}] - Confidence: {entity.get('confidence', 0):.2f}")
    
    print("\n[Press Enter for next section...]")
    input()
    
    # Sentiment & Intent Analysis
    print_section("3. SENTIMENT & INTENT ANALYSIS")
    sent_int = results['sentiment_and_intent']
    
    print(f"\n📊 Overall Analysis:")
    print(f"  • Sentiment: {sent_int['overall_sentiment']}")
    print(f"  • Intent: {sent_int['overall_intent']}")
    
    print(f"\n\n💬 Per-Utterance Analysis:")
    for i, utt in enumerate(sent_int.get('utterance_analysis', [])[:5], 1):
        print(f"\n  {i}. \"{utt['text'][:60]}...\"")
        print(f"     Sentiment: {utt['sentiment']} (conf: {utt.get('sentiment_confidence', 0):.2f})")
        print(f"     Intent: {utt['intent']} (conf: {utt.get('intent_confidence', 0):.2f})")
    
    print("\n[Press Enter for next section...]")
    input()
    
    # SOAP Note
    print_section("4. SOAP NOTE (BONUS FEATURE)")
    soap = results['soap_note']
    
    print("\n🏥 SUBJECTIVE (Patient-Reported):")
    print(f"  Chief Complaint: {soap['Subjective']['chief_complaint']}")
    print(f"  History of Present Illness:")
    print(f"    {soap['Subjective']['history_of_present_illness']}")
    
    print("\n🔬 OBJECTIVE (Clinical Observations):")
    print(f"  Physical Exam: {soap['Objective']['physical_exam']}")
    print(f"  Observations: {soap['Objective']['observations']}")
    
    print("\n🩺 ASSESSMENT (Diagnosis & Reasoning):")
    print(f"  Diagnosis: {soap['Assessment']['diagnosis']}")
    print(f"  Severity: {soap['Assessment']['severity']}")
    
    print("\n📋 PLAN (Treatment & Follow-up):")
    print(f"  Treatment: {soap['Plan']['treatment']}")
    print(f"  Follow-up: {soap['Plan']['follow_up']}")
    
    print("\n[Press Enter for next section...]")
    input()
    
    # Keywords
    print_section("5. KEY MEDICAL PHRASES")
    keywords = results.get('keywords', [])
    print("\n🔑 Extracted Keywords:")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")
    
    print("\n[Press Enter for metadata...]")
    input()
    
    # Metadata
    print_section("6. PROCESSING METADATA")
    metadata = results.get('metadata', {})
    print(f"\n📊 Pipeline Statistics:")
    print(f"  • Total Dialogue Turns: {metadata.get('num_dialogue_turns', 0)}")
    print(f"  • Patient Turns: {metadata.get('num_patient_turns', 0)}")
    print(f"  • Doctor Turns: {metadata.get('num_doctor_turns', 0)}")
    print(f"  • Entities Extracted: {metadata.get('num_entities_extracted', 0)}")


def save_demo_output(results: Dict):
    """Save demonstration-specific output"""
    # Create a demo report
    demo_report = {
        'demo_title': 'Physician Notetaker - Clinical NLP System Demonstration',
        'sample_case': 'Ms. Jones - Post-MVA Whiplash Injury Follow-up',
        'pipeline_outputs': {
            'medical_summary': results['medical_summary'],
            'soap_note': results['soap_note'],
            'sentiment': results['sentiment_and_intent']['overall_sentiment'],
            'intent': results['sentiment_and_intent']['overall_intent']
        },
        'statistics': results.get('metadata', {}),
        'disclaimer': CLINICAL_DISCLAIMER
    }
    
    # Save demo report
    from utils import save_json_output
    save_json_output(demo_report, 'demo_report.json')


if __name__ == "__main__":
    main()
