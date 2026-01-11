"""
Simple Demo Script to Generate and Visualize Clinical NLP Results
This script runs the NER module and then visualizes the results
"""

import json
from ner_module import ClinicalNER
from visualize_results import ClinicalNLPVisualizer

CLINICAL_DISCLAIMER = """
⚠️  CLINICAL DISCLAIMER:
This is a demonstration system and should NOT be used for actual medical decisions.
Always consult qualified healthcare professionals for medical advice.
"""

# Sample clinical text
SAMPLE_TEXT = """
Patient Report:
The patient was involved in a car accident three weeks ago. They hit their head on the 
steering wheel and experienced immediate neck pain and back pain. The patient went to 
City Hospital Accident and Emergency where they were examined. X-rays were taken and 
the doctor diagnosed a whiplash injury and lower back strain.

The patient has been attending physiotherapy sessions - they've completed ten sessions 
so far. They've also been taking painkillers regularly for the discomfort. The neck 
stiffness has improved significantly, and their range of motion in the cervical and 
lumbar spine is nearly back to normal.

The physiotherapist said the patient is making good progress and expects them to make 
a full recovery within six months. There should be no long-term damage from the soft 
tissue injuries sustained in the accident.
"""


def generate_sample_results():
    """Generate sample NLP results"""
    print("="*80)
    print("🔬 RUNNING CLINICAL NLP ANALYSIS")
    print("="*80)
    
    print("\n📝 Input Text:")
    print("-"*80)
    print(SAMPLE_TEXT)
    print("-"*80)
    
    # Initialize NER
    print("\n🤖 Initializing Clinical NER...")
    ner = ClinicalNER()
    
    # Extract entities
    print("🔍 Extracting entities...")
    entities = ner.extract_entities(SAMPLE_TEXT)
    
    # Get structured summary
    print("📊 Generating structured summary...")
    summary = ner.extract_structured_summary(entities)
    
    # Create complete output
    complete_output = {
        'metadata': {
            'timestamp': '2026-01-11T00:25:00',
            'input_length': len(SAMPLE_TEXT),
            'model': 'BioClinicalBERT + Rule-based',
        },
        'input_text': SAMPLE_TEXT,
        'entities': entities,
        'structured_summary': summary,
        'soap_note': {
            'subjective': 'Patient reports car accident 3 weeks ago with neck and back pain',
            'objective': 'X-rays performed, improved range of motion in cervical and lumbar spine',
            'assessment': 'Whiplash injury and lower back strain',
            'plan': 'Continue physiotherapy, painkillers as needed, expect full recovery in 6 months'
        },
        'sentiment': {
            'overall': 'Cautiously Optimistic',
            'score': 0.65,
            'emotions': ['concern', 'hope', 'relief']
        },
        'intent': {
            'primary': 'Medical Documentation',
            'confidence': 0.92
        }
    }
    
    # Save to file
    print("\n💾 Saving results to 'clinical_nlp_results.json'...")
    with open('clinical_nlp_results.json', 'w') as f:
        json.dump(complete_output, f, indent=2)
    
    print("✅ Results saved successfully!")
    
    # Display quick summary
    print("\n" + "="*80)
    print("📋 QUICK SUMMARY")
    print("="*80)
    print(f"Total Entities Extracted: {len(entities)}")
    
    entity_types = {}
    for entity in entities:
        e_type = entity.get('type', 'UNKNOWN')
        entity_types[e_type] = entity_types.get(e_type, 0) + 1
    
    for e_type, count in sorted(entity_types.items()):
        print(f"  • {e_type}: {count}")
    
    return complete_output


def main():
    """Main demo function"""
    print(CLINICAL_DISCLAIMER)
    print("\n" + "="*80)
    print("🎯 CLINICAL NLP DEMO - ANALYSIS & VISUALIZATION")
    print("="*80)
    
    # Step 1: Generate results
    print("\n[STEP 1] Generating NLP Results...")
    results = generate_sample_results()
    
    # Step 2: Visualize results
    print("\n[STEP 2] Visualizing Results...")
    print("="*80)
    
    viz = ClinicalNLPVisualizer('clinical_nlp_results.json')
    
    # Display detailed summary
    viz.display_summary()
    
    # Generate statistics
    viz.generate_statistics_report()
    
    # Create visualizations
    print("\n[STEP 3] Creating Visualizations...")
    print("="*80)
    
    viz.plot_entity_distribution()
    viz.plot_confidence_scores()
    viz.plot_entity_network()
    viz.create_comprehensive_dashboard()
    
    # Export to CSV
    viz.export_to_csv()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  • clinical_nlp_results.json")
    print("  • entity_distribution.png")
    print("  • confidence_scores.png")
    print("  • entity_pie_chart.png")
    print("  • comprehensive_dashboard.png")
    print("  • clinical_entities.csv")
    
    print("\n💡 Next Steps:")
    print("  1. Open the PNG files to view the visualizations")
    print("  2. Open clinical_entities.csv in Excel for detailed analysis")
    print("  3. Review clinical_nlp_results.json for raw data")
    
    print("\n" + CLINICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
