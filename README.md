# 🩺 Physician Notetaker - Clinical NLP System

A comprehensive **Clinical Natural Language Processing (NLP) pipeline** that transforms physician-patient conversation transcripts into structured medical documentation.

## 📋 Overview

This system processes unstructured medical dialogue and generates:
- **Structured Medical Summaries** (JSON)
- **Extracted Clinical Entities** (Symptoms, Diagnosis, Treatment, Prognosis)
- **Patient Sentiment & Intent Analysis**
- **Structured SOAP Notes** (Subjective, Objective, Assessment, Plan)

### Key Features

✅ **Medical Entity Extraction** - Identifies symptoms, diagnoses, treatments, and more  
✅ **Clinical Summarization** - Converts dialogue into structured medical summaries  
✅ **Sentiment Analysis** - Detects patient emotional state (Anxious, Neutral, Reassured)  
✅ **Intent Detection** - Identifies patient communication intent  
✅ **SOAP Note Generation** - Creates clinically formatted documentation  
✅ **Safety-First Design** - Never hallucinates medical facts, explicitly marks missing data

---

## 🚀 Quick Start

### Installation

1. **Clone or Download** this repository

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download spaCy Model** (optional, for enhanced preprocessing)
```bash
python -m spacy download en_core_web_sm
```

### Run the Demo

```bash
python demo.py
```

This will process the sample transcript and display all outputs interactively.

---

## 💡 Usage

### Basic Usage

```python
from pipeline import ClinicalNLPPipeline

# Initialize pipeline
pipeline = ClinicalNLPPipeline()

# Process a transcript
raw_transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: I'm doing better, but I still have some discomfort.
...
"""

# Run complete pipeline
results = pipeline.process_transcript(raw_transcript, save_outputs=True)

# Display results
pipeline.display_results(results)
```

### Process from File

```python
from pipeline import process_transcript_file

# Process transcript from file
results = process_transcript_file('path/to/transcript.txt')
```

### Access Individual Components

```python
from preprocessing import preprocess_transcript
from ner_module import extract_clinical_entities
from summarization_module import summarize_dialogue
from sentiment_module import analyze_patient_sentiment_intent
from soap_generator import generate_soap_note

# Preprocess
dialogue, cleaned_text = preprocess_transcript(raw_transcript)

# Extract entities
entities, entity_summary = extract_clinical_entities(cleaned_text, dialogue)

# Generate summary
summary = summarize_dialogue(dialogue, entity_summary)

# Analyze sentiment
sentiment_intent = analyze_patient_sentiment_intent(dialogue)

# Generate SOAP note
soap_note = generate_soap_note(dialogue, entity_summary, summary)
```

---

## 📊 Output Format

### Medical Summary
```json
{
  "patient_name": "Ms. Jones",
  "symptoms": ["Neck pain", "Back pain", "Head impact"],
  "diagnosis": "Whiplash injury",
  "treatment": ["10 physiotherapy sessions", "Painkillers"],
  "current_status": "Occasional backache",
  "prognosis": "Full recovery expected within six months"
}
```

### Extracted Entities
```json
{
  "entity_list": [
    {
      "text": "neck pain",
      "type": "SYMPTOM",
      "confidence": 0.92,
      "start_char": 45,
      "end_char": 54
    }
  ],
  "entity_summary": {
    "symptoms": ["neck pain", "back pain"],
    "diagnosis": "Whiplash injury",
    "treatment": ["physiotherapy", "painkillers"]
  }
}
```

### Sentiment & Intent
```json
{
  "overall_sentiment": "Reassured",
  "overall_intent": "Seeking reassurance",
  "utterance_analysis": [
    {
      "text": "I'm a bit worried about my back pain...",
      "sentiment": "Anxious",
      "intent": "Expressing concern",
      "sentiment_confidence": 0.87,
      "intent_confidence": 0.82
    }
  ]
}
```

### SOAP Note
```json
{
  "Subjective": {
    "chief_complaint": "Neck and back pain",
    "history_of_present_illness": "Patient had car accident, experienced pain for 4 weeks..."
  },
  "Objective": {
    "physical_exam": "Full range of motion in cervical and lumbar spine. No tenderness.",
    "observations": "Patient appears in good condition."
  },
  "Assessment": {
    "diagnosis": "Whiplash injury",
    "severity": "Mild, improving"
  },
  "Plan": {
    "treatment": "Continue physiotherapy as needed. Use analgesics for pain relief.",
    "follow_up": "Return if pain persists beyond 6 months"
  }
}
```

---

## 🏗️ Architecture

```
┌─────────────────────┐
│  Raw Transcript     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Preprocessing      │
│  - Speaker tagging  │
│  - Normalization    │
│  - Coreference      │
└──────────┬──────────┘
           │
┌──────────▼──────────────────────────────────┐
│         Clinical NLP Pipeline               │
├─────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ NER Module   │  │ Summarization    │   │
│  └──────────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ Sentiment    │  │ Intent Detection │   │
│  └──────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────┐ │
│  │      SOAP Generator (Bonus)          │ │
│  └──────────────────────────────────────┘ │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────┐
│  Structured JSON Outputs     │
│  - Medical Summary           │
│  - Entities                  │
│  - Sentiment/Intent          │
│  - SOAP Note                 │
└──────────────────────────────┘
```

---

## 🔧 Technical Details

### Models & Approach

| Component | Approach | Model/Technique |
|-----------|----------|-----------------|
| **NER** | Rule-based + Pattern Matching | Clinical regex patterns, BioClinicalBERT-ready |
| **Summarization** | Template-based Extraction | Field-specific extraction with T5 support |
| **Sentiment** | Keyword-based Classification | Healthcare-specific sentiment keywords |
| **Intent** | Pattern Matching | Clinical dialogue intent patterns |
| **SOAP** | Hybrid Rule-based Mapping | Speaker attribution + semantic grouping |

### Entity Types Extracted

- **SYMPTOM**: Pain, discomfort, physical sensations
- **DIAGNOSIS**: Medical conditions, injuries
- **TREATMENT**: Medications, therapies, procedures
- **PROGNOSIS**: Expected outcomes, recovery predictions
- **DURATION**: Time periods, temporal expressions
- **ANATOMY**: Body parts, anatomical references
- **FACILITY**: Healthcare facilities
- **PROCEDURE**: Medical procedures and tests

### Sentiment Classes

- **Anxious**: Worried, concerned, fearful
- **Neutral**: Factual, balanced reporting
- **Reassured**: Relieved, confident, positive

### Intent Classes

- **Reporting Symptoms**: Describing medical issues
- **Seeking Reassurance**: Asking for positive prognosis
- **Expressing Concern**: Voicing worries
- **Confirming Recovery**: Acknowledging improvement
- **Asking Follow-up**: Questions about treatment

---

## 📁 Project Structure

```
excell ai clicnal nlp/
├── config.py                 # Configuration & constants
├── requirements.txt          # Python dependencies
├── preprocessing.py          # Text preprocessing module
├── ner_module.py            # Named Entity Recognition
├── summarization_module.py  # Medical summarization
├── sentiment_module.py      # Sentiment & intent analysis
├── soap_generator.py        # SOAP note generation
├── pipeline.py              # Main orchestrator
├── utils.py                 # Helper functions
├── evaluation.py            # Metrics & evaluation
├── demo.py                  # Interactive demonstration
├── test_pipeline.py         # Unit tests
├── sample_transcript.txt    # Sample input data
├── README.md                # This file
├── problem-statement.md     # Assignment requirements
├── online-help.md           # Research references
└── PRD.md                   # Product requirements
```

---

## 🧪 Testing

Run the test suite:

```bash
python test_pipeline.py
```

Or use pytest:

```bash
pytest test_pipeline.py -v
```

---

## ⚠️ Clinical Safety Notice

**IMPORTANT**: This is a **documentation assistant tool** only. It does NOT provide medical advice, diagnosis, or treatment recommendations.

- All outputs must be reviewed by qualified healthcare professionals
- Never use this system for clinical decision-making
- The system does not replace physician judgment
- Designed for documentation and research purposes only

---

## 🎯 Key Design Principles

### 1. **No Hallucination**
The system never fabricates medical information. Missing data is explicitly marked as "Not mentioned".

### 2. **Conservative Extraction**
Prioritizes precision over recall. Only extracts information present in the transcript.

### 3. **Transparency**
All extractions include confidence scores and can be traced back to source text.

### 4. **Structured Output**
All outputs are JSON-formatted for easy integration with other systems.

### 5. **Clinical Accuracy**
Uses domain-specific patterns and terminology validated against clinical standards.

---

## 📚 Example Use Cases

### 1. **Clinical Documentation**
Automatically generate structured notes from recorded consultations

### 2. **Medical Research**
Extract structured data from large corpora of medical dialogues

### 3. **Patient Sentiment Monitoring**
Track patient emotional state across multiple visits

### 4. **Training & Education**
Demonstrate NLP techniques for medical students and developers

### 5. **EHR Integration**
Generate SOAP notes compatible with Electronic Health Record systems

---

## 🔬 Model Fine-tuning (Future Enhancement)

For production deployment, consider fine-tuning transformer models:

```python
# Example: Fine-tune BioClinicalBERT for NER
from transformers import AutoModelForTokenClassification, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=len(ENTITY_TYPES)
)

# Train on labeled clinical dialogue dataset
trainer = Trainer(
    model=model,
    train_dataset=clinical_ner_dataset,
    # ... training configuration
)
trainer.train()
```

---

## 🤝 Contributing

This project was developed as an academic assignment demonstrating clinical NLP techniques. For production use, consider:

1. Fine-tuning models on labeled medical dialogue data
2. Adding medical entity linking (UMLS, SNOMED-CT)
3. Implementing more sophisticated coreference resolution
4. Adding multi-language support
5. Integrating with EHR systems via FHIR

---

## 📖 References

### Research Papers & Resources

- **BioBERT**: Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining" (2020)
- **ClinicalBERT**: Alsentzer et al., "Publicly Available Clinical BERT Embeddings" (2019)
- **Clinical Text Summarization**: PMC10635391 - "Clinical Text Summarization: Adapting Large Language Models"
- **Spark NLP for Healthcare**: John Snow Labs comparison studies

### Datasets (for future training)

- **MIMIC-III**: Medical Information Mart for Intensive Care
- **i2b2 Challenge Datasets**: Clinical NLP shared tasks
- **MedDialog**: Medical dialogue dataset
- **MTSamples**: Medical transcription samples

---

## 📄 License

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

Developed as part of the Clinical NLP assignment for 7th semester.

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named transformers`
```bash
# Solution:
pip install transformers torch
```

**Issue**: `CUDA out of memory`
```python
# Solution: Force CPU mode
import torch
torch.cuda.is_available = lambda: False
```

**Issue**: Missing sample transcript
```bash
# Solution: Ensure sample_transcript.txt exists
python demo.py
```

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Examine the demo.py for usage examples

---

**Made with ❤️ for advancing Clinical NLP**
#   e x c e l - a i - c l i n i c a l - n l p  
 