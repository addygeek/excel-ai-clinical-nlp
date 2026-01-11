"""
Test Suite for Clinical NLP Pipeline
Unit tests for all components
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import ClinicalPreprocessor, preprocess_transcript
from ner_module import ClinicalNER, extract_clinical_entities
from summarization_module import ClinicalSummarizer, summarize_dialogue
from sentiment_module import ClinicalSentimentAnalyzer, ClinicalIntentDetector
from soap_generator import SOAPGenerator, generate_soap_note
from pipeline import ClinicalNLPPipeline
from evaluation import ClinicalNLPEvaluator


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing module"""
    
    def setUp(self):
        self.preprocessor = ClinicalPreprocessor()
        self.sample_text = """
        Physician: Good morning, Ms. Jones.
        Patient: Good morning, doctor. I have neck pain.
        """
    
    def test_parse_transcript(self):
        """Test transcript parsing"""
        dialogue = self.preprocessor.parse_transcript(self.sample_text)
        
        self.assertIsInstance(dialogue, list)
        self.assertGreater(len(dialogue), 0)
        self.assertEqual(dialogue[0]['speaker'], 'doctor')
        self.assertEqual(dialogue[1]['speaker'], 'patient')
    
    def test_temporal_normalization(self):
        """Test temporal expression normalization"""
        text = "I had pain for four weeks and ten sessions"
        normalized = self.preprocessor.normalize_temporal_expressions(text)
        
        self.assertIn("4 weeks", normalized)
        self.assertIn("10 sessions", normalized)
    
    def test_full_preprocessing(self):
        """Test complete preprocessing pipeline"""
        dialogue, cleaned_text = self.preprocessor.process(self.sample_text)
        
        self.assertIsInstance(dialogue, list)
        self.assertIsInstance(cleaned_text, str)
        self.assertGreater(len(dialogue), 0)


class TestNER(unittest.TestCase):
    """Test Named Entity Recognition"""
    
    def setUp(self):
        self.ner = ClinicalNER()
        self.sample_text = "Patient has whiplash injury with neck pain and back pain. Received physiotherapy."
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        entities = self.ner.extract_entities(self.sample_text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Check entity structure
        for entity in entities:
            self.assertIn('text', entity)
            self.assertIn('type', entity)
            self.assertIn('confidence', entity)
    
    def test_symptom_extraction(self):
        """Test symptom entity extraction"""
        entities = self.ner.extract_entities(self.sample_text)
        symptom_entities = [e for e in entities if e['type'] == 'SYMPTOM']
        
        self.assertGreater(len(symptom_entities), 0)
        
        # Check for expected symptoms
        symptom_texts = [e['text'] for e in symptom_entities]
        self.assertTrue(any('pain' in text for text in symptom_texts))
    
    def test_diagnosis_extraction(self):
        """Test diagnosis entity extraction"""
        entities = self.ner.extract_entities(self.sample_text)
        diagnosis_entities = [e for e in entities if e['type'] == 'DIAGNOSIS']
        
        self.assertGreater(len(diagnosis_entities), 0)
    
    def test_structured_summary(self):
        """Test structured summary generation"""
        entities = self.ner.extract_entities(self.sample_text)
        summary = self.ner.extract_structured_summary(entities)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('symptoms', summary)
        self.assertIn('diagnosis', summary)
        self.assertIn('treatment', summary)


class TestSummarization(unittest.TestCase):
    """Test medical summarization"""
    
    def setUp(self):
        self.summarizer = ClinicalSummarizer()
        self.dialogue = [
            {"speaker": "patient", "text": "I had a car accident and have neck pain."},
            {"speaker": "doctor", "text": "I diagnosed whiplash injury."},
            {"speaker": "patient", "text": "I received ten sessions of physiotherapy."},
            {"speaker": "doctor", "text": "You should make a full recovery."}
        ]
    
    def test_summary_generation(self):
        """Test summary generation"""
        summary = self.summarizer.generate_summary(self.dialogue)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('patient_name', summary)
        self.assertIn('symptoms', summary)
        self.assertIn('diagnosis', summary)
        self.assertIn('treatment', summary)
        self.assertIn('prognosis', summary)
    
    def test_keyword_extraction(self):
        """Test keyword extraction"""
        keywords = self.summarizer.extract_keywords(self.dialogue)
        
        self.assertIsInstance(keywords, list)


class TestSentimentIntent(unittest.TestCase):
    """Test sentiment and intent analysis"""
    
    def setUp(self):
        self.sentiment_analyzer = ClinicalSentimentAnalyzer()
        self.intent_detector = ClinicalIntentDetector()
        self.dialogue = [
            {"speaker": "patient", "text": "I'm worried about my back pain."},
            {"speaker": "patient", "text": "But I'm feeling better now."},
            {"speaker": "patient", "text": "Will I recover fully?"}
        ]
    
    def test_sentiment_analysis(self):
        """Test sentiment classification"""
        results = self.sentiment_analyzer.analyze_sentiment(self.dialogue)
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_sentiment', results)
        self.assertIn(results['overall_sentiment'], ['Anxious', 'Neutral', 'Reassured'])
    
    def test_intent_detection(self):
        """Test intent classification"""
        results = self.intent_detector.detect_intent(self.dialogue)
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_intent', results)
        self.assertIn('utterance_analysis', results)


class TestSOAPGenerator(unittest.TestCase):
    """Test SOAP note generation"""
    
    def setUp(self):
        self.soap_generator = SOAPGenerator()
        self.dialogue = [
            {"speaker": "patient", "text": "I had a car accident. I have neck and back pain."},
            {"speaker": "doctor", "text": "Physical examination shows full range of motion."},
            {"speaker": "doctor", "text": "Diagnosis is whiplash injury."},
            {"speaker": "doctor", "text": "Continue physiotherapy. Return if pain worsens."}
        ]
    
    def test_soap_generation(self):
        """Test SOAP note structure"""
        soap = self.soap_generator.generate_soap_note(self.dialogue)
        
        self.assertIsInstance(soap, dict)
        self.assertIn('Subjective', soap)
        self.assertIn('Objective', soap)
        self.assertIn('Assessment', soap)
        self.assertIn('Plan', soap)
    
    def test_subjective_section(self):
        """Test Subjective section"""
        soap = self.soap_generator.generate_soap_note(self.dialogue)
        
        self.assertIn('chief_complaint', soap['Subjective'])
        self.assertIn('history_of_present_illness', soap['Subjective'])
    
    def test_objective_section(self):
        """Test Objective section"""
        soap = self.soap_generator.generate_soap_note(self.dialogue)
        
        self.assertIn('physical_exam', soap['Objective'])
        self.assertIn('observations', soap['Objective'])
    
    def test_assessment_section(self):
        """Test Assessment section"""
        soap = self.soap_generator.generate_soap_note(self.dialogue)
        
        self.assertIn('diagnosis', soap['Assessment'])
        self.assertIn('severity', soap['Assessment'])
    
    def test_plan_section(self):
        """Test Plan section"""
        soap = self.soap_generator.generate_soap_note(self.dialogue)
        
        self.assertIn('treatment', soap['Plan'])
        self.assertIn('follow_up', soap['Plan'])


class TestPipeline(unittest.TestCase):
    """Test complete pipeline"""
    
    def setUp(self):
        self.pipeline = ClinicalNLPPipeline()
        self.sample_transcript = """
        Physician: Good morning, Ms. Jones.
        Patient: I had a car accident. My neck hurts.
        Physician: I see whiplash injury. You'll need physiotherapy.
        Patient: Will I recover?
        Physician: Yes, full recovery expected in six months.
        """
    
    def test_pipeline_execution(self):
        """Test complete pipeline execution"""
        results = self.pipeline.process_transcript(self.sample_transcript, save_outputs=False)
        
        self.assertIsInstance(results, dict)
        self.assertIn('medical_summary', results)
        self.assertIn('extracted_entities', results)
        self.assertIn('sentiment_and_intent', results)
        self.assertIn('soap_note', results)
    
    def test_output_structure(self):
        """Test output structure validity"""
        results = self.pipeline.process_transcript(self.sample_transcript, save_outputs=False)
        
        # Check medical summary
        summary = results['medical_summary']
        required_fields = ['patient_name', 'symptoms', 'diagnosis', 'treatment', 'current_status', 'prognosis']
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check SOAP note
        soap = results['soap_note']
        required_sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
        for section in required_sections:
            self.assertIn(section, soap)


class TestEvaluation(unittest.TestCase):
    """Test evaluation module"""
    
    def setUp(self):
        self.evaluator = ClinicalNLPEvaluator()
    
    def test_ner_evaluation(self):
        """Test NER metrics"""
        pred_entities = [
            {'text': 'neck pain', 'type': 'SYMPTOM'},
            {'text': 'whiplash', 'type': 'DIAGNOSIS'}
        ]
        
        gold_entities = [
            {'text': 'neck pain', 'type': 'SYMPTOM'},
            {'text': 'back pain', 'type': 'SYMPTOM'},
        ]
        
        metrics = self.evaluator.evaluate_ner(pred_entities, gold_entities)
        
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
    
    def test_soap_evaluation(self):
        """Test SOAP note evaluation"""
        pred_soap = {
            'Subjective': {'chief_complaint': 'Pain', 'history_of_present_illness': 'Details'},
            'Objective': {'physical_exam': 'Normal', 'observations': 'Good'},
            'Assessment': {'diagnosis': 'Injury', 'severity': 'Mild'},
            'Plan': {'treatment': 'Therapy', 'follow_up': 'As needed'}
        }
        
        gold_soap = pred_soap.copy()
        
        metrics = self.evaluator.evaluate_soap_note(pred_soap, gold_soap)
        
        self.assertIn('structural_completeness', metrics)
        self.assertEqual(metrics['structural_completeness'], 1.0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestNER))
    suite.addTests(loader.loadTestsFromTestCase(TestSummarization))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentIntent))
    suite.addTests(loader.loadTestsFromTestCase(TestSOAPGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
