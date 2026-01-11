"""
Evaluation Module for Clinical NLP System
Metrics and evaluation for NER, summarization, and classification tasks
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import json


class ClinicalNLPEvaluator:
    """
    Evaluation metrics for clinical NLP tasks
    """
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    # ================================================================
    # NER EVALUATION
    # ================================================================
    
    def evaluate_ner(self, predicted_entities: List[Dict], gold_entities: List[Dict]) -> Dict:
        """
        Evaluate NER performance
        
        Args:
            predicted_entities: List of predicted entities
            gold_entities: List of gold standard entities
        
        Returns:
            Evaluation metrics
        """
        # Entity-level exact match
        pred_set = set((e['text'].lower(), e['type']) for e in predicted_entities)
        gold_set = set((e['text'].lower(), e['type']) for e in gold_entities)
        
        # Calculate metrics
        true_positives = len(pred_set & gold_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    # ================================================================
    # SUMMARIZATION EVALUATION
    # ================================================================
    
    def evaluate_summarization(self, predicted_summary: Dict, gold_summary: Dict) -> Dict:
        """
        Evaluate summarization quality
        
        Args:
            predicted_summary: Generated summary
            gold_summary: Reference summary
        
        Returns:
            Evaluation metrics
        """
        # Field-level accuracy
        matching_fields = 0
        total_fields = 0
        
        for field in gold_summary.keys():
            total_fields += 1
            if field in predicted_summary:
                # Simple string matching (can be enhanced with ROUGE/BLEU)
                pred_val = str(predicted_summary[field]).lower()
                gold_val = str(gold_summary[field]).lower()
                
                if pred_val == gold_val or gold_val in pred_val:
                    matching_fields += 1
        
        field_accuracy = matching_fields / total_fields if total_fields > 0 else 0
        
        return {
            'field_accuracy': field_accuracy,
            'matching_fields': matching_fields,
            'total_fields': total_fields
        }
    
    # ================================================================
    # SENTIMENT CLASSIFICATION EVALUATION
    # ================================================================
    
    def evaluate_sentiment(self, predicted_labels: List[str], gold_labels: List[str]) -> Dict:
        """
        Evaluate sentiment classification
        
        Args:
            predicted_labels: Predicted sentiment labels
            gold_labels: True sentiment labels
        
        Returns:
            Classification metrics
        """
        accuracy = accuracy_score(gold_labels, predicted_labels)
        
        # Multi-class metrics
        precision = precision_score(gold_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(gold_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(gold_labels, predicted_labels, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(gold_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    # ================================================================
    # SOAP NOTE EVALUATION
    # ================================================================
    
    def evaluate_soap_note(self, predicted_soap: Dict, gold_soap: Dict) -> Dict:
        """
        Evaluate SOAP note quality
        
        Args:
            predicted_soap: Generated SOAP note
            gold_soap: Reference SOAP note
        
        Returns:
            Structural completeness and accuracy metrics
        """
        # Check structural completeness
        required_sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
        present_sections = sum(1 for section in required_sections if section in predicted_soap)
        structural_completeness = present_sections / len(required_sections)
        
        # Check subsection completeness
        total_subsections = 0
        present_subsections = 0
        
        for section in required_sections:
            if section in gold_soap and isinstance(gold_soap[section], dict):
                for subsection in gold_soap[section].keys():
                    total_subsections += 1
                    if (section in predicted_soap and 
                        isinstance(predicted_soap[section], dict) and 
                        subsection in predicted_soap[section] and
                        predicted_soap[section][subsection] != "Not mentioned"):
                        present_subsections += 1
        
        subsection_completeness = present_subsections / total_subsections if total_subsections > 0 else 0
        
        return {
            'structural_completeness': structural_completeness,
            'subsection_completeness': subsection_completeness,
            'present_sections': present_sections,
            'total_sections': len(required_sections)
        }
    
    # ================================================================
    # COMPREHENSIVE EVALUATION
    # ================================================================
    
    def evaluate_pipeline(self, results: Dict, gold_standard: Dict = None) -> Dict:
        """
        Comprehensive evaluation of entire pipeline
        
        Args:
            results: Pipeline output
            gold_standard: Optional gold standard annotations
        
        Returns:
            Complete evaluation metrics
        """
        evaluation = {
            'pipeline_completeness': self._check_pipeline_completeness(results),
            'output_validity': self._validate_outputs(results)
        }
        
        # If gold standard is provided, compute accuracy metrics
        if gold_standard:
            if 'entities' in gold_standard:
                evaluation['ner_metrics'] = self.evaluate_ner(
                    results['extracted_entities']['entity_list'],
                    gold_standard['entities']
                )
            
            if 'summary' in gold_standard:
                evaluation['summary_metrics'] = self.evaluate_summarization(
                    results['medical_summary'],
                    gold_standard['summary']
                )
            
            if 'soap' in gold_standard:
                evaluation['soap_metrics'] = self.evaluate_soap_note(
                    results['soap_note'],
                    gold_standard['soap']
                )
        
        return evaluation
    
    def _check_pipeline_completeness(self, results: Dict) -> Dict:
        """Check if all pipeline components produced outputs"""
        required_components = [
            'medical_summary',
            'extracted_entities',
            'sentiment_and_intent',
            'soap_note'
        ]
        
        present = {comp: comp in results for comp in required_components}
        completeness = sum(present.values()) / len(required_components)
        
        return {
            'completeness_score': completeness,
            'component_status': present
        }
    
    def _validate_outputs(self, results: Dict) -> Dict:
        """Validate that outputs are well-formed"""
        validation = {}
        
        # Validate medical summary
        if 'medical_summary' in results:
            required_fields = ['patient_name', 'symptoms', 'diagnosis', 'treatment', 'current_status', 'prognosis']
            summary = results['medical_summary']
            has_all_fields = all(field in summary for field in required_fields)
            validation['summary_valid'] = has_all_fields
        
        # Validate SOAP note structure
        if 'soap_note' in results:
            required_sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
            soap = results['soap_note']
            has_all_sections = all(section in soap for section in required_sections)
            validation['soap_valid'] = has_all_sections
        
        # Validate entities
        if 'extracted_entities' in results:
            entities = results['extracted_entities'].get('entity_list', [])
            validation['entities_extracted'] = len(entities) > 0
        
        validation['all_valid'] = all(validation.values())
        
        return validation
    
    def generate_evaluation_report(self, evaluation: Dict) -> str:
        """
        Generate human-readable evaluation report
        
        Args:
            evaluation: Evaluation results
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("CLINICAL NLP EVALUATION REPORT")
        report.append("="*70)
        
        # Pipeline completeness
        if 'pipeline_completeness' in evaluation:
            comp = evaluation['pipeline_completeness']
            report.append("\nPIPELINE COMPLETENESS:")
            report.append(f"  Overall Score: {comp['completeness_score']:.2%}")
            for component, status in comp['component_status'].items():
                status_symbol = "✓" if status else "✗"
                report.append(f"  {status_symbol} {component}")
        
        # Output validity
        if 'output_validity' in evaluation:
            valid = evaluation['output_validity']
            report.append("\nOUTPUT VALIDITY:")
            for key, value in valid.items():
                if key != 'all_valid':
                    status_symbol = "✓" if value else "✗"
                    report.append(f"  {status_symbol} {key}")
        
        # NER metrics
        if 'ner_metrics' in evaluation:
            ner = evaluation['ner_metrics']
            report.append("\nNER PERFORMANCE:")
            report.append(f"  Precision: {ner['precision']:.2%}")
            report.append(f"  Recall: {ner['recall']:.2%}")
            report.append(f"  F1 Score: {ner['f1_score']:.2%}")
        
        # Summary metrics
        if 'summary_metrics' in evaluation:
            summ = evaluation['summary_metrics']
            report.append("\nSUMMARIZATION PERFORMANCE:")
            report.append(f"  Field Accuracy: {summ['field_accuracy']:.2%}")
        
        # SOAP metrics
        if 'soap_metrics' in evaluation:
            soap = evaluation['soap_metrics']
            report.append("\nSOAP NOTE PERFORMANCE:")
            report.append(f"  Structural Completeness: {soap['structural_completeness']:.2%}")
            report.append(f"  Subsection Completeness: {soap['subsection_completeness']:.2%}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test evaluation
    print("Testing Clinical NLP Evaluator...")
    
    # Sample data
    pred_entities = [
        {'text': 'neck pain', 'type': 'SYMPTOM'},
        {'text': 'whiplash injury', 'type': 'DIAGNOSIS'}
    ]
    
    gold_entities = [
        {'text': 'neck pain', 'type': 'SYMPTOM'},
        {'text': 'back pain', 'type': 'SYMPTOM'},
        {'text': 'whiplash injury', 'type': 'DIAGNOSIS'}
    ]
    
    evaluator = ClinicalNLPEvaluator()
    ner_metrics = evaluator.evaluate_ner(pred_entities, gold_entities)
    
    print("\nNER Evaluation:")
    print(f"  Precision: {ner_metrics['precision']:.2%}")
    print(f"  Recall: {ner_metrics['recall']:.2%}")
    print(f"  F1 Score: {ner_metrics['f1_score']:.2%}")
    
    print("\n✅ Evaluation test complete!")
