"""
Sentiment and Intent Analysis Module
Analyzes patient sentiment and intent in clinical dialogues
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import logging
from config import SENTIMENT_MODEL, INTENT_MODEL, DEVICE
import re

logger = logging.getLogger(__name__)


class ClinicalSentimentAnalyzer:
    """
    Sentiment analysis for healthcare-specific patient emotions
    """
    
    def __init__(self):
        """Initialize sentiment analysis model"""
        self.device = DEVICE
        self.sentiment_classes = SENTIMENT_MODEL['classes']
        
        logger.info("Initializing sentiment analyzer...")
        
        # For this implementation, we'll use rule-based sentiment + keyword matching
        # In production, you would fine-tune ClinicalBERT on medical dialogue sentiment data
        self.use_model = False
        
        # Define sentiment keywords
        self._init_sentiment_keywords()
        
        logger.info("Sentiment analyzer initialized")
    
    def _init_sentiment_keywords(self):
        """Initialize keywords for each sentiment class"""
        self.sentiment_keywords = {
            'Anxious': [
                'worried', 'concerned', 'nervous', 'afraid', 'scared',
                'anxious', 'uncertain', 'fear', 'worry', 'stress'
            ],
            'Neutral': [
                'okay', 'fine', 'normal', 'usual', 'regular',
                'same', 'stable', 'manageable'
            ],
            'Reassured': [
                'better', 'improving', 'relief', 'good', 'great',
                'glad', 'thankful', 'appreciate', 'positive', 'recovering',
                'happy', 'pleased', 'encouraged'
            ]
        }
    
    def analyze_sentiment(self, dialogue: List[Dict]) -> Dict:
        """
        Analyze sentiment across the entire dialogue
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            Sentiment analysis results
        """
        logger.info("Analyzing patient sentiment...")
        
        # Extract patient utterances only
        patient_utterances = [
            turn for turn in dialogue if turn['speaker'] == 'patient'
        ]
        
        # Analyze each utterance
        utterance_analysis = []
        sentiment_scores = {'Anxious': 0, 'Neutral': 0, 'Reassured': 0}
        
        for utterance in patient_utterances:
            text = utterance['text']
            sentiment, confidence = self._classify_sentiment(text)
            
            utterance_analysis.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': confidence
            })
            
            # Accumulate scores
            sentiment_scores[sentiment] += confidence
        
        # Determine overall sentiment
        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        
        result = {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_scores,
            'utterance_analysis': utterance_analysis
        }
        
        logger.info(f"Overall sentiment: {overall_sentiment}")
        return result
    
    def _classify_sentiment(self, text: str) -> tuple:
        """
        Classify sentiment of a single utterance
        
        Args:
            text: Utterance text
        
        Returns:
            Tuple of (sentiment_class, confidence)
        """
        text_lower = text.lower()
        scores = {'Anxious': 0.0, 'Neutral': 0.5, 'Reassured': 0.0}
        
        # Count keyword matches
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment] += 0.3
        
        # Contextual rules
        if 'but' in text_lower and any(word in text_lower for word in ['better', 'improving']):
            # "worried but getting better" -> mixed, leaning reassured
            scores['Reassured'] += 0.2
            scores['Anxious'] += 0.1
        
        if '?' in text:
            # Questions often indicate seeking reassurance
            scores['Anxious'] += 0.1
        
        # Negative modifiers
        if any(neg in text_lower for neg in ['not', "n't", 'no']):
            # "not worried" -> reassured
            if 'worried' in text_lower or 'concerned' in text_lower:
                scores['Reassured'] += 0.3
                scores['Anxious'] -= 0.2
        
        # Normalize scores
        max_score = max(scores.values())
        if max_score == 0:
            return 'Neutral', 0.7
        
        sentiment = max(scores, key=scores.get)
        confidence = min(max_score, 0.95)
        
        return sentiment, confidence


class ClinicalIntentDetector:
    """
    Intent detection for patient utterances in clinical dialogue
    """
    
    def __init__(self):
        """Initialize intent detection"""
        self.device = DEVICE
        self.intent_classes = INTENT_MODEL['classes']
        
        logger.info("Initializing intent detector...")
        
        # Rule-based intent detection
        self._init_intent_patterns()
        
        logger.info("Intent detector initialized")
    
    def _init_intent_patterns(self):
        """Initialize patterns for each intent class"""
        self.intent_patterns = {
            'Reporting Symptoms': [
                r'\b(pain|hurt|discomfort|ache|feel)\b',
                r'\b(had|have|experiencing|experience)\b',
                r'\b(my (?:neck|back|head))\b',
            ],
            'Seeking Reassurance': [
                r'\b(will (?:I|it)|going to|expect)\b',
                r'\b(don\'t need to worry|future|affect me)\b',
                r'\b(will (?:I|this)|can I)\b',
            ],
            'Expressing Concern': [
                r'\b(worried|concerned|anxious|afraid)\b',
                r'\b(what if|should I|problem)\b',
            ],
            'Confirming Recovery': [
                r'\b(better|improving|getting better|recovery)\b',
                r'\b(less|reduced|not as bad)\b',
            ],
            'Asking Follow-up': [
                r'\?',
                r'\b(when|how|what|why|should)\b.*\?',
            ]
        }
    
    def detect_intent(self, dialogue: List[Dict]) -> Dict:
        """
        Detect intent across dialogue
        
        Args:
            dialogue: List of dialogue turns
        
        Returns:
            Intent analysis results
        """
        logger.info("Detecting patient intent...")
        
        # Extract patient utterances
        patient_utterances = [
            turn for turn in dialogue if turn['speaker'] == 'patient'
        ]
        
        # Analyze each utterance
        utterance_analysis = []
        intent_counts = {intent: 0 for intent in self.intent_classes}
        
        for utterance in patient_utterances:
            text = utterance['text']
            intent, confidence = self._classify_intent(text)
            
            utterance_analysis.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'intent': intent,
                'confidence': confidence
            })
            
            intent_counts[intent] += 1
        
        # Determine overall/primary intent
        overall_intent = max(intent_counts, key=intent_counts.get)
        
        result = {
            'overall_intent': overall_intent,
            'intent_distribution': intent_counts,
            'utterance_analysis': utterance_analysis
        }
        
        logger.info(f"Overall intent: {overall_intent}")
        return result
    
    def _classify_intent(self, text: str) -> tuple:
        """
        Classify intent of a single utterance
        
        Args:
            text: Utterance text
        
        Returns:
            Tuple of (intent_class, confidence)
        """
        scores = {intent: 0.0 for intent in self.intent_classes}
        
        # Match patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[intent] += 0.4
        
        # Contextual rules
        text_lower = text.lower()
        
        # Reporting symptoms typically has specific pain/symptom mentions
        if any(word in text_lower for word in ['pain', 'hurt', 'discomfort', 'trouble']):
            scores['Reporting Symptoms'] += 0.3
        
        # Seeking reassurance often has future-oriented language
        if any(word in text_lower for word in ['future', 'will', 'going to', 'expect', 'worry about']):
            scores['Seeking Reassurancee'] += 0.2
        
        # Questions at the end often indicate seeking reassurance or follow-up
        if text.strip().endswith('?'):
            if any(word in text_lower for word in ['better', 'recover', 'long']):
                scores['Seeking Reassurance'] += 0.3
            else:
                scores['Asking Follow-up'] += 0.2
        
        # Determine max intent
        max_score = max(scores.values())
        if max_score == 0:
            return 'Reporting Symptoms', 0.5  # Default
        
        intent = max(scores, key=scores.get)
        confidence = min(max_score, 0.95)
        
        return intent, confidence


def analyze_patient_sentiment_intent(dialogue: List[Dict]) -> Dict:
    """
    Convenience function for combined sentiment and intent analysis
    
    Args:
        dialogue: Structured dialogue
    
    Returns:
        Combined analysis results
    """
    sentiment_analyzer = ClinicalSentimentAnalyzer()
    intent_detector = ClinicalIntentDetector()
    
    sentiment_results = sentiment_analyzer.analyze_sentiment(dialogue)
    intent_results = intent_detector.detect_intent(dialogue)
    
    # Combine for final output
    combined_results = {
        'overall_sentiment': sentiment_results['overall_sentiment'],
        'overall_intent': intent_results['overall_intent'],
        'utterance_analysis': []
    }
    
    # Merge utterance analyses
    for i in range(min(len(sentiment_results['utterance_analysis']), 
                       len(intent_results['utterance_analysis']))):
        sentiment_utt = sentiment_results['utterance_analysis'][i]
        intent_utt = intent_results['utterance_analysis'][i]
        
        combined_results['utterance_analysis'].append({
            'text': sentiment_utt['text'],
            'sentiment': sentiment_utt['sentiment'],
            'intent': intent_utt['intent'],
            'sentiment_confidence': sentiment_utt['confidence'],
            'intent_confidence': intent_utt['confidence']
        })
    
    return combined_results


if __name__ == "__main__":
    # Test sentiment and intent analysis
    print("Testing Sentiment and Intent Analysis...")
    
    test_dialogue = [
        {"speaker": "patient", "text": "I'm a bit worried about my back pain, but I hope it gets better soon."},
        {"speaker": "doctor", "text": "It should improve with physiotherapy."},
        {"speaker": "patient", "text": "I'm doing better now. The pain is less severe."},
        {"speaker": "patient", "text": "So, I don't need to worry about this affecting me in the future?"},
        {"speaker": "doctor", "text": "No, you should make a full recovery."},
        {"speaker": "patient", "text": "That's great to hear. Thank you!"},
    ]
    
    results = analyze_patient_sentiment_intent(test_dialogue)
    
    print(f"\nOverall Sentiment: {results['overall_sentiment']}")
    print(f"Overall Intent: {results['overall_intent']}")
    print(f"\nUtterance Analysis:")
    for utt in results['utterance_analysis']:
        print(f"  - '{utt['text'][:50]}...'")
        print(f"    Sentiment: {utt['sentiment']} (conf: {utt['sentiment_confidence']:.2f})")
        print(f"    Intent: {utt['intent']} (conf: {utt['intent_confidence']:.2f})")
    
    print("\nSentiment/Intent analysis test complete!")
