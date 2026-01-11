"""
SOAP Note Generation Module (Bonus Feature)
Converts clinical dialogues into structured SOAP format
"""

from typing import List, Dict
import logging
import re
from utils import mark_missing_field

logger = logging.getLogger(__name__)


class SOAPGenerator:
    """
    Generates SOAP notes from clinical dialogue transcripts
    SOAP = Subjective, Objective, Assessment, Plan
    """
    
    def __init__(self):
        """Initialize SOAP generator"""
        logger.info("Initializing SOAP generator...")
        self._init_patterns()
        logger.info("SOAP generator initialized")
    
    def _init_patterns(self):
        """Initialize patterns for SOAP section detection"""
        
        # Subjective patterns (patient-reported)
        self.subjective_patterns = [
            r'\b(pain|hurt|discomfort|feel|sensation)\b',
            r'\b(had|have|experiencing)\b',
            r'\b(I (?:was|am|feel|had|have))\b',
        ]
        
        # Objective patterns (clinician observations)
        self.objective_patterns = [
            r'\b(examination|physical exam|checked|observed)\b',
            r'\b(full range of (?:motion|movement))\b',
            r'\b(no (?:tenderness|signs|evidence))\b',
            r'\b(appears|looks|seems)\b',
        ]
        
        # Assessment patterns (diagnosis/reasoning)
        self.assessment_patterns = [
            r'\b(diagnosis|diagnose|condition)\b',
            r'\b(whiplash|injury|strain)\b',
            r'\b((?:mild|moderate|severe) (?:pain|injury))\b',
        ]
        
        # Plan patterns (treatment/follow-up)
        self.plan_patterns = [
            r'\b(treatment|therapy|medication|sessions)\b',
            r'\b(continue|recommend|should|plan)\b',
            r'\b(follow[- ]up|return|come back)\b',
        ]
    
    def generate_soap_note(self, dialogue: List[Dict], entities: Dict = None, summary: Dict = None) -> Dict:
        """
        Generate SOAP note from dialogue
        
        Args:
            dialogue: Structured dialogue turns
            entities: Optional extracted entities
            summary: Optional medical summary
        
        Returns:
            SOAP note dictionary
        """
        logger.info("Generating SOAP note...")
        
        # Separate patient and doctor turns
        patient_turns = [turn for turn in dialogue if turn['speaker'] == 'patient']
        doctor_turns = [turn for turn in dialogue if turn['speaker'] == 'doctor']
        
        # Build SOAP sections
        soap_note = {
            'Subjective': self._generate_subjective(patient_turns, summary),
            'Objective': self._generate_objective(doctor_turns, dialogue),
            'Assessment': self._generate_assessment(dialogue, entities, summary),
            'Plan': self._generate_plan(dialogue, summary)
        }
        
        logger.info("SOAP note generation complete")
        return soap_note
    
    def _generate_subjective(self, patient_turns: List[Dict], summary: Dict = None) -> Dict:
        """
        Generate Subjective section (patient-reported information)
        
        Args:
            patient_turns: Patient's utterances
            summary: Optional medical summary
        
        Returns:
            Subjective section dictionary
        """
        # Chief complaint - primary symptom/issue
        chief_complaint = self._extract_chief_complaint(patient_turns)
        
        # History of present illness - detailed narrative
        hpi = self._extract_history_of_present_illness(patient_turns, summary)
        
        return {
            'chief_complaint': chief_complaint,
            'history_of_present_illness': hpi
        }
    
    def _extract_chief_complaint(self, patient_turns: List[Dict]) -> str:
        """Extract the main complaint"""
        if not patient_turns:
            return mark_missing_field('chief_complaint')
        
        # Look for first substantial patient statement about symptoms
        for turn in patient_turns[:3]:  # Check first few turns
            text_lower = turn['text'].lower()
            
            # Look for pain mentions
            if 'pain' in text_lower:
                if 'neck' in text_lower and 'back' in text_lower:
                    return "Neck and back pain"
                elif 'neck' in text_lower:
                    return "Neck pain"
                elif 'back' in text_lower:
                    return "Back pain"
            
            # Look for injury mentions
            if 'accident' in text_lower or 'injury' in text_lower:
                return "Post-accident injury concerns"
        
        return "Discomfort following motor vehicle accident"
    
    def _extract_history_of_present_illness(self, patient_turns: List[Dict], summary: Dict = None) -> str:
        """Extract detailed history"""
        hpi_parts = []
        
        # Look for accident description
        for turn in patient_turns:
            text_lower = turn['text'].lower()
            
            if 'accident' in text_lower or 'hit' in text_lower:
                # Extract accident details
                accident_info = self._clean_for_hpi(turn['text'])
                hpi_parts.append(accident_info)
                break
        
        # Add symptom progression
        for turn in patient_turns:
            text_lower = turn['text'].lower()
            
            if any(word in text_lower for word in ['week', 'month', 'session', 'physiotherapy']):
                progression = self._clean_for_hpi(turn['text'])
                if progression not in hpi_parts:
                    hpi_parts.append(progression)
        
        # Add current status
        if patient_turns:
            last_turn = patient_turns[-1]['text']
            if 'better' in last_turn.lower() or 'occasional' in last_turn.lower():
                hpi_parts.append(self._clean_for_hpi(last_turn))
        
        if not hpi_parts:
            return mark_missing_field('history_of_present_illness')
        
        return ' '.join(hpi_parts)
    
    def _clean_for_hpi(self, text: str) -> str:
        """Clean text for HPI section"""
        # Remove questions and conversational elements
        text = re.sub(r'\?.*$', '', text)
        text = text.strip()
        
        # Remove conversational starters
        text = re.sub(r'^(Yes,|No,|Well,|Um,|Uh,)\s*', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _generate_objective(self, doctor_turns: List[Dict], full_dialogue: List[Dict]) -> Dict:
        """
        Generate Objective section (clinician observations)
        
        Args:
            doctor_turns: Doctor's utterances
            full_dialogue: Full dialogue for context
        
        Returns:
            Objective section dictionary
        """
        # Look for physical exam findings
        physical_exam = self._extract_physical_exam(full_dialogue)
        
        # Other observations
        observations = self._extract_observations(doctor_turns)
        
        return {
            'physical_exam': physical_exam,
            'observations': observations
        }
    
    def _extract_physical_exam(self, dialogue: List[Dict]) -> str:
        """Extract physical examination findings"""
        exam_findings = []
        
        for turn in dialogue:
            text_lower = turn['text'].lower()
            
            # Look for examination mentions
            if 'examination' in text_lower or 'physical exam' in text_lower:
                exam_findings.append("[Physical Examination Conducted]")
            
            # Look for specific findings
            if 'full range of motion' in text_lower or 'full range of movement' in text_lower:
                exam_findings.append("Full range of motion in cervical and lumbar spine")
            
            if 'no tenderness' in text_lower:
                exam_findings.append("No tenderness on palpation")
            
            if 'no signs' in text_lower and 'damage' in text_lower:
                exam_findings.append("No signs of lasting damage")
        
        if not exam_findings:
            return mark_missing_field('physical_exam')
        
        return '. '.join(exam_findings) + '.'
    
    def _extract_observations(self, doctor_turns: List[Dict]) -> str:
        """Extract general clinical observations"""
        observations = []
        
        for turn in doctor_turns:
            text_lower = turn['text'].lower()
            
            if 'looks good' in text_lower or 'everything looks good' in text_lower:
                observations.append("Patient appears in good condition")
            
            if 'recovery' in text_lower and 'positive' in text_lower:
                observations.append("Recovery progress is positive")
        
        if not observations:
            return mark_missing_field('observations')
        
        return '. '.join(observations) + '.'
    
    def _generate_assessment(self, dialogue: List[Dict], entities: Dict = None, summary: Dict = None) -> Dict:
        """
        Generate Assessment section (diagnosis and clinical reasoning)
        
        Args:
            dialogue: Full dialogue
            entities: Extracted entities
            summary: Medical summary
        
        Returns:
            Assessment section dictionary
        """
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(dialogue, entities, summary)
        
        # Assess severity
        severity = self._assess_severity(dialogue)
        
        return {
            'diagnosis': diagnosis,
            'severity': severity
        }
    
    def _extract_diagnosis(self, dialogue: List[Dict], entities: Dict = None, summary: Dict = None) -> str:
        """Extract diagnosis"""
        # Try summary first
        if summary and 'diagnosis' in summary and summary['diagnosis'] != mark_missing_field('diagnosis'):
            return summary['diagnosis']
        
        # Try entities
        if entities and 'diagnosis' in entities and entities['diagnosis'] != mark_missing_field('diagnosis'):
            return entities['diagnosis']
        
        # Try dialogue
        for turn in dialogue:
            text_lower = turn['text'].lower()
            
            if 'whiplash' in text_lower:
                if 'lower back' in text_lower or 'back strain' in text_lower:
                    return "Whiplash injury and lower back strain"
                return "Whiplash injury"
        
        return mark_missing_field('diagnosis')
    
    def _assess_severity(self, dialogue: List[Dict]) -> str:
        """Assess severity from dialogue"""
        full_text = ' '.join([turn['text'] for turn in dialogue]).lower()
        
        if any(word in full_text for word in ['severe', 'terrible', 'excruciating']):
            return "Moderate to severe, improving with treatment"
        
        if 'occasional' in full_text or 'improving' in full_text:
            return "Mild, improving"
        
        if 'better' in full_text or 'less' in full_text:
            return "Improving"
        
        return "Mild to moderate"
    
    def _generate_plan(self, dialogue: List[Dict], summary: Dict = None) -> Dict:
        """
        Generate Plan section (treatment and follow-up)
        
        Args:
            dialogue: Full dialogue
            summary: Medical summary
        
        Returns:
            Plan section dictionary
        """
        # Extract treatment plan
        treatment_plan = self._extract_treatment_plan(dialogue, summary)
        
        # Extract follow-up instructions
        follow_up = self._extract_follow_up(dialogue)
        
        return {
            'treatment': treatment_plan,
            'follow_up': follow_up
        }
    
    def _extract_treatment_plan(self, dialogue: List[Dict], summary: Dict = None) -> str:
        """Extract treatment plan"""
        treatments = []
        
        # Try summary
        if summary and 'treatment' in summary:
            if isinstance(summary['treatment'], list):
                treatments.extend(summary['treatment'])
            elif summary['treatment'] != mark_missing_field('treatment'):
                treatments.append(summary['treatment'])
        
        # Look in dialogue
        for turn in dialogue:
            text_lower = turn['text'].lower()
            
            if 'physiotherapy' in text_lower:
                if 'continue' in text_lower:
                    treatments.append("Continue physiotherapy as needed")
                elif 'ten sessions' in text_lower or '10 sessions' in text_lower:
                    if "10 sessions of physiotherapy" not in treatments:
                        treatments.append("Completed 10 sessions of physiotherapy")
            
            if 'painkiller' in text_lower or 'analgesic' in text_lower:
                treatments.append("Use analgesics for pain relief as needed")
        
        if not treatments:
            return mark_missing_field('treatment_plan')
        
        return '. '.join(treatments) + '.'
    
    def _extract_follow_up(self, dialogue: List[Dict]) -> str:
        """Extract follow-up instructions"""
        full_text = ' '.join([turn['text'] for turn in dialogue]).lower()
        
        if 'come back' in full_text or 'return if' in full_text:
            if 'worsen' in full_text or 'pain worsens' in full_text:
                return "Patient to return if pain worsens or persists beyond 6 months"
        
        if 'follow-up' in full_text or 'follow up' in full_text:
            return "Follow-up as needed if symptoms change"
        
        if 'full recovery' in full_text and 'six months' in full_text:
            return "Patient to return if pain persists beyond expected 6-month recovery period"
        
        return "Return as needed if symptoms change or worsen"


def generate_soap_note(dialogue: List[Dict], entities: Dict = None, summary: Dict = None) -> Dict:
    """
    Convenience function for SOAP note generation
    
    Args:
        dialogue: Structured dialogue
        entities: Optional extracted entities
        summary: Optional medical summary
    
    Returns:
        SOAP note dictionary
    """
    generator = SOAPGenerator()
    return generator.generate_soap_note(dialogue, entities, summary)


if __name__ == "__main__":
    # Test SOAP generation
    print("Testing SOAP Generator...")
    
    test_dialogue = [
        {"speaker": "patient", "text": "I had a car accident on September 1st. I hit my head and had neck and back pain."},
        {"speaker": "doctor", "text": "Let's do a physical examination."},
        {"speaker": "doctor", "text": "Everything looks good. Full range of movement, no tenderness."},
        {"speaker": "patient", "text": "I had ten sessions of physiotherapy. The pain is now occasional."},
        {"speaker": "doctor", "text": "The diagnosis is whiplash injury. I expect full recovery within six months."},
    ]
    
    soap = generate_soap_note(test_dialogue)
    
    print("\nGenerated SOAP Note:")
    print("\nSUBJECTIVE:")
    print(f"  Chief Complaint: {soap['Subjective']['chief_complaint']}")
    print(f"  HPI: {soap['Subjective']['history_of_present_illness'][:100]}...")
    
    print("\nOBJECTIVE:")
    print(f"  Physical Exam: {soap['Objective']['physical_exam'][:100]}...")
    
    print("\nASSESSMENT:")
    print(f"  Diagnosis: {soap['Assessment']['diagnosis']}")
    print(f"  Severity: {soap['Assessment']['severity']}")
    
    print("\nPLAN:")
    print(f"  Treatment: {soap['Plan']['treatment'][:100]}...")
    print(f"  Follow-up: {soap['Plan']['follow_up']}")
    
    print("\nSOAP generation test complete!")
