#!/usr/bin/env python3
"""
Medical entity extraction patterns for MRIA when spaCy models are not available.

This module provides comprehensive pattern-based medical entity extraction
as a fallback when spaCy models cannot be loaded.
"""

import re
from typing import List, Dict, Tuple
from datetime import datetime


class MedicalPatternExtractor:
    """Enhanced pattern-based medical entity extractor."""
    
    def __init__(self):
        """Initialize comprehensive medical patterns."""
        self.patterns = {
            'date': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
            ],
            'medication': [
                r'\b(?:metformin|insulin|lisinopril|amlodipine|atorvastatin|levothyroxine|aspirin|warfarin)\b',
                r'\b\w+\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|cc|units?)\s*(?:BID|TID|QID|daily|twice\s+daily|once\s+daily)?\b',
                r'\b[A-Z][a-z]+\s+\d+\s*mg\b',
                r'\b(?:Rx|prescription|medication|drug|medicine):\s*([^.\n]+)',
            ],
            'vitals': [
                r'\b(?:BP|Blood\s+Pressure):?\s*\d{2,3}[/\\]\d{2,3}\s*(?:mmHg)?\b',
                r'\b(?:HR|Heart\s+Rate):?\s*\d{2,3}\s*(?:bpm|beats?\s+per\s+minute)?\b',
                r'\b(?:Temp|Temperature):?\s*\d{2,3}(?:\.\d)?\s*°?[FC]?\b',
                r'\b(?:Weight):?\s*\d{2,3}(?:\.\d)?\s*(?:lbs?|kg|pounds?|kilograms?)\b',
                r'\b(?:Height):?\s*\d+[\'\"]\s*\d*[\"]*\b',
                r'\b(?:BMI):?\s*\d{2,3}(?:\.\d)?\b'
            ],            'lab_values': [
                r'\b(?:HbA1c|A1C|Hemoglobin\s+A1C):?\s*\d+(?:\.\d+)?%?\b',
                r'\b(?:glucose|sugar|blood\s+sugar):?\s*\d{2,3}\s*(?:mg/dL|mmol/L)?\b',
                r'\b(?:cholesterol|total\s+cholesterol):?\s*\d{2,3}\s*(?:mg/dL)?\b',
                r'\b(?:LDL|HDL):?\s*\d{2,3}\s*(?:mg/dL)?\b',
                r'\b(?:triglycerides):?\s*\d{2,4}\s*(?:mg/dL)?\b',
                r'\b(?:creatinine):?\s*\d+(?:\.\d+)?\s*(?:mg/dL)?\b',
                # Electrolyte patterns for lab reports like SODIUM, POTASSIUM
                r'\b(?:SODIUM|POTASSIUM|CHLORIDE|CO2),?\s*(?:SERUM)?\s*\d+(?:\.\d+)?\s*(?:mEq/L|mmol/L)\b',
                r'\b(?:Na|K|Cl)\+?\s*\d+(?:\.\d+)?\s*(?:mEq/L|mmol/L)\b',
                # General lab value pattern: TEST_NAME VALUE UNIT
                r'\b[A-Z][A-Z,\s]+\s+\d+(?:\.\d+)?\s*(?:mEq/L|mg/dL|mmol/L|units?/L|%)\b'
            ],
            'symptoms': [
                r'\b(?:fatigue|tired|weakness|exhaustion)\b',
                r'\b(?:thirst|polydipsia|increased\s+thirst)\b',
                r'\b(?:urination|polyuria|increased\s+urination)\b',
                r'\b(?:blurred\s+vision|vision\s+problems)\b',
                r'\b(?:chest\s+pain|angina|shortness\s+of\s+breath|dyspnea)\b',
                r'\b(?:nausea|vomiting|diarrhea|constipation)\b',
                r'\b(?:headache|dizziness|vertigo)\b',
                r'\b(?:fever|chills|night\s+sweats)\b'
            ],
            'conditions': [
                r'\b(?:Type\s+[12]\s+Diabetes\s+Mellitus|T1DM|T2DM|diabetes)\b',
                r'\b(?:hypertension|high\s+blood\s+pressure|HTN)\b',
                r'\b(?:hyperlipidemia|dyslipidemia|high\s+cholesterol)\b',
                r'\b(?:coronary\s+artery\s+disease|CAD|heart\s+disease)\b',
                r'\b(?:myocardial\s+infarction|heart\s+attack|MI)\b',
                r'\b(?:stroke|cerebrovascular\s+accident|CVA)\b',
                r'\b(?:obesity|overweight|BMI\s+>\s*30)\b',
                r'\b(?:asthma|COPD|emphysema)\b',
                r'\b(?:depression|anxiety|bipolar)\b'
            ],
            'procedures': [
                r'\b(?:blood\s+test|lab\s+work|laboratory\s+studies)\b',
                r'\b(?:ECG|EKG|electrocardiogram)\b',
                r'\b(?:chest\s+X-ray|CXR|radiograph)\b',
                r'\b(?:CT\s+scan|computed\s+tomography)\b',
                r'\b(?:MRI|magnetic\s+resonance\s+imaging)\b',
                r'\b(?:ultrasound|sonography|echo)\b',
                r'\b(?:biopsy|endoscopy|colonoscopy)\b'
            ],
            'medical_facility': [
                r'\b(?:Hospital|Clinic|Medical\s+Center|Diagnostic\s+Services|Laboratory)\b',
                r'\b(?:Dr|Doctor|Physician)\.\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b(?:Reg\.?\s*No\.?|Registration\s+No\.?|Reg\s*Date):?\s*[A-Z]?\d+\b'
            ],
            'patient_info': [
                r'\b(?:Name|Patient):?\s*(?:Mrs?\.?\s*)?([A-Z][A-Z\s]+)\b',
                r'\b(?:Age|Sex|Gender):?\s*(\d+)\s*(?:Year\(s\)|years?|Y)?\s*/?(?:Male|Female|M|F)?\b',
                r'\b(?:MR|Medical\s+Record|Patient\s+ID):?\s*#?\s*([A-Z]?\d+)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """Extract medical entities using pattern matching."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self._get_confidence(entity_type, match.group())
                    })
        
        # Remove duplicates and overlapping entities
        entities = self._deduplicate_entities(entities)
        return sorted(entities, key=lambda x: x['start'])
    
    def _get_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence score based on entity type and content."""
        confidence_map = {
            'date': 0.95,
            'medication': 0.90,
            'vitals': 0.95,
            'lab_values': 0.95,
            'symptoms': 0.80,
            'conditions': 0.85,
            'procedures': 0.85
        }
        
        base_confidence = confidence_map.get(entity_type, 0.75)
        
        # Adjust confidence based on text specificity
        if re.search(r'\d+', text):  # Contains numbers
            base_confidence += 0.05
        if len(text.split()) > 1:  # Multi-word
            base_confidence += 0.03
        
        return min(base_confidence, 0.98)
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return []
        
        # Sort by start position
        entities = sorted(entities, key=lambda x: x['start'])
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # Keep the one with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
