"""
AI-powered recommendation generator using DistilGPT-2.

This module provides intelligent, context-aware medical recommendations
based on patient data, medical conditions, and treatment history.
"""

import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

logger = logging.getLogger(__name__)


class AIRecommendationGenerator:
    """
    Generates intelligent medical recommendations using DistilGPT-2.
    
    This class uses a pre-trained language model to generate contextual,
    evidence-based medical recommendations based on patient data.
    """
    
    def __init__(self):
        """Initialize the AI recommendation generator."""
        self.model_name = "distilgpt2"
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the DistilGPT-2 model and tokenizer."""
        try:
            logger.info("Loading DistilGPT-2 model for recommendation generation...")
            
            # Initialize tokenizer and model
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("DistilGPT-2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DistilGPT-2 model: {str(e)}")
            # Fallback to CPU if GPU fails
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=-1,
                    pad_token_id=50256  # Default EOS token ID
                )
                logger.info("DistilGPT-2 model loaded on CPU")
            except Exception as fallback_error:
                logger.error(f"Failed to load model on CPU: {str(fallback_error)}")
                self.generator = None
    
    def generate_treatment_recommendations(
        self, 
        condition: str, 
        medication: str,
        patient_context: Dict[str, Any],
        num_recommendations: int = 5
    ) -> List[str]:
        """
        Generate treatment recommendations based on patient context.
        
        Args:
            condition: Patient's medical condition
            medication: Current medication
            patient_context: Additional patient information
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of AI-generated recommendations
        """
        if not self.generator:
            logger.warning("AI model not available, falling back to template recommendations")
            return self._fallback_treatment_recommendations(condition, medication)
        
        try:
            # Create context-rich prompt
            prompt = self._build_treatment_prompt(condition, medication, patient_context)
            
            # Generate recommendations
            generated_text = self._generate_text(prompt, max_length=200)
            
            # Parse and clean recommendations
            recommendations = self._parse_recommendations(generated_text, num_recommendations)
            
            # Validate and filter recommendations
            valid_recommendations = self._validate_medical_recommendations(recommendations)
            
            # Ensure we have enough recommendations
            if len(valid_recommendations) < num_recommendations:
                fallback_recs = self._fallback_treatment_recommendations(condition, medication)
                valid_recommendations.extend(fallback_recs[:num_recommendations - len(valid_recommendations)])
            
            return valid_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating treatment recommendations: {str(e)}")
            return self._fallback_treatment_recommendations(condition, medication)
    
    def generate_monitoring_recommendations(
        self,
        condition: str,
        risk_factors: List[Dict[str, Any]],
        patient_context: Dict[str, Any],
        num_recommendations: int = 4
    ) -> List[str]:
        """
        Generate monitoring recommendations based on patient risk profile.
        
        Args:
            condition: Primary medical condition
            risk_factors: List of identified risk factors
            patient_context: Patient demographic and clinical data
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of AI-generated monitoring recommendations
        """
        if not self.generator:
            return self._fallback_monitoring_recommendations(condition)
        
        try:
            prompt = self._build_monitoring_prompt(condition, risk_factors, patient_context)
            generated_text = self._generate_text(prompt, max_length=150)
            recommendations = self._parse_recommendations(generated_text, num_recommendations)
            valid_recommendations = self._validate_medical_recommendations(recommendations)
            
            if len(valid_recommendations) < num_recommendations:
                fallback_recs = self._fallback_monitoring_recommendations(condition)
                valid_recommendations.extend(fallback_recs[:num_recommendations - len(valid_recommendations)])            
            return valid_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating monitoring recommendations: {str(e)}")
            return self._fallback_monitoring_recommendations(condition)
    
    def generate_preventive_recommendations(
        self,
        risk_factors: List[Dict[str, Any]],
        patient_context: Dict[str, Any],
        num_recommendations: int = 4
    ) -> List[str]:
        """
        Generate preventive care recommendations based on risk factors.
        
        Args:
            risk_factors: List of identified risk factors
            patient_context: Patient demographic and clinical data
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of AI-generated preventive recommendations
        """
        if not self.generator:
            return self._fallback_preventive_recommendations(risk_factors)
        
        try:
            prompt = self._build_preventive_prompt(risk_factors, patient_context)
            generated_text = self._generate_text(prompt, max_length=150)
            recommendations = self._parse_recommendations(generated_text, num_recommendations)
            valid_recommendations = self._validate_medical_recommendations(recommendations)
            
            if len(valid_recommendations) < num_recommendations:
                fallback_recs = self._fallback_preventive_recommendations(risk_factors)
                valid_recommendations.extend(fallback_recs[:num_recommendations - len(valid_recommendations)])
            
            return valid_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating preventive recommendations: {str(e)}")
            return self._fallback_preventive_recommendations(risk_factors)
    
    def _build_treatment_prompt(
        self,
        condition: str,
        medication: str,
        patient_context: Dict[str, Any]
    ) -> str:
        """Build a medical prompt for treatment recommendations."""
        age = patient_context.get('age', 'unknown')
        gender = patient_context.get('gender', 'unknown')
        visits = patient_context.get('visit_count', 1)
        tests = patient_context.get('tests', [])
        
        prompt = f"""Medical case: Patient with {condition} prescribed {medication}. 
Demographics: Age {age}, Gender {gender}. Healthcare visits: {visits}.
Laboratory tests: {', '.join(tests) if tests else 'None recorded'}.

Clinical recommendations for ongoing management:
1. Monitor blood glucose levels and adjust medication dosage if needed
2. Schedule follow-up appointment in 3 months to assess treatment response
3. Educate patient on proper medication administration and timing
4. Implement lifestyle modifications including dietary counseling
5."""
        
        return prompt
    def _build_monitoring_prompt(
        self,
        condition: str,
        risk_factors: List[Dict[str, Any]],
        patient_context: Dict[str, Any]
    ) -> str:
        """Build a medical prompt for monitoring recommendations."""
        age = patient_context.get('age', 'unknown')
        risk_summary = ", ".join([rf.get('factor', '') for rf in risk_factors[:3]])
        
        prompt = f"""Patient monitoring case: {condition} in {age}-year-old patient.
Risk factors identified: {risk_summary}.
Clinical monitoring protocol recommendations:
1. Schedule quarterly follow-up appointments to assess disease progression
2. Perform regular laboratory studies including complete metabolic panel
3. Monitor for signs and symptoms of disease complications
4. Assess medication adherence and therapeutic response
5."""
        
        return prompt
    def _build_preventive_prompt(
        self,
        risk_factors: List[Dict[str, Any]],
        patient_context: Dict[str, Any]
    ) -> str:
        """Build a medical prompt for preventive recommendations."""
        age = patient_context.get('age', 'unknown')
        risk_summary = ", ".join([rf.get('factor', '') for rf in risk_factors[:3]])
        conditions = patient_context.get('conditions', [])
        
        prompt = f"""Preventive care planning: {age}-year-old patient.
Current conditions: {', '.join(conditions) if conditions else 'None known'}.
Identified risk factors: {risk_summary}.
Evidence-based preventive care recommendations:
1. Implement regular physical activity program appropriate for patient's fitness level
2. Provide nutritional counseling focusing on evidence-based dietary modifications
3. Schedule age-appropriate screening tests and health maintenance visits
4. Establish tobacco cessation program if applicable
5."""        
        return prompt
    
    def _generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using the AI model."""
        try:
            # Generate text with controlled parameters for better diversity
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,  # Higher temperature for more creativity
                do_sample=True,
                top_p=0.9,  # Nucleus sampling for better quality
                repetition_penalty=1.2,  # Reduce repetition
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            generated_text = outputs[0]['generated_text']            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return ""
    
    def _parse_recommendations(self, generated_text: str, num_recommendations: int) -> List[str]:
        """Parse generated text into individual recommendations."""
        recommendations = []
        
        # Split by common delimiters and numbered lists
        lines = generated_text.replace('\n', '|').replace('. ', '|').replace(': ', '|').split('|')
        
        for line in lines:
            line = line.strip()
            
            # Filter for reasonable length and content
            if 15 <= len(line) <= 200:
                # Clean up numbering and common artifacts
                line = line.replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '').replace('6.', '')
                line = line.replace('1.', '').replace('•', '').replace('-', '').strip()
                
                # Skip if starts with common non-recommendation phrases
                skip_starts = ['the', 'this', 'it', 'that', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where']
                if not any(line.lower().startswith(start + ' ') for start in skip_starts):
                    # Ensure it's a proper sentence
                    if line and line[0].isupper() and not line.endswith('.'):
                        line += '.'
                    elif line and not line[0].isupper():
                        line = line.capitalize()
                    
                    if line and line not in recommendations:  # Avoid duplicates
                        recommendations.append(line)
                        
                        if len(recommendations) >= num_recommendations:
                            break        
        return recommendations[:num_recommendations]
    
    def _validate_medical_recommendations(self, recommendations: List[str]) -> List[str]:
        """Validate and filter medical recommendations for safety and relevance."""
        valid_recommendations = []
        
        # Medical terms that indicate valid recommendations
        medical_keywords = [
            'monitor', 'check', 'test', 'screen', 'examine', 'follow-up', 'follow up',
            'medication', 'treatment', 'therapy', 'lifestyle', 'diet', 'dietary',
            'exercise', 'blood pressure', 'glucose', 'cholesterol', 'diabetes',
            'appointment', 'visit', 'consultation', 'evaluate', 'assess', 'review',
            'education', 'counseling', 'manage', 'control', 'maintain', 'improve',
            'prevent', 'reduce', 'increase', 'supplement', 'prescription', 'dosage',
            'medical', 'clinical', 'health', 'care', 'patient', 'provider',
            'screening', 'laboratory', 'lab', 'hba1c', 'hemoglobin', 'insulin'
        ]
        
        # Terms to avoid (potentially harmful or inappropriate)
        avoid_terms = [
            'stop all medication', 'discontinue all', 'emergency surgery',
            'immediate surgery', 'intensive care', 'life threatening'
        ]
        
        for rec in recommendations:
            rec_lower = rec.lower()
            
            # Check if recommendation contains medical keywords
            has_medical_content = any(keyword in rec_lower for keyword in medical_keywords)
            
            # Check if recommendation contains terms to avoid
            has_harmful_content = any(term in rec_lower for term in avoid_terms)
            
            # More lenient validation for AI-generated content
            basic_medical_check = any(word in rec_lower for word in ['patient', 'medication', 'treatment', 'monitor', 'check', 'medical', 'health', 'care'])
            
            # Validate recommendation
            if ((has_medical_content or basic_medical_check) and 
                not has_harmful_content and 
                len(rec) > 10 and 
                len(rec) < 200):
                
                # Ensure proper capitalization
                if not rec[0].isupper():
                    rec = rec.capitalize()
                
                # Ensure proper punctuation
                if not rec.endswith('.'):
                    rec = rec + '.'
                    
                valid_recommendations.append(rec)
        
        return valid_recommendations
    
    def _fallback_treatment_recommendations(self, condition: str, medication: str) -> List[str]:
        """Fallback recommendations when AI model is unavailable."""
        base_recommendations = [
            "Continue monitoring patient response to treatment",
            "Schedule regular follow-up appointments",
            "Track medication adherence and side effects",
            "Monitor vital signs and clinical parameters"
        ]
        
        # Add condition-specific recommendations
        if "diabetes" in condition.lower():
            base_recommendations.extend([
                "Monitor HbA1c levels every 3-6 months",
                "Check blood glucose levels regularly",
                "Screen for diabetic complications annually",
                "Encourage lifestyle modifications including diet and exercise"
            ])
        elif "hypertension" in condition.lower():
            base_recommendations.extend([
                "Monitor blood pressure regularly",
                "Check for end-organ damage",
                "Assess cardiovascular risk factors",
                "Promote sodium restriction and weight management"
            ])
        
        return base_recommendations
    
    def _fallback_monitoring_recommendations(self, condition: str) -> List[str]:
        """Fallback monitoring recommendations."""
        base_monitoring = [
            "Schedule regular clinical assessments",
            "Monitor treatment response and side effects",
            "Track relevant biomarkers and lab values",
            "Assess for disease progression"
        ]
        
        if "diabetes" in condition.lower():
            base_monitoring.extend([
                "Monitor HbA1c every 3-6 months",
                "Annual eye and foot examinations",
                "Regular kidney function tests",
                "Lipid profile monitoring"
            ])
        
        return base_monitoring
    
    def _fallback_preventive_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Fallback preventive recommendations."""
        base_preventive = [
            "Maintain regular physical activity",
            "Follow a balanced, nutritious diet",
            "Schedule routine health screenings",
            "Avoid tobacco and limit alcohol consumption"
        ]
        
        # Add risk-specific recommendations
        if any("diabetes" in rf.get("factor", "") for rf in risk_factors):
            base_preventive.extend([
                "Monitor blood glucose levels",
                "Maintain healthy weight",
                "Regular cardiovascular screening"
            ])
        
        return base_preventive


# Global instance for reuse
_ai_generator = None


@lru_cache(maxsize=1)
def get_ai_recommendation_generator() -> AIRecommendationGenerator:
    """
    Get singleton instance of AI recommendation generator.
    
    Returns:
        AIRecommendationGenerator instance
    """
    global _ai_generator
    if _ai_generator is None:
        _ai_generator = AIRecommendationGenerator()
    return _ai_generator
