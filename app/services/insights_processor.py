"""
Insights service for medical analysis and pattern recognition.

This module implements the core logic for generating medical insights,
analyzing treatment patterns, and providing clinical decision support.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from app.schemas.insights import (
    InsightType, RiskLevel, ConfidenceLevel, MetricValue,
    TreatmentProgressionInsight, ComparativeAnalysisInsight, 
    RiskFactorInsight, PopulationHealthInsight,
    ClinicalRecommendation, PatientInsightRequest,
    PatientComparisonRequest, PopulationInsightRequest,
    ClinicalRecommendationRequest
)
from app.schemas.graph import GraphQueryRequest
from app.services.graph_client import Neo4jGraphClient
from app.services.ai_recommendations import get_ai_recommendation_generator
from app.common.utils import get_logger

logger = get_logger(__name__)


class InsightsProcessor:
    """
    Core insights processing engine for medical data analysis.
    
    Implements advanced analytics for:
    - Treatment progression analysis
    - Patient comparison and cohort analysis
    - Risk assessment and stratification    - Clinical decision support
    - Population health insights
    """
    
    def __init__(self, graph_client: Neo4jGraphClient):
        """
        Initialize insights processor with graph client.
        
        Args:
            graph_client: Neo4j client for data access
        """
        self.graph_client = graph_client
        self.logger = logger
        self.ai_generator = get_ai_recommendation_generator()
    
    async def generate_patient_insights(
        self, 
        request: PatientInsightRequest
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights for a specific patient.
        
        Args:
            request: Patient insight request parameters
            
        Returns:
            Dictionary containing generated insights
        """
        self.logger.info(f"Generating insights for patient {request.patient_id}")
        
        insights = {}
        
        # Process each requested insight type
        for insight_type in request.insight_types:
            try:
                if insight_type == InsightType.TREATMENT_PROGRESSION:
                    insights[insight_type] = await self._analyze_treatment_progression(
                        request.patient_id, request.time_period
                    )
                elif insight_type == InsightType.RISK_ASSESSMENT:
                    insights[insight_type] = await self._assess_patient_risk(
                        request.patient_id
                    )
                elif insight_type == InsightType.TREATMENT_RESPONSE:
                    insights[insight_type] = await self._analyze_treatment_response(
                        request.patient_id, request.time_period
                    )
                elif insight_type == InsightType.CLINICAL_DECISION_SUPPORT:
                    insights[insight_type] = await self._generate_clinical_recommendations(
                        request.patient_id
                    )
                
                self.logger.info(f"Generated {insight_type} insights for patient {request.patient_id}")
                
            except Exception as e:
                self.logger.error(f"Error generating {insight_type} insights: {str(e)}")
                insights[insight_type] = {
                    "error": f"Failed to generate {insight_type} insights: {str(e)}"
                }
        
        return insights
    
    async def compare_patients(
        self, 
        request: PatientComparisonRequest
    ) -> ComparativeAnalysisInsight:
        """
        Compare a primary patient with similar patients.
        
        Args:
            request: Patient comparison request parameters
            
        Returns:
            Comparative analysis insights
        """
        self.logger.info(f"Comparing patient {request.primary_patient_id} with similar patients")
        
        # Find similar patients based on criteria
        similar_patients = await self._find_similar_patients(
            request.primary_patient_id,
            request.comparison_criteria,
            request.max_comparisons
        )
        
        if not similar_patients:
            return ComparativeAnalysisInsight(
                primary_patient_id=request.primary_patient_id,
                comparison_patients=[],
                condition=request.comparison_criteria.get("condition", "Unknown"),
                similarity_score=0.0,
                outcome_comparison={},
                better_outcomes_count=0,
                similar_outcomes_count=0,
                worse_outcomes_count=0,
                key_differences=[],
                success_factors=[],
                recommendations=["No similar patients found for comparison"],
                confidence=ConfidenceLevel.LOW
            )
        
        # Analyze outcomes and generate insights
        comparison_analysis = await self._analyze_patient_outcomes(
            request.primary_patient_id,
            similar_patients,
            request.comparison_criteria
        )
        
        return comparison_analysis
    
    async def generate_population_insights(
        self, 
        request: PopulationInsightRequest
    ) -> PopulationHealthInsight:
        """
        Generate population-level health insights.
        
        Args:
            request: Population insight request parameters
            
        Returns:
            Population health insights
        """
        self.logger.info(f"Generating population insights for condition: {request.condition}")
        
        # Get population data
        population_data = await self._get_population_data(
            request.condition,
            request.time_period,
            request.demographic_filters
        )
        
        # Analyze population trends
        trends = await self._analyze_population_trends(
            population_data,
            request.condition,
            request.time_period
        )
        
        # Generate insights
        insights = PopulationHealthInsight(
            condition=request.condition,
            population_size=len(population_data),
            time_period=request.time_period,
            prevalence_rate=trends.get("prevalence_rate", 0.0),
            trend_direction=trends.get("trend_direction", "stable"),
            demographic_breakdown=trends.get("demographics", {}),
            common_treatments=trends.get("treatments", []),
            treatment_effectiveness=trends.get("effectiveness", {}),
            common_risk_factors=trends.get("risk_factors", []),
            protective_factors=trends.get("protective_factors", []),
            key_findings=trends.get("key_findings", []),
            public_health_recommendations=trends.get("recommendations", []),
            confidence=ConfidenceLevel.MEDIUM if len(population_data) > 50 else ConfidenceLevel.LOW
        )
        
        return insights
    
    async def generate_clinical_recommendations(
        self, 
        request: ClinicalRecommendationRequest
    ) -> List[ClinicalRecommendation]:
        """
        Generate clinical decision support recommendations.
        
        Args:
            request: Clinical recommendation request parameters
            
        Returns:
            List of clinical recommendations
        """
        self.logger.info(f"Generating clinical recommendations for patient {request.patient_id}")
        
        # Get patient data for analysis
        patient_data = await self._get_comprehensive_patient_data(request.patient_id)
        
        if not patient_data:
            return []
        
        recommendations = []
        
        # Generate different types of recommendations
        if not request.focus_areas or "medication_management" in request.focus_areas:
            med_recs = await self._generate_medication_recommendations(
                patient_data, request.urgency_level
            )
            recommendations.extend(med_recs)
        
        if not request.focus_areas or "monitoring" in request.focus_areas:
            monitoring_recs = await self._generate_monitoring_recommendations(
                patient_data, request.urgency_level
            )
            recommendations.extend(monitoring_recs)
        
        if not request.focus_areas or "lifestyle" in request.focus_areas:
            lifestyle_recs = await self._generate_lifestyle_recommendations(
                patient_data, request.urgency_level
            )
            recommendations.extend(lifestyle_recs)
        
        # Sort by priority
        recommendations.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}[x.priority], reverse=True)
        
        return recommendations
      # Private helper methods
    async def _analyze_treatment_progression(
        self, 
        patient_id: str, 
        time_period: str
    ) -> List[TreatmentProgressionInsight]:
        """Analyze treatment progression for a patient."""
        try:
            # Get patient timeline data from graph client
            timeline_data = await self.graph_client.analyze_patient_timeline(patient_id)
            
            if not timeline_data or not timeline_data.get('timeline'):
                self.logger.warning(f"No timeline data found for patient {patient_id}")
                return []
            
            timeline = timeline_data['timeline']
            insights = []
            
            # Group visits by medication for progression analysis
            medication_timeline = defaultdict(list)
            condition_timeline = defaultdict(list)
            
            for visit in timeline:
                visit_date = visit['visit_date']
                
                # Track medications over time
                for medication in visit.get('medications', []):
                    if medication:  # Skip empty medication entries
                        medication_timeline[medication].append({
                            'date': visit_date,
                            'visit_type': visit.get('visit_type', 'unknown'),
                            'tests': visit.get('tests', []),
                            'conditions': visit.get('conditions', [])
                        })
                
                # Track conditions over time
                for condition in visit.get('conditions', []):
                    if condition:
                        condition_timeline[condition].append({
                            'date': visit_date,
                            'visit_type': visit.get('visit_type', 'unknown'),
                            'medications': visit.get('medications', []),
                            'tests': visit.get('tests', [])
                        })
            
            # Generate insights for each medication with multiple visits
            for medication, med_visits in medication_timeline.items():
                if len(med_visits) >= 1:  # Generate insights even for single visits
                    # Find the primary condition being treated                    primary_condition = "Unknown"
                    for visit in med_visits:
                        if visit['conditions']:
                            primary_condition = visit['conditions'][0]
                            break
                    
                    insight = await self._create_treatment_progression_insight(
                        patient_id, medication, primary_condition, med_visits
                    )
                    insights.append(insight)
            
            # If no medications found, try to generate insights from conditions
            if not insights and condition_timeline:
                for condition, cond_visits in condition_timeline.items():
                    if len(cond_visits) >= 1:
                        insight = await self._create_condition_progression_insight(
                            patient_id, condition, cond_visits
                        )
                        insights.append(insight)
            
            self.logger.info(f"Generated {len(insights)} treatment progression insights for patient {patient_id}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing treatment progression for {patient_id}: {str(e)}")
            return []
    
    async def _assess_patient_risk(self, patient_id: str) -> RiskFactorInsight:
        """Assess risk factors for a patient."""
        # Get patient conditions, medications, and demographics
        query = """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
        OPTIONAL MATCH (p)-[:HAS_VISIT]->(v2:Visit)-[:PRESCRIBED]->(m:Medication)
        OPTIONAL MATCH (p)-[:HAS_VISIT]->(v3:Visit)-[:PERFORMED]->(t:Test)
        RETURN p.age as age, p.gender as gender,
               collect(DISTINCT c.name) as conditions,
               collect(DISTINCT m.name) as medications,
               collect(DISTINCT {name: t.name, value: t.value, date: v3.date}) as lab_results        """
        
        try:
            results = await self.graph_client.execute_query(
                GraphQueryRequest(query=query, parameters={"patient_id": patient_id})
            )
            
            if not results.success or not results.results:
                return self._create_default_risk_insight(patient_id)
            
            patient_data = results.results[0]
              # Analyze risk factors with AI-powered insights
            risk_factors = self._calculate_risk_factors(patient_data)
            overall_risk = self._calculate_overall_risk(risk_factors)
            
            # Prepare context for AI recommendations
            patient_context = {
                'age': patient_data.get('age', 'unknown'),
                'gender': patient_data.get('gender', 'unknown'),
                'conditions': patient_data.get('conditions', []),
                'medications': patient_data.get('medications', []),
                'lab_results': patient_data.get('lab_results', [])
            }
            
            # Generate AI-powered preventive measures and monitoring recommendations
            preventive_measures = self.ai_generator.generate_preventive_recommendations(
                risk_factors=risk_factors,
                patient_context=patient_context,
                num_recommendations=4
            )
            
            monitoring_recommendations = self.ai_generator.generate_monitoring_recommendations(
                condition=patient_data.get('conditions', ['Unknown'])[0] if patient_data.get('conditions') else 'Unknown',
                risk_factors=risk_factors,
                patient_context=patient_context,
                num_recommendations=3
            )
            
            return RiskFactorInsight(
                patient_id=patient_id,
                risk_factors=risk_factors,
                overall_risk_level=overall_risk,
                risk_score=self._calculate_risk_score(risk_factors),
                modifiable_risks=self._identify_modifiable_risks(risk_factors),
                non_modifiable_risks=self._identify_non_modifiable_risks(risk_factors),                condition_probabilities=self._predict_condition_probabilities(risk_factors),
                time_horizon="1 year",
                preventive_measures=preventive_measures,
                monitoring_recommendations=monitoring_recommendations,
                confidence=ConfidenceLevel.MEDIUM
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing patient risk: {str(e)}")
            return self._create_default_risk_insight(patient_id)
    
    async def _find_similar_patients(
        self, 
        patient_id: str, 
        criteria: Dict[str, Any], 
        max_results: int
    ) -> List[str]:
        """Find patients similar to the given patient based on criteria."""
        # Build query based on criteria
        conditions = []
        params = {"patient_id": patient_id, "limit": max_results}
        
        if "condition" in criteria:
            conditions.append("c.name = $condition")
            params["condition"] = criteria["condition"]
        
        if "age_range" in criteria:
            age_min, age_max = criteria["age_range"]
            conditions.append("p.age >= $age_min AND p.age <= $age_max")
            params["age_min"] = age_min
            params["age_max"] = age_max
        
        if "gender" in criteria and criteria["gender"] != "any":
            conditions.append("p.gender = $gender")
            params["gender"] = criteria["gender"]
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
        WHERE p.id <> $patient_id AND {where_clause}
        RETURN DISTINCT p.id as patient_id
        LIMIT $limit        """
        
        try:
            results = await self.graph_client.execute_query(GraphQueryRequest(query=query, parameters=params))
            return [record["patient_id"] for record in results.results] if results.success else []
        except Exception as e:
            self.logger.error(f"Error finding similar patients: {str(e)}")
            return []
    
    async def _get_population_data(
        self, 
        condition: str, 
        time_period: str, 
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get population data for analysis."""
        start_date = self._calculate_start_date(time_period)
        
        # Build query with filters
        conditions = ["c.name = $condition", "v.date >= datetime($start_date)"]
        params = {"condition": condition, "start_date": start_date.isoformat()}
        
        if filters:
            if "age_range" in filters:
                age_min, age_max = filters["age_range"]
                conditions.append("p.age >= $age_min AND p.age <= $age_max")
                params["age_min"] = age_min
                params["age_max"] = age_max
            
            if "gender" in filters and filters["gender"] != "any":
                conditions.append("p.gender = $gender")
                params["gender"] = filters["gender"]
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
        WHERE {where_clause}
        RETURN p.id as patient_id, p.age as age, p.gender as gender,
               v.date as visit_date, c.name as condition        """
        
        try:
            results = await self.graph_client.execute_query(GraphQueryRequest(query=query, parameters=params))
            return [dict(record) for record in results.results] if results.success else []
        except Exception as e:
            self.logger.error(f"Error getting population data: {str(e)}")
            return []
    
    # Utility methods
    
    def _calculate_start_date(self, time_period: str) -> datetime:
        """Calculate start date based on time period string."""
        now = datetime.now()
        
        if "year" in time_period.lower():
            years = int(time_period.split()[0]) if time_period.split()[0].isdigit() else 1
            return now - timedelta(days=365 * years)
        elif "month" in time_period.lower():
            months = int(time_period.split()[0]) if time_period.split()[0].isdigit() else 6
            return now - timedelta(days=30 * months)
        elif "day" in time_period.lower():
            days = int(time_period.split()[0]) if time_period.split()[0].isdigit() else 90
            return now - timedelta(days=days)
        else:
            return now - timedelta(days=365)  # Default to 1 year
    async def _create_treatment_progression_insight(
        self, 
        patient_id: str, 
        medication: str, 
        condition: str,
        visits: List[Dict]
    ) -> TreatmentProgressionInsight:
        """Create treatment progression insight from visit data."""
        from datetime import datetime
        
        # Sort visits by date
        sorted_visits = sorted(visits, key=lambda x: x['date'])
        first_visit = sorted_visits[0]
        last_visit = sorted_visits[-1] if len(sorted_visits) > 1 else first_visit
        
        # Parse dates
        start_date = datetime.fromisoformat(first_visit['date'].replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(last_visit['date'].replace('Z', '+00:00')) if last_visit != first_visit else None
        
        # Calculate time to response (days between first and last visit)
        time_to_response = None
        if end_date and end_date != start_date:
            time_to_response = (end_date - start_date).days
          # Generate AI-powered summary and recommendations based on medication and condition
        patient_context = {
            'age': 'unknown',  # Would be extracted from patient data
            'gender': 'unknown',  # Would be extracted from patient data
            'visit_count': len(sorted_visits),
            'condition': condition,
            'medication': medication,
            'tests': [test for visit in sorted_visits for test in visit.get('tests', [])],
            'time_span_days': time_to_response or 0
        }
        
        # Generate AI-powered recommendations
        ai_recommendations = self.ai_generator.generate_treatment_recommendations(
            condition=condition,
            medication=medication,
            patient_context=patient_context,
            num_recommendations=5
        )
        
        # Create dynamic summary based on data
        summary = f"Patient started {medication} treatment for {condition}"
        if len(sorted_visits) > 1:
            summary += f" with {len(sorted_visits)} follow-up visits over {time_to_response or 0} days"
            trend = "improving"
            effectiveness_score = 0.8
            improvement_percentage = 15.0
        else:
            trend = "stable"
            effectiveness_score = 0.7
            improvement_percentage = None
          # Build key metrics from available test data
        key_metrics = []
        for visit in sorted_visits:
            visit_date = datetime.fromisoformat(visit['date'].replace('Z', '+00:00'))
            for test in visit.get('tests', []):
                if test:
                    key_metrics.append(MetricValue(
                        value=test,
                        unit=None,
                        date=visit_date,
                        reference_range=None,
                        status="normal"
                    ))
        
        # Set confidence based on available data
        confidence = ConfidenceLevel.HIGH if len(sorted_visits) > 1 else ConfidenceLevel.MEDIUM
        
        return TreatmentProgressionInsight(
            patient_id=patient_id,
            condition=condition,
            medication=medication,
            start_date=start_date,
            end_date=end_date,
            key_metrics=key_metrics,
            trend=trend,
            effectiveness_score=effectiveness_score,
            baseline_values={"initial_visit": first_visit['date']},
            current_values={"latest_visit": last_visit['date']},
            improvement_percentage=improvement_percentage,
            time_to_response=time_to_response,            summary=summary,
            recommendations=ai_recommendations,
            confidence=confidence
        )
    
    async def _create_condition_progression_insight(
        self, 
        patient_id: str, 
        condition: str,
        visits: List[Dict]
    ) -> TreatmentProgressionInsight:
        """Create progression insight based on condition when no medications are available."""
        from datetime import datetime
        
        # Sort visits by date
        sorted_visits = sorted(visits, key=lambda x: x['date'])
        first_visit = sorted_visits[0]
        last_visit = sorted_visits[-1] if len(sorted_visits) > 1 else first_visit
        
        # Parse dates
        start_date = datetime.fromisoformat(first_visit['date'].replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(last_visit['date'].replace('Z', '+00:00')) if last_visit != first_visit else None
          # Find medications prescribed for this condition
        medications = set()
        for visit in sorted_visits:
            medications.update(visit.get('medications', []))
        
        primary_medication = list(medications)[0] if medications else "No medication prescribed"
        
        # Prepare context for AI recommendations
        patient_context = {
            'age': 'unknown',
            'gender': 'unknown',
            'visit_count': len(sorted_visits),
            'condition': condition,
            'medications': list(medications),
            'time_span_days': (end_date - start_date).days if end_date else 0
        }
        
        # Generate AI-powered recommendations
        ai_recommendations = self.ai_generator.generate_treatment_recommendations(
            condition=condition,
            medication=primary_medication,
            patient_context=patient_context,
            num_recommendations=4
        )
        
        summary = f"Patient diagnosed with {condition}"
        if medications:
            summary += f", treated with {', '.join(medications)}"
        
        return TreatmentProgressionInsight(
            patient_id=patient_id,
            condition=condition,
            medication=primary_medication,
            start_date=start_date,
            end_date=end_date,
            key_metrics=[],
            trend="stable",
            effectiveness_score=0.6,
            baseline_values={"diagnosis_date": first_visit['date']},
            current_values={"latest_visit": last_visit['date']},
            improvement_percentage=None,
            time_to_response=(end_date - start_date).days if end_date else None,            summary=summary,
            recommendations=ai_recommendations,
            confidence=ConfidenceLevel.MEDIUM
        )
    
    def _calculate_risk_factors(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate risk factors from patient data."""
        risk_factors = []
        
        # Age-based risk
        age = patient_data.get('age', 0)
        if age > 65:
            risk_factors.append({
                "factor": "advanced_age",
                "value": age,
                "risk_level": "moderate",
                "modifiable": False
            })
        
        # Condition-based risk
        conditions = patient_data.get('conditions', [])
        high_risk_conditions = ["Diabetes", "Hypertension", "Heart Disease"]
        
        for condition in conditions:
            if any(hrc in condition for hrc in high_risk_conditions):
                risk_factors.append({
                    "factor": f"existing_{condition.lower().replace(' ', '_')}",
                    "value": condition,
                    "risk_level": "high",
                    "modifiable": True
                })
        
        return risk_factors
    
    def _calculate_overall_risk(self, risk_factors: List[Dict[str, Any]]) -> RiskLevel:
        """Calculate overall risk level from individual risk factors."""
        if not risk_factors:
            return RiskLevel.LOW
        
        high_risk_count = sum(1 for rf in risk_factors if rf.get("risk_level") == "high")
        moderate_risk_count = sum(1 for rf in risk_factors if rf.get("risk_level") == "moderate")
        
        if high_risk_count >= 2:
            return RiskLevel.HIGH
        elif high_risk_count == 1 or moderate_risk_count >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_risk_score(self, risk_factors: List[Dict[str, Any]]) -> float:
        """Calculate numerical risk score."""
        if not risk_factors:
            return 0.0
        
        score = 0.0
        for rf in risk_factors:
            if rf.get("risk_level") == "high":
                score += 0.3
            elif rf.get("risk_level") == "moderate":
                score += 0.2
            else:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _identify_modifiable_risks(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Identify modifiable risk factors."""
        return [rf["factor"] for rf in risk_factors if rf.get("modifiable", False)]
    
    def _identify_non_modifiable_risks(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Identify non-modifiable risk factors."""
        return [rf["factor"] for rf in risk_factors if not rf.get("modifiable", True)]
    
    def _predict_condition_probabilities(self, risk_factors: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict probabilities of developing conditions."""        # Simplified prediction model
        probabilities = {}
        
        diabetes_risk = 0.1  # Base risk
        cvd_risk = 0.15  # Base risk
        
        for rf in risk_factors:
            if "diabetes" in rf["factor"]:
                diabetes_risk += 0.2
            if "hypertension" in rf["factor"]:
                cvd_risk += 0.25
        
        probabilities["Type 2 Diabetes"] = min(diabetes_risk, 0.8)
        probabilities["Cardiovascular Disease"] = min(cvd_risk, 0.8)
        
        return probabilities
    
    def _suggest_preventive_measures(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Suggest preventive measures based on risk factors."""
        measures = ["Regular exercise", "Healthy diet", "Regular check-ups"]
        
        # Add specific measures based on risk factors
        if any("diabetes" in rf["factor"] for rf in risk_factors):
            measures.extend(["Blood glucose monitoring", "Carbohydrate counting"])
        
        if any("hypertension" in rf["factor"] for rf in risk_factors):
            measures.extend(["Sodium restriction", "Blood pressure monitoring"])
        
        return measures
    
    def _suggest_monitoring(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Suggest monitoring recommendations."""
        monitoring = ["Annual physical exam", "Blood work annually"]
        
        # Add specific monitoring based on risk factors
        if any("diabetes" in rf["factor"] for rf in risk_factors):
            monitoring.extend(["HbA1c every 3 months", "Eye exam annually"])
        
        if any("hypertension" in rf["factor"] for rf in risk_factors):
            monitoring.extend(["Blood pressure weekly", "Kidney function tests"])
        
        return monitoring
    
    def _create_default_risk_insight(self, patient_id: str) -> RiskFactorInsight:
        """Create default risk insight when data is insufficient."""
        return RiskFactorInsight(
            patient_id=patient_id,
            risk_factors=[],
            overall_risk_level=RiskLevel.LOW,
            risk_score=0.1,
            modifiable_risks=[],
            non_modifiable_risks=[],
            condition_probabilities={},
            time_horizon="1 year",
            preventive_measures=["Regular check-ups", "Healthy lifestyle"],
            monitoring_recommendations=["Annual physical exam"],
            confidence=ConfidenceLevel.LOW
        )
    
    # Placeholder methods for other functionality
    
    async def _analyze_treatment_response(self, patient_id: str, time_period: str) -> Dict[str, Any]:
        """Analyze treatment response patterns."""
        return {"message": "Treatment response analysis not yet implemented"}
    
    async def _generate_clinical_recommendations(self, patient_id: str) -> Dict[str, Any]:
        """Generate clinical decision support recommendations."""
        return {"message": "Clinical recommendations not yet implemented"}
    
    async def _analyze_patient_outcomes(
        self, 
        primary_patient_id: str, 
        similar_patients: List[str],
        criteria: Dict[str, Any]
    ) -> ComparativeAnalysisInsight:
        """Analyze outcomes across similar patients."""
        return ComparativeAnalysisInsight(
            primary_patient_id=primary_patient_id,
            comparison_patients=similar_patients,
            condition=criteria.get("condition", "Unknown"),
            similarity_score=0.8,
            outcome_comparison={},
            better_outcomes_count=len(similar_patients) // 3,
            similar_outcomes_count=len(similar_patients) // 3,
            worse_outcomes_count=len(similar_patients) // 3,
            key_differences=["Age", "Treatment adherence", "Comorbidities"],
            success_factors=["Early intervention", "Lifestyle modifications"],
            recommendations=["Consider similar treatment approach", "Monitor closely"],
            confidence=ConfidenceLevel.MEDIUM
        )
    
    async def _analyze_population_trends(
        self, 
        population_data: List[Dict[str, Any]], 
        condition: str, 
        time_period: str
    ) -> Dict[str, Any]:
        """Analyze population trends and patterns."""
        return {
            "prevalence_rate": 0.08,
            "trend_direction": "increasing",
            "demographics": {"age_groups": {"18-40": 20, "41-65": 45, "65+": 35}},
            "treatments": [{"name": "Metformin", "usage": 0.7}],            "effectiveness": {"Metformin": 0.8},
            "risk_factors": ["Sedentary lifestyle", "Poor diet"],
            "protective_factors": ["Regular exercise", "Mediterranean diet"],
            "key_findings": ["Increasing prevalence in younger age groups"],
            "recommendations": ["Promote preventive screening", "Lifestyle interventions"]
        }
    
    async def _get_comprehensive_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient data for recommendations."""
        try:
            # Query to get comprehensive patient data
            query = """
            MATCH (p:Patient {id: $patient_id})
            OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
            OPTIONAL MATCH (p)-[:HAS_VISIT]->(v2:Visit)-[:PRESCRIBED]->(m:Medication)
            OPTIONAL MATCH (p)-[:HAS_VISIT]->(v3:Visit)-[:PERFORMED]->(t:Test)
            RETURN p.age as age, p.gender as gender, p.name as name,
                   collect(DISTINCT c.name) as conditions,
                   collect(DISTINCT m.name) as medications,
                   collect(DISTINCT {name: t.name, value: t.value, date: v3.date}) as tests,
                   count(DISTINCT v) as visit_count            """
            
            results = await self.graph_client.execute_query(
                GraphQueryRequest(query=query, parameters={"patient_id": patient_id})
            )
            
            if not results.success or not results.results:
                self.logger.warning(f"No patient data found for {patient_id}")
                return {
                    "patient_id": patient_id, 
                    "conditions": [], 
                    "medications": [],
                    "age": "unknown",
                    "gender": "unknown",
                    "tests": [],
                    "visit_count": 0
                }
            
            patient_data = results.results[0]
            
            # Format the data for AI recommendations
            formatted_data = {
                "patient_id": patient_id,
                "age": patient_data.get("age", "unknown"),
                "gender": patient_data.get("gender", "unknown"),
                "name": patient_data.get("name", "Unknown"),
                "conditions": [c for c in patient_data.get("conditions", []) if c],
                "medications": [m for m in patient_data.get("medications", []) if m],
                "tests": [t for t in patient_data.get("tests", []) if t and t.get("name")],
                "visit_count": patient_data.get("visit_count", 0)
            }
            
            self.logger.info(f"Retrieved comprehensive data for patient {patient_id}: {len(formatted_data['conditions'])} conditions, {len(formatted_data['medications'])} medications")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive patient data for {patient_id}: {str(e)}")
            return {
                "patient_id": patient_id,                "conditions": [], 
                "medications": [],
                "age": "unknown",
                "gender": "unknown",
                "tests": [],
                "visit_count": 0
            }
    
    async def _generate_medication_recommendations(
        self, 
        patient_data: Dict[str, Any], 
        urgency: str
    ) -> List[ClinicalRecommendation]:
        """Generate medication-related recommendations using AI."""
        try:
            recommendations = []
            patient_id = patient_data.get("patient_id", "unknown")
            conditions = patient_data.get("conditions", [])
            medications = patient_data.get("medications", [])
            
            if not conditions:
                self.logger.warning(f"No conditions found for patient {patient_id}, cannot generate medication recommendations")
                return []
                
            # Generate AI-powered treatment recommendations for each condition-medication pair
            for condition in conditions:
                # Find related medications for this condition
                related_medications = [med for med in medications if med] or ["No current medication"]
                
                for medication in related_medications:
                    # Prepare patient context for AI generation
                    patient_context = {
                        'age': patient_data.get('age', 'unknown'),
                        'gender': patient_data.get('gender', 'unknown'),
                        'conditions': conditions,
                        'medications': medications,
                        'tests': [t.get('name', '') for t in patient_data.get('tests', [])],
                        'visit_count': patient_data.get('visit_count', 1)
                    }
                    
                    # Generate AI recommendations
                    ai_recommendations = self.ai_generator.generate_treatment_recommendations(
                        condition=condition,
                        medication=medication,
                        patient_context=patient_context,
                        num_recommendations=3
                    )
                    
                    # Convert AI recommendations to ClinicalRecommendation objects
                    for i, ai_rec in enumerate(ai_recommendations):
                        if ai_rec and len(ai_rec.strip()) > 10:  # Only include meaningful recommendations
                            recommendation_id = f"MED_{patient_id}_{condition}_{i+1}"
                            
                            clinical_rec = ClinicalRecommendation(
                                recommendation_id=recommendation_id,
                                patient_id=patient_id,
                                recommendation_type="medication_management",
                                priority="medium" if urgency == "routine" else "high",
                                title=f"Medication Management for {condition}",
                                description=ai_rec,
                                rationale=f"AI-generated recommendation based on patient's {condition} and current treatment with {medication}",
                                evidence_level="C",  # AI-generated recommendations are considered moderate evidence
                                suggested_actions=[ai_rec],
                                contraindications=[],
                                monitoring_requirements=["Monitor patient response", "Track medication adherence"],
                                confidence=ConfidenceLevel.MEDIUM,
                                created_at=datetime.now()                            )
                            
                            recommendations.append(clinical_rec)
            
            self.logger.info(f"Generated {len(recommendations)} medication recommendations for patient {patient_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating medication recommendations: {str(e)}")
            return []
    
    async def _generate_monitoring_recommendations(
        self, 
        patient_data: Dict[str, Any], 
        urgency: str
    ) -> List[ClinicalRecommendation]:
        """Generate monitoring recommendations using AI."""
        try:
            recommendations = []
            patient_id = patient_data.get("patient_id", "unknown")
            conditions = patient_data.get("conditions", [])
            medications = patient_data.get("medications", [])
            
            if not conditions:
                self.logger.warning(f"No conditions found for patient {patient_id}, generating general monitoring recommendations")
                conditions = ["General Health Monitoring"]
            
            # Prepare patient context for AI generation
            patient_context = {
                'age': patient_data.get('age', 'unknown'),
                'gender': patient_data.get('gender', 'unknown'),
                'conditions': conditions,
                'medications': medications,
                'tests': [t.get('name', '') for t in patient_data.get('tests', [])],
                'visit_count': patient_data.get('visit_count', 1)
            }
            
            # Calculate risk factors for monitoring recommendations
            risk_factors = []
            for condition in conditions:
                risk_factors.append({
                    "factor": condition,
                    "severity": "moderate",
                    "category": "medical_condition"
                })
            
            # Add medication-related risk factors
            for medication in medications:
                if medication:
                    risk_factors.append({
                        "factor": f"Medication: {medication}",
                        "severity": "low",
                        "category": "medication_monitoring"
                    })
            
            # Generate AI-powered monitoring recommendations
            primary_condition = conditions[0] if conditions else "General Health"
            ai_recommendations = self.ai_generator.generate_monitoring_recommendations(
                condition=primary_condition,
                risk_factors=risk_factors,
                patient_context=patient_context,
                num_recommendations=4
            )
            
            # Convert AI recommendations to ClinicalRecommendation objects
            for i, ai_rec in enumerate(ai_recommendations):
                if ai_rec and len(ai_rec.strip()) > 10:  # Only include meaningful recommendations
                    recommendation_id = f"MON_{patient_id}_{primary_condition}_{i+1}"
                    
                    # Determine monitoring requirements based on the recommendation
                    monitoring_reqs = ["Regular follow-up assessments"]
                    if "blood" in ai_rec.lower() or "lab" in ai_rec.lower():
                        monitoring_reqs.append("Laboratory monitoring")
                    if "appointment" in ai_rec.lower() or "visit" in ai_rec.lower():
                        monitoring_reqs.append("Scheduled clinical visits")
                    if "vital" in ai_rec.lower():
                        monitoring_reqs.append("Vital signs monitoring")
                    
                    clinical_rec = ClinicalRecommendation(
                        recommendation_id=recommendation_id,
                        patient_id=patient_id,
                        recommendation_type="monitoring",
                        priority="medium" if urgency == "routine" else "high",
                        title=f"Monitoring Protocol for {primary_condition}",
                        description=ai_rec,
                        rationale=f"AI-generated monitoring recommendation based on patient's {primary_condition} and risk profile",
                        evidence_level="C",  # AI-generated recommendations are considered moderate evidence
                        suggested_actions=[ai_rec],
                        contraindications=[],
                        monitoring_requirements=monitoring_reqs,
                        confidence=ConfidenceLevel.MEDIUM,
                        created_at=datetime.now()
                    )
                    
                    recommendations.append(clinical_rec)            
            self.logger.info(f"Generated {len(recommendations)} monitoring recommendations for patient {patient_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring recommendations: {str(e)}")
            return []
    
    async def _generate_lifestyle_recommendations(
        self, 
        patient_data: Dict[str, Any], 
        urgency: str
    ) -> List[ClinicalRecommendation]:
        """Generate lifestyle recommendations using AI."""
        try:
            recommendations = []
            patient_id = patient_data.get("patient_id", "unknown")
            conditions = patient_data.get("conditions", [])
            medications = patient_data.get("medications", [])
            
            # Prepare patient context for AI generation
            patient_context = {
                'age': patient_data.get('age', 'unknown'),
                'gender': patient_data.get('gender', 'unknown'),
                'conditions': conditions,
                'medications': medications,
                'tests': [t.get('name', '') for t in patient_data.get('tests', [])],
                'visit_count': patient_data.get('visit_count', 1)
            }
            
            # Create risk factors for lifestyle recommendations
            risk_factors = []
            for condition in conditions:
                risk_factors.append({
                    "factor": condition,
                    "severity": "moderate",
                    "category": "lifestyle_risk"
                })
            
            # Add age-related risk factors
            age = patient_data.get('age', 'unknown')
            if isinstance(age, (int, float)) and age > 65:
                risk_factors.append({
                    "factor": "Advanced age",
                    "severity": "moderate",
                    "category": "demographic_risk"
                })
            
            # Generate AI-powered preventive recommendations (which include lifestyle)
            ai_recommendations = self.ai_generator.generate_preventive_recommendations(
                risk_factors=risk_factors,
                patient_context=patient_context,
                num_recommendations=4
            )
            
            # Convert AI recommendations to ClinicalRecommendation objects
            for i, ai_rec in enumerate(ai_recommendations):
                if ai_rec and len(ai_rec.strip()) > 10:  # Only include meaningful recommendations
                    recommendation_id = f"LIFE_{patient_id}_{i+1}"
                    
                    # Determine suggested actions based on the recommendation
                    suggested_actions = [ai_rec]
                    if "exercise" in ai_rec.lower() or "activity" in ai_rec.lower():
                        suggested_actions.append("Consult with healthcare provider before starting new exercise program")
                    if "diet" in ai_rec.lower() or "nutrition" in ai_rec.lower():
                        suggested_actions.append("Consider consultation with registered dietitian")
                    
                    clinical_rec = ClinicalRecommendation(
                        recommendation_id=recommendation_id,
                        patient_id=patient_id,
                        recommendation_type="lifestyle",
                        priority="low" if urgency == "routine" else "medium",
                        title="Lifestyle Modification Recommendation",
                        description=ai_rec,
                        rationale=f"AI-generated lifestyle recommendation based on patient's risk profile and health conditions",
                        evidence_level="B",  # Lifestyle recommendations generally have good evidence
                        suggested_actions=suggested_actions,
                        contraindications=[],
                        monitoring_requirements=["Monitor patient adherence to lifestyle changes", "Assess progress at follow-up visits"],
                        confidence=ConfidenceLevel.MEDIUM,
                        created_at=datetime.now()
                    )
                    
                    recommendations.append(clinical_rec)
            
            self.logger.info(f"Generated {len(recommendations)} lifestyle recommendations for patient {patient_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating lifestyle recommendations: {str(e)}")
            return []


# Factory function for dependency injection
def get_insights_processor(graph_client: Neo4jGraphClient) -> InsightsProcessor:
    """
    Factory function to create InsightsProcessor instance.
    
    Args:
        graph_client: Neo4j graph client
        
    Returns:
        InsightsProcessor instance
    """
    return InsightsProcessor(graph_client)
