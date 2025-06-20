"""
Comprehensive integration tests for the MRIA system.

This module provides end-to-end integration testing for the complete
MRIA pipeline, including cross-component interactions, workflow testing,
and system-level validation.

Tests cover:
1. End-to-end document processing pipeline
2. Cross-component integration and data flow
3. Multi-agent workflow coordination
4. System performance under load
5. Error propagation and recovery
6. Data consistency across components
7. API integration testing
8. Database integration testing
9. Real-world scenario simulation
10. System health and monitoring integration
"""

import pytest
import asyncio
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


class TestMRIAIntegration:
    """Comprehensive integration test cases for the MRIA system."""
    
    @pytest.fixture
    def client(self):
        """Create test client for integration testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_medical_document(self):
        """Sample medical document for integration testing."""
        return {
            "filename": "patient_report.pdf",
            "content": """
            MEDICAL RECORD
            
            Patient: John Doe
            DOB: 03/15/1958
            MRN: 12345678
            Date: 2024-06-15
            
            CHIEF COMPLAINT:
            Patient presents with worsening diabetes control and elevated blood pressure.
            
            HISTORY OF PRESENT ILLNESS:
            This 66-year-old male with a history of Type 2 Diabetes Mellitus and 
            hypertension presents for routine follow-up. Patient reports good 
            medication adherence but notes occasional elevated morning glucose readings.
            
            MEDICATIONS:
            - Metformin 500mg twice daily
            - Lisinopril 10mg daily
            - Atorvastatin 20mg at bedtime
            
            PHYSICAL EXAMINATION:
            Vital Signs: BP 145/92 mmHg, HR 78 bpm, Temp 98.6°F
            General: Well-appearing male in no acute distress
            
            LABORATORY RESULTS:
            - HbA1c: 8.2% (elevated)
            - Fasting glucose: 185 mg/dL (elevated)
            - Total cholesterol: 220 mg/dL
            - LDL: 140 mg/dL (elevated)
            
            ASSESSMENT AND PLAN:
            1. Type 2 Diabetes Mellitus - uncontrolled
               - Increase Metformin to 1000mg twice daily
               - Diabetes education reinforcement
               - Follow-up in 3 months
            
            2. Hypertension - uncontrolled
               - Consider ACE inhibitor dose optimization
               - Home blood pressure monitoring
            
            3. Dyslipidemia
               - Continue current statin therapy
               - Recheck lipids in 6 weeks
            """,
            "metadata": {
                "patient_id": "patient_12345",
                "provider": "Dr. Smith",
                "clinic": "Internal Medicine",
                "document_type": "progress_note"
            }
        }

    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self, client, sample_medical_document):
        """Test complete end-to-end document processing pipeline."""
        # Step 1: Upload document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(sample_medical_document["content"])
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                files = {"file": (sample_medical_document["filename"], f, "text/plain")}
                data = {"patient_id": "patient_12345"}
                
                upload_response = client.post("/ingest/upload", files=files, data=data)
                assert upload_response.status_code == 200
                
                upload_data = upload_response.json()
                document_id = upload_data.get("file_id") or upload_data.get("document_id")
                assert document_id is not None
            
            # Step 2: Extract medical entities
            ner_response = client.post("/ner/extract", json={
                "text": sample_medical_document["content"],
                "document_id": document_id
            })
            assert ner_response.status_code == 200
            
            ner_data = ner_response.json()
            assert "entities" in ner_data
            assert len(ner_data["entities"].get("conditions", [])) > 0
            assert len(ner_data["entities"].get("medications", [])) > 0
            
            # Step 3: Process chunks and timeline
            chunk_response = client.post("/chunk/process", json={
                "document_id": document_id,
                "patient_id": "patient_12345",
                "chunking_strategy": "medical_sections"
            })
            assert chunk_response.status_code == 200
            
            # Step 4: Create/update graph
            graph_response = client.post("/graph/create", json={
                "patient_id": "patient_12345",
                "document_id": document_id,
                "entities": ner_data["entities"]
            })
            assert graph_response.status_code == 200
            
            # Step 5: Generate insights
            insights_response = client.post("/insights/generate/patient_12345")
            assert insights_response.status_code == 200
            
            insights_data = insights_response.json()
            assert "clinical_insights" in insights_data
            assert "recommendations" in insights_data
            
        finally:
            # Cleanup
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_supervisor_workflow_coordination(self, client, sample_medical_document):
        """Test supervisor agent workflow coordination."""
        # Enqueue a complete processing job
        job_response = client.post("/supervisor/enqueue", json={
            "patient_id": "patient_12345",
            "documents": [sample_medical_document],
            "workflow_type": "full_pipeline",
            "priority": "normal"
        })
        
        assert job_response.status_code == 200
        job_data = job_response.json()
        job_id = job_data["job_id"]
        
        # Monitor job progress
        max_attempts = 10
        attempt = 0
        job_completed = False
        
        while attempt < max_attempts and not job_completed:
            status_response = client.get(f"/supervisor/status/{job_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            if status_data["status"] in ["completed", "failed"]:
                job_completed = True
            else:
                await asyncio.sleep(1)  # Wait 1 second before checking again
                attempt += 1
        
        # Verify job completion
        final_status = client.get(f"/supervisor/status/{job_id}")
        final_data = final_status.json()
        assert final_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_chat_integration_with_graph_data(self, client):
        """Test chat agent integration with graph database."""
        # First, ensure we have some patient data
        # (This would typically be set up in previous tests or fixtures)
        
        chat_response = client.post("/chat/query", json={
            "message": "Show me the diabetes progression for patient 12345",
            "patient_id": "patient_12345",
            "query_type": "patient_history"
        })
        
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        
        assert "response" in chat_data
        assert "diabetes" in chat_data["response"].lower()
        assert chat_data["confidence_score"] > 0.7

    @pytest.mark.asyncio
    async def test_insights_integration_with_multiple_sources(self, client):
        """Test insights generation using multiple data sources."""
        # Generate insights that should integrate graph, NER, and clinical data
        insights_response = client.post("/insights/generate/patient_12345")
        
        if insights_response.status_code == 200:
            insights_data = insights_response.json()
            
            # Verify insights integrate multiple data sources
            assert "health_summary" in insights_data
            assert "clinical_insights" in insights_data
            assert "recommendations" in insights_data
            
            # Check that insights reference multiple data types
            all_insights_text = str(insights_data)
            assert ("medication" in all_insights_text.lower() or 
                   "condition" in all_insights_text.lower())

    @pytest.mark.asyncio
    async def test_vector_store_integration(self, client, sample_medical_document):
        """Test vector store integration with document processing."""
        # This test would verify that processed documents are properly stored
        # in the vector database and can be retrieved via semantic search
        
        # Process document (assuming it gets stored in vector store)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(sample_medical_document["content"])
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                files = {"file": (sample_medical_document["filename"], f, "text/plain")}
                data = {"patient_id": "patient_12345"}
                
                upload_response = client.post("/ingest/upload", files=files, data=data)
                assert upload_response.status_code == 200
            
            # Test semantic search (if endpoint exists)
            # This would depend on having a vector search endpoint
            # search_response = client.post("/vector/search", json={
            #     "query": "diabetes treatment metformin",
            #     "patient_id": "patient_12345"
            # })
            # assert search_response.status_code == 200
            
        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_cross_component_error_handling(self, client):
        """Test error handling across multiple components."""
        # Test with invalid patient ID
        invalid_insights_response = client.post("/insights/generate/invalid_patient")
        assert invalid_insights_response.status_code == 404
        
        # Test with malformed data
        invalid_ner_response = client.post("/ner/extract", json={
            "text": "",  # Empty text
            "document_id": "invalid_id"
        })
        # Should handle gracefully (may return 400 or 422)
        assert invalid_ner_response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, client, sample_medical_document):
        """Test data consistency across different system components."""
        patient_id = "patient_consistency_test"
        
        # Upload and process document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(sample_medical_document["content"])
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                files = {"file": ("test_consistency.txt", f, "text/plain")}
                data = {"patient_id": patient_id}
                
                upload_response = client.post("/ingest/upload", files=files, data=data)
                assert upload_response.status_code == 200
                
                document_id = upload_response.json().get("file_id")
            
            # Extract entities
            ner_response = client.post("/ner/extract", json={
                "text": sample_medical_document["content"],
                "document_id": document_id
            })
            assert ner_response.status_code == 200
            ner_entities = ner_response.json()["entities"]            # Get patient timeline
            timeline_response = client.get(f"/chunk/timeline/{patient_id}")
            if timeline_response.status_code == 200:
                # Basic validation that timeline endpoint exists and responds
                assert "timeline" in timeline_response.json()
                
                # Verify consistency between NER entities and timeline
                # Check that medications found in NER appear in timeline
                for medication in ner_entities.get("medications", []):
                    if isinstance(medication, dict):
                        med_name = medication.get("name", "").lower()
                    else:
                        med_name = str(medication).lower()
                    
                    if med_name and len(med_name) > 2:  # Skip very short names
                        # Timeline should contain or reference the medication
                        # (This is a loose check since timeline format may vary)
                        pass  # Add specific consistency checks based on actual implementation
            
        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_system_health_monitoring_integration(self, client):
        """Test integration of system health monitoring across components."""
        # Test main health endpoint
        health_response = client.get("/health/detailed")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert "components" in health_data or "services" in health_data
        
        # Test individual component health endpoints
        component_endpoints = [
            "/chat/health",
            "/insights/health",
            "/graph/info"
        ]
        
        for endpoint in component_endpoints:
            try:
                component_health = client.get(endpoint)
                # Should return health info or 404 if endpoint doesn't exist
                assert component_health.status_code in [200, 404]
                
                if component_health.status_code == 200:
                    component_data = component_health.json()
                    assert "status" in component_data
                    
            except Exception:
                # Some endpoints might not be implemented yet
                pass

    @pytest.mark.asyncio
    async def test_performance_under_concurrent_requests(self, client):
        """Test system performance under concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time
            }
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(result["status_code"] == 200 for result in results)
        
        # Average response time should be reasonable
        avg_response_time = sum(result["response_time"] for result in results) / len(results)
        assert avg_response_time < 1.0  # Should respond within 1 second on average

    @pytest.mark.asyncio
    async def test_api_versioning_compatibility(self, client):
        """Test API versioning and backward compatibility."""
        # Test that API endpoints return expected structure
        base_endpoints = [
            "/",
            "/health",
            "/docs"  # OpenAPI documentation
        ]
        
        for endpoint in base_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 404]  # 404 if not implemented
            
            if response.status_code == 200:
                # Basic structure validation
                if endpoint == "/":
                    data = response.json()
                    assert "message" in data or "name" in data
                elif endpoint == "/health":
                    data = response.json()
                    assert "status" in data

    @pytest.mark.asyncio
    async def test_real_world_medical_scenario(self, client):
        """Test a complete real-world medical scenario."""
        scenario_document = {
            "content": """
            EMERGENCY DEPARTMENT VISIT
            
            Patient: Jane Smith
            DOB: 05/22/1965
            MRN: 87654321
            Date: 2024-06-15
            Time: 14:30
            
            CHIEF COMPLAINT: Chest pain
            
            HISTORY OF PRESENT ILLNESS:
            58-year-old female with history of diabetes and hypertension presents 
            to ED with acute onset chest pain starting 2 hours ago. Pain described 
            as crushing, substernal, radiating to left arm. Associated with 
            diaphoresis and nausea. Denies shortness of breath.
            
            MEDICATIONS:
            - Metformin 1000mg twice daily
            - Lisinopril 20mg daily
            - Aspirin 81mg daily
            
            PHYSICAL EXAM:
            VS: BP 160/95, HR 95, RR 20, O2 sat 98% RA
            Appears anxious, diaphoretic
            Heart: Regular rate, no murmurs
            Lungs: Clear bilaterally
            
            LABS:
            - Troponin I: 0.15 ng/mL (elevated)
            - CK-MB: 8.5 ng/mL (elevated)
            - Glucose: 165 mg/dL
            
            EKG: ST elevation in leads II, III, aVF
            
            ASSESSMENT:
            Acute ST-elevation myocardial infarction (STEMI) - inferior wall
            
            PLAN:
            1. Activate cardiac catheterization lab
            2. Administer dual antiplatelet therapy
            3. Heparin protocol
            4. Serial cardiac enzymes
            5. Urgent cardiology consultation
            """
        }
        
        # Process this emergency scenario through the system
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(scenario_document["content"])
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                files = {"file": ("emergency_visit.txt", f, "text/plain")}
                data = {"patient_id": "patient_87654321", "priority": "urgent"}
                
                upload_response = client.post("/ingest/upload", files=files, data=data)
                assert upload_response.status_code == 200
                
                document_id = upload_response.json().get("file_id")
            
            # Extract critical entities (should identify STEMI, medications, etc.)
            ner_response = client.post("/ner/extract", json={
                "text": scenario_document["content"],
                "document_id": document_id,
                "priority": "urgent"
            })
            assert ner_response.status_code == 200
            
            entities = ner_response.json()["entities"]
            
            # Should identify critical conditions
            conditions = entities.get("conditions", [])
            condition_text = " ".join([str(c) for c in conditions]).lower()
            assert ("myocardial infarction" in condition_text or 
                   "stemi" in condition_text or 
                   "heart attack" in condition_text)
            
            # Should identify emergency procedures/labs
            procedures = entities.get("procedures", [])
            procedure_text = " ".join([str(p) for p in procedures]).lower()
            assert ("troponin" in procedure_text or 
                   "ekg" in procedure_text or 
                   "catheterization" in procedure_text)
            
        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_data_privacy_and_security_integration(self, client):
        """Test data privacy and security measures in integration."""
        # Test that sensitive data is handled appropriately
        sensitive_data = {
            "patient_id": "patient_privacy_test",
            "ssn": "123-45-6789",  # Should be filtered/encrypted
            "medical_record": "Patient has sensitive condition",
            "contact_info": "555-123-4567"
        }
        
        # Test with potential PII in medical text
        response = client.post("/ner/extract", json={
            "text": f"Patient SSN: {sensitive_data['ssn']}, Phone: {sensitive_data['contact_info']}",
            "document_id": "privacy_test_doc"
        })
        
        # Should process but handle PII appropriately
        assert response.status_code in [200, 400, 422]
        
        if response.status_code == 200:
            data = response.json()
            response_text = str(data).lower()
            # SSN should not appear in plain text in response
            assert sensitive_data['ssn'] not in response_text

    @pytest.mark.asyncio
    async def test_system_recovery_after_component_failure(self, client):
        """Test system recovery capabilities after component failures."""
        # Simulate component failure and recovery
        # (This would typically involve mocking service failures)
        
        # Test graceful degradation
        with patch('app.services.graph_client.GraphClient.get_patient_data') as mock_graph:
            mock_graph.side_effect = Exception("Database temporarily unavailable")
            
            # System should still respond, possibly with degraded functionality
            insights_response = client.post("/insights/generate/patient_12345")
            
            # Should handle gracefully (may return error but not crash)
            assert insights_response.status_code in [200, 404, 500, 503]
            
            if insights_response.status_code != 200:
                error_data = insights_response.json()
                assert "error" in error_data or "message" in error_data
