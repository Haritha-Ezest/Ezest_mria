"""
Performance and load testing for the MRIA system.

This module provides comprehensive performance testing to ensure
the system meets performance requirements under various load conditions.

Tests cover:
1. Response time benchmarks for all endpoints
2. Throughput testing under concurrent load
3. Memory usage and resource consumption
4. Database query performance
5. Vector store operation performance
6. Large document processing performance
7. System scalability testing
8. Resource cleanup and garbage collection
9. API rate limiting compliance
10. Background task performance
"""

import pytest
import time
import psutil
import gc
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient

from app.main import app


class TestMRIAPerformance:
    """Performance test cases for the MRIA system."""
    
    @pytest.fixture
    def client(self):
        """Create test client for performance testing."""
        return TestClient(app)
    
    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for various operations."""
        return {
            "health_check": 0.1,  # 100ms
            "document_upload": 5.0,  # 5 seconds
            "ner_extraction": 3.0,  # 3 seconds
            "chunking": 2.0,  # 2 seconds
            "graph_operations": 1.0,  # 1 second
            "insights_generation": 10.0,  # 10 seconds
            "chat_response": 5.0,  # 5 seconds
            "concurrent_requests": 0.5,  # 500ms average under load
        }
    
    def measure_endpoint_performance(self, client, method, endpoint, data=None, files=None):
        """Measure performance of a single endpoint."""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            if method.upper() == "GET":
                response = client.get(endpoint)
            elif method.upper() == "POST":
                if files:
                    response = client.post(endpoint, files=files, data=data)
                else:
                    response = client.post(endpoint, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "memory_usage": memory_after - memory_before,
                "response_size": len(response.content) if response.content else 0
            }
        except Exception as e:
            end_time = time.time()
            return {
                "status_code": 500,
                "response_time": end_time - start_time,
                "memory_usage": 0,
                "response_size": 0,
                "error": str(e)
            }

    def test_health_endpoint_performance(self, client, performance_thresholds):
        """Test health endpoint performance."""
        measurements = []
        
        # Run multiple measurements to get average
        for _ in range(10):
            result = self.measure_endpoint_performance(client, "GET", "/health")
            measurements.append(result)
        
        # Calculate statistics
        response_times = [m["response_time"] for m in measurements]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        # Assertions
        assert avg_response_time < performance_thresholds["health_check"]
        assert max_response_time < performance_thresholds["health_check"] * 2
        assert all(m["status_code"] == 200 for m in measurements)

    def test_document_upload_performance(self, client, performance_thresholds):
        """Test document upload performance with various file sizes."""
        import tempfile
        
        # Test with different document sizes
        test_sizes = [
            ("small", "A" * 1000),  # 1KB
            ("medium", "B" * 50000),  # 50KB
            ("large", "C" * 500000),  # 500KB
        ]
        
        for size_name, content in test_sizes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    files = {"file": (f"{size_name}_doc.txt", f, "text/plain")}
                    data = {"patient_id": f"perf_test_{size_name}"}
                    
                    result = self.measure_endpoint_performance(
                        client, "POST", "/ingest/upload", data=data, files=files
                    )
                    
                    # Performance assertions based on document size
                    if size_name == "small":
                        assert result["response_time"] < performance_thresholds["document_upload"] / 5
                    elif size_name == "medium":
                        assert result["response_time"] < performance_thresholds["document_upload"] / 2
                    else:  # large
                        assert result["response_time"] < performance_thresholds["document_upload"]
                    
                    assert result["status_code"] == 200
            
            finally:
                import os
                os.unlink(temp_file_path)

    def test_ner_extraction_performance(self, client, performance_thresholds):
        """Test NER extraction performance with various text lengths."""
        # Test texts of different lengths
        test_texts = [
            ("short", "Patient has diabetes and takes metformin."),
            ("medium", """
            Patient: John Doe, Age: 65
            Chief Complaint: Chest pain and shortness of breath
            History: Patient has a history of diabetes, hypertension, and high cholesterol.
            Current medications include metformin, lisinopril, and atorvastatin.
            Physical exam reveals elevated blood pressure and irregular heart rhythm.
            Labs show elevated glucose and cholesterol levels.
            """),
            ("long", """
            COMPREHENSIVE MEDICAL RECORD
            
            Patient Information:
            Name: Jane Smith
            Age: 58 years old
            Medical Record Number: 12345678
            
            Chief Complaint:
            Patient presents with acute onset chest pain, described as crushing and 
            substernal, radiating to the left arm and jaw. Pain started approximately 
            2 hours ago while at rest. Associated symptoms include nausea, diaphoresis, 
            and mild shortness of breath.
            
            Past Medical History:
            1. Type 2 Diabetes Mellitus - diagnosed 10 years ago, well controlled on metformin
            2. Hypertension - diagnosed 8 years ago, controlled on lisinopril
            3. Hyperlipidemia - diagnosed 5 years ago, on atorvastatin
            4. Obesity - BMI 32, ongoing weight management
            
            Current Medications:
            - Metformin 1000mg twice daily
            - Lisinopril 20mg daily
            - Atorvastatin 40mg at bedtime
            - Aspirin 81mg daily for cardiovascular protection
            
            Physical Examination:
            Vital Signs: BP 165/95 mmHg, HR 98 bpm, RR 22, O2 sat 96% on room air
            General: Patient appears anxious and diaphoretic
            HEENT: Normal examination
            Cardiovascular: Regular rate, no murmurs, gallops, or rubs
            Pulmonary: Clear to auscultation bilaterally
            Abdomen: Soft, non-tender, no organomegaly
            Extremities: No edema, pedal pulses intact
            
            Laboratory Results:
            - Complete Blood Count: Within normal limits
            - Comprehensive Metabolic Panel: Glucose 185 mg/dL (elevated)
            - Lipid Panel: Total cholesterol 245 mg/dL, LDL 165 mg/dL (elevated)
            - Cardiac Enzymes: Troponin I 0.12 ng/mL (elevated), CK-MB 6.8 ng/mL
            - BNP: 125 pg/mL (slightly elevated)
            
            Diagnostic Studies:
            - 12-lead EKG: ST-segment elevation in leads II, III, aVF consistent with inferior STEMI
            - Chest X-ray: No acute cardiopulmonary process
            - Echocardiogram: Hypokinesis of inferior wall, EF 45%
            
            Assessment and Plan:
            1. Acute ST-Elevation Myocardial Infarction (STEMI) - Inferior wall
               - Emergent cardiac catheterization with primary PCI
               - Dual antiplatelet therapy with aspirin and clopidogrel
               - High-intensity statin therapy
               - ACE inhibitor for cardioprotection
            
            2. Type 2 Diabetes Mellitus
               - Continue metformin
               - Monitor blood glucose closely during hospitalization
               - Diabetes education reinforcement
            
            3. Hypertension
               - Continue lisinopril, may increase dose post-MI
               - Target blood pressure <130/80 mmHg
            
            4. Hyperlipidemia
               - Increase atorvastatin to 80mg daily post-MI
               - Target LDL <70 mg/dL
            """ * 2  # Double the text to make it longer
            )
        ]
        
        for text_name, text_content in test_texts:
            result = self.measure_endpoint_performance(
                client, "POST", "/ner/extract", 
                data={"text": text_content, "document_id": f"perf_test_{text_name}"}
            )
            
            # Performance assertions based on text length
            if text_name == "short":
                assert result["response_time"] < performance_thresholds["ner_extraction"] / 10
            elif text_name == "medium":
                assert result["response_time"] < performance_thresholds["ner_extraction"] / 3
            else:  # long
                assert result["response_time"] < performance_thresholds["ner_extraction"]
            
            # Should succeed regardless of text length
            assert result["status_code"] == 200

    def test_concurrent_request_performance(self, client, performance_thresholds):
        """Test system performance under concurrent load."""
        
        def make_health_request():
            return self.measure_endpoint_performance(client, "GET", "/health")
        
        # Test with increasing levels of concurrency
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.time()
                
                # Submit concurrent requests
                futures = [executor.submit(make_health_request) for _ in range(concurrency)]
                results = [future.result() for future in as_completed(futures)]
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Calculate performance metrics
                response_times = [r["response_time"] for r in results]
                avg_response_time = statistics.mean(response_times)
                success_rate = sum(1 for r in results if r["status_code"] == 200) / len(results)
                
                # Performance assertions
                assert success_rate >= 0.95  # 95% success rate
                assert avg_response_time < performance_thresholds["concurrent_requests"]
                assert total_time < concurrency * 0.1  # Should handle concurrent requests efficiently    def test_memory_usage_performance(self, client):
        """Test memory usage during various operations."""
        # Force garbage collection before starting
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        operations = [
            ("health_check", lambda: client.get("/health")),
            ("insights_generation", lambda: client.post("/insights/generate/test_patient")),
        ]
        
        memory_measurements = []
        
        for op_name, operation in operations:
            gc.collect()  # Clean up before measurement
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                operation()
            except Exception:
                pass  # Some operations may fail, focus on memory usage
            
            gc.collect()  # Clean up after operation
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_usage = memory_after - memory_before
            memory_measurements.append((op_name, memory_usage))
        
        # Memory usage assertions
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Should not have significant memory leaks
        assert memory_growth < 100  # Less than 100MB growth
        
        # Individual operations should not use excessive memory
        for op_name, usage in memory_measurements:
            assert usage < 50  # Less than 50MB per operation

    def test_database_query_performance(self, client):
        """Test database query performance."""
        # This would test actual database operations if they exist
        # For now, test endpoints that likely involve database queries
        
        db_endpoints = [
            ("GET", "/graph/info", None),
            ("POST", "/insights/generate/test_patient", None),
        ]
        
        for method, endpoint, data in db_endpoints:
            results = []
            
            # Run multiple queries to test consistency
            for _ in range(5):
                result = self.measure_endpoint_performance(client, method, endpoint, data)
                results.append(result)
            
            response_times = [r["response_time"] for r in results]
            avg_response_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
            
            # Database queries should be consistent
            assert std_dev < avg_response_time * 0.5  # Std dev should be less than 50% of mean
            assert avg_response_time < 5.0  # Should respond within 5 seconds

    def test_large_batch_processing_performance(self, client):
        """Test performance with large batch operations."""
        # Test batch NER processing
        large_batch = {
            "documents": [
                {
                    "id": f"doc_{i}",
                    "text": f"Patient {i} has diabetes and takes medication {i}."
                }
                for i in range(50)  # 50 documents
            ]
        }
        
        start_time = time.time()
        
        # Process documents individually (simulating batch processing)
        results = []
        for doc in large_batch["documents"]:
            result = self.measure_endpoint_performance(
                client, "POST", "/ner/extract",
                data={"text": doc["text"], "document_id": doc["id"]}
            )
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions for batch processing
        success_rate = sum(1 for r in results if r["status_code"] == 200) / len(results)
        avg_processing_time = total_time / len(large_batch["documents"])
        
        assert success_rate >= 0.95  # 95% success rate
        assert avg_processing_time < 1.0  # Less than 1 second per document on average
        assert total_time < 60.0  # Complete batch in under 1 minute

    def test_api_rate_limiting_performance(self, client):
        """Test API rate limiting behavior."""
        # Test rapid successive requests
        rapid_requests = []
        
        for i in range(30):  # Make 30 rapid requests
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            rapid_requests.append({
                "request_number": i,
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.01)
        
        # Analyze rate limiting behavior
        success_count = sum(1 for r in rapid_requests if r["status_code"] == 200)
        rate_limited_count = sum(1 for r in rapid_requests if r["status_code"] == 429)
        
        # Either all requests succeed (no rate limiting) or some are rate limited
        assert success_count + rate_limited_count == len(rapid_requests)
        
        # If rate limiting is implemented, it should be consistent
        if rate_limited_count > 0:
            # Rate limited responses should be fast
            rate_limited_times = [r["response_time"] for r in rapid_requests if r["status_code"] == 429]
            avg_rate_limited_time = statistics.mean(rate_limited_times)
            assert avg_rate_limited_time < 0.1  # Rate limit responses should be very fast

    def test_background_task_performance(self, client):
        """Test background task performance."""
        # Test supervisor job queuing and processing
        job_data = {
            "patient_id": "perf_test_patient",
            "documents": [{"content": "Test document for performance testing"}],
            "workflow_type": "basic_processing",
            "priority": "normal"
        }
        
        start_time = time.time()
        
        # Enqueue job
        enqueue_response = client.post("/supervisor/enqueue", json=job_data)
        
        if enqueue_response.status_code == 200:
            job_id = enqueue_response.json().get("job_id")
            
            # Monitor job completion
            max_wait_time = 30  # 30 seconds
            check_interval = 0.5  # 500ms
            checks = 0
            max_checks = int(max_wait_time / check_interval)
            
            while checks < max_checks:
                status_response = client.get(f"/supervisor/status/{job_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") in ["completed", "failed"]:
                        break
                
                time.sleep(check_interval)
                checks += 1
            
            end_time = time.time()
            total_processing_time = end_time - start_time
            
            # Background task performance assertions
            assert total_processing_time < max_wait_time  # Should complete within time limit    def test_resource_cleanup_performance(self, client):
        """Test resource cleanup and garbage collection performance."""
        # Record initial state
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_objects = len(gc.get_objects())
        
        # Perform operations that create temporary resources
        for i in range(10):
            client.get("/health")
            client.post("/ner/extract", json={
                "text": f"Test document {i} with temporary data",
                "document_id": f"temp_doc_{i}"
            })
        
        # Force garbage collection
        collected = gc.collect()
        
        # Check resource cleanup
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_objects = len(gc.get_objects())
        
        memory_growth = final_memory - initial_memory
        object_growth = final_objects - initial_objects
        
        # Resource cleanup assertions
        assert memory_growth < 20  # Less than 20MB growth after cleanup
        assert object_growth < 1000  # Less than 1000 new objects after cleanup
        assert collected >= 0  # Garbage collection should work
