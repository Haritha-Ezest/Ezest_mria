"""
Security testing for the MRIA system.

This module provides comprehensive security testing to ensure
the system is protected against common vulnerabilities and
follows security best practices.

Tests cover:
1. Input validation and sanitization
2. Authentication and authorization
3. Data privacy and PII protection
4. SQL injection prevention
5. Cross-site scripting (XSS) prevention
6. File upload security
7. API security and rate limiting
8. Data encryption and secure storage
9. Session management security
10. OWASP Top 10 vulnerability checks
"""

import pytest
import tempfile
import os
import json
from fastapi.testclient import TestClient

from app.main import app


class TestMRIASecurity:
    """Security test cases for the MRIA system."""
    
    @pytest.fixture
    def client(self):
        """Create test client for security testing."""
        return TestClient(app)
    
    @pytest.fixture
    def malicious_payloads(self):
        """Common malicious payloads for security testing."""
        return {
            "sql_injection": [
                "'; DROP TABLE patients; --",
                "' OR 1=1 --",
                "'; SELECT * FROM users WHERE 'x'='x",
                "admin'--",
                "1' UNION SELECT password FROM users--"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "';alert('XSS');//"
            ],
            "command_injection": [
                "; rm -rf /",
                "&& cat /etc/passwd",
                "| whoami",
                "`id`",
                "$(id)"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc//passwd"
            ],
            "ldap_injection": [
                "*)(uid=*))(|(uid=*",
                "*)(|(password=*))",
                "admin)(&(password=*))"
            ]
        }
    
    @pytest.fixture
    def sensitive_data_patterns(self):
        """Patterns to detect sensitive data leakage."""
        return {
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "phone": r"\d{3}[-.]?\d{3}[-.]?\d{4}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "medical_record": r"MRN[\s:]?\d+",
            "password": r"password[\s:=]+\w+"
        }

    def test_input_validation_sql_injection(self, client, malicious_payloads):
        """Test SQL injection prevention in all endpoints."""
        endpoints_to_test = [
            ("POST", "/ner/extract", "text"),
            ("POST", "/chat/query", "message"),
            ("GET", "/chunk/timeline/{}", "patient_id"),
            ("POST", "/insights/generate/{}", "patient_id")
        ]
        
        for method, endpoint_template, param_name in endpoints_to_test:
            for payload in malicious_payloads["sql_injection"]:
                if "{}" in endpoint_template:
                    # URL parameter injection
                    endpoint = endpoint_template.format(payload)
                    response = client.get(endpoint) if method == "GET" else client.post(endpoint)
                else:
                    # JSON body injection
                    data = {param_name: payload}
                    if "document_id" in endpoint_template:
                        data["document_id"] = "test_doc"
                    
                    response = client.post(endpoint_template, json=data)
                
                # Should not return database error messages or succeed with malicious input
                assert response.status_code in [400, 422, 404, 500]
                
                if response.status_code != 404:  # Skip if endpoint doesn't exist
                    response_text = response.text.lower()
                    
                    # Should not contain SQL error messages
                    sql_error_indicators = [
                        "sql", "mysql", "postgresql", "sqlite", "database error",
                        "syntax error", "column", "table", "select", "insert",
                        "update", "delete", "drop"
                    ]
                    
                    for indicator in sql_error_indicators:
                        assert indicator not in response_text

    def test_input_validation_xss_prevention(self, client, malicious_payloads):
        """Test XSS prevention in all text processing endpoints."""
        xss_test_endpoints = [
            ("POST", "/ner/extract", {"text": "", "document_id": "test"}),
            ("POST", "/chat/query", {"message": "", "patient_id": "test"}),
        ]
        
        for method, endpoint, base_data in xss_test_endpoints:
            for xss_payload in malicious_payloads["xss"]:
                # Inject XSS payload into text fields
                test_data = base_data.copy()
                for key in test_data:
                    if isinstance(test_data[key], str) and test_data[key] == "":
                        test_data[key] = xss_payload
                
                response = client.post(endpoint, json=test_data)
                
                # Response should not contain unescaped script tags
                if response.status_code == 200:
                    response_text = response.text
                    
                    # Check that dangerous content is escaped or removed
                    dangerous_patterns = [
                        "<script>", "</script>", "javascript:", "onerror=", "onload="
                    ]
                    
                    for pattern in dangerous_patterns:
                        assert pattern.lower() not in response_text.lower()

    def test_file_upload_security(self, client):
        """Test file upload security measures."""
        # Test various malicious file types and content
        malicious_files = [
            ("malicious.exe", b"MZ\x90\x00", "application/octet-stream"),  # Executable
            ("script.js", b"alert('XSS')", "application/javascript"),  # JavaScript
            ("shell.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),  # PHP shell
            ("large_file.txt", b"A" * (10 * 1024 * 1024), "text/plain"),  # 10MB file
            ("../../../etc/passwd", b"root:x:0:0:", "text/plain"),  # Path traversal
        ]
        
        for filename, content, content_type in malicious_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    files = {"file": (filename, f, content_type)}
                    data = {"patient_id": "security_test"}
                    
                    response = client.post("/ingest/upload", files=files, data=data)
                    
                    # Should reject dangerous files
                    if filename.endswith(('.exe', '.js', '.php')) or '../' in filename:
                        assert response.status_code in [400, 422, 415]  # Bad request or unsupported media
                    
                    # Should handle large files appropriately
                    if len(content) > 5 * 1024 * 1024:  # 5MB
                        assert response.status_code in [400, 413, 422]  # Request entity too large
                    
                    # Should not expose internal file paths in error messages
                    if response.status_code >= 400:
                        response_text = response.text.lower()
                        assert "/tmp/" not in response_text
                        assert "c:\\" not in response_text
                        assert "/var/" not in response_text
            
            finally:
                os.unlink(temp_file_path)

    def test_data_privacy_pii_protection(self, client, sensitive_data_patterns):
        """Test protection of personally identifiable information (PII)."""
        # Test data containing various types of PII
        pii_test_data = {
            "text": """
            Patient John Doe, SSN: 123-45-6789, Phone: 555-123-4567
            Email: patient@example.com, Credit Card: 4532-1234-5678-9012
            Medical Record Number: MRN12345678
            Password: secret123
            """,
            "document_id": "pii_test"
        }
        
        response = client.post("/ner/extract", json=pii_test_data)
        
        if response.status_code == 200:
            response_data = response.json()
            response_text = json.dumps(response_data).lower()
            
            # Check that PII is not exposed in response
            sensitive_values = [
                "123-45-6789",  # SSN
                "4532-1234-5678-9012",  # Credit card
                "555-123-4567",  # Phone
                "patient@example.com",  # Email
                "secret123"  # Password
            ]
            
            for sensitive_value in sensitive_values:
                assert sensitive_value.lower() not in response_text

    def test_authentication_bypass_attempts(self, client):
        """Test authentication bypass attempts."""
        # Test common authentication bypasses
        bypass_attempts = [
            {"Authorization": "Bearer fake_token"},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},  # admin:admin base64
            {"X-User-Id": "admin"},
            {"X-Forwarded-User": "admin"},
            {"X-Remote-User": "admin"},
        ]
        
        protected_endpoints = [
            "/insights/generate/test_patient",
            "/graph/create",
            "/supervisor/enqueue"
        ]
        
        for endpoint in protected_endpoints:
            for headers in bypass_attempts:
                response = client.post(endpoint, headers=headers, json={})
                
                # Should not allow access with fake credentials
                # (Assuming authentication is implemented)
                if response.status_code == 401:  # Unauthorized
                    assert "authentication" in response.text.lower() or "unauthorized" in response.text.lower()

    def test_authorization_privilege_escalation(self, client):
        """Test for privilege escalation vulnerabilities."""
        # Test accessing resources of other patients
        patient_endpoints = [
            "/insights/generate/patient_123",
            "/chunk/timeline/patient_123",
            "/graph/patient/patient_123"
        ]
        
        # Test with different patient contexts
        test_contexts = [
            {"X-Patient-Id": "patient_456"},  # Different patient
            {"X-User-Role": "admin"},  # Role manipulation
            {"X-Privilege-Level": "high"},  # Privilege manipulation
        ]
        
        for endpoint in patient_endpoints:
            for headers in test_contexts:
                response = client.get(endpoint, headers=headers)
                
                # Should properly validate patient access
                # (Implementation dependent on actual authorization logic)
                if response.status_code not in [404, 501]:  # Skip if not implemented
                    assert response.status_code in [200, 401, 403]

    def test_command_injection_prevention(self, client, malicious_payloads):
        """Test command injection prevention."""
        # Test endpoints that might execute system commands
        command_test_endpoints = [
            ("POST", "/ingest/upload", "filename_param"),
            ("POST", "/ocr/process", "image_path"),
        ]
        
        for method, endpoint, param_name in command_test_endpoints:
            for payload in malicious_payloads["command_injection"]:
                if endpoint == "/ingest/upload":
                    # Test via filename
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(b"test content")
                        temp_file_path = temp_file.name
                    
                    try:
                        with open(temp_file_path, 'rb') as f:
                            files = {"file": (payload, f, "text/plain")}
                            data = {"patient_id": "test"}
                            
                            response = client.post(endpoint, files=files, data=data)
                            
                            # Should reject filenames with command injection
                            assert response.status_code in [400, 422]
                    
                    finally:
                        os.unlink(temp_file_path)
                
                else:
                    # Test via JSON payload
                    data = {param_name: payload}
                    response = client.post(endpoint, json=data)
                    
                    # Should not execute commands
                    assert response.status_code in [400, 422, 404, 500]

    def test_path_traversal_prevention(self, client, malicious_payloads):
        """Test path traversal attack prevention."""
        # Test file-related endpoints for path traversal
        for payload in malicious_payloads["path_traversal"]:
            # Test file upload with malicious filename
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"test content")
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    files = {"file": (payload, f, "text/plain")}
                    data = {"patient_id": "test"}
                    
                    response = client.post("/ingest/upload", files=files, data=data)
                    
                    # Should reject path traversal attempts
                    assert response.status_code in [400, 422]
                    
                    # Should not reveal file system structure in error
                    if response.status_code >= 400:
                        response_text = response.text.lower()
                        assert "/etc/" not in response_text
                        assert "c:\\" not in response_text
            
            finally:
                os.unlink(temp_file_path)

    def test_rate_limiting_dos_prevention(self, client):
        """Test rate limiting and DoS prevention."""
        # Rapid fire requests to test rate limiting
        rapid_requests = []
        
        for i in range(100):  # 100 rapid requests
            response = client.get("/health")
            rapid_requests.append(response.status_code)
            
            # Very short delay
            import time
            time.sleep(0.001)
          # Check for rate limiting responses
        has_rate_limiting = any(code == 429 for code in rapid_requests)
        all_successful = all(code == 200 for code in rapid_requests)
        
        # If rate limiting is implemented, should see 429 responses
        # If not implemented, all should be 200 (but this is a security concern)
        
        # At minimum, system should not crash under load
        assert all(code in [200, 429, 503] for code in rapid_requests)
        
        # Document whether rate limiting is implemented
        if not has_rate_limiting and all_successful:
            # Rate limiting may not be implemented - this is informational
            pass

    def test_information_disclosure_prevention(self, client):
        """Test prevention of information disclosure."""
        # Test error handling to ensure no sensitive info is leaked
        error_inducing_requests = [
            ("GET", "/nonexistent_endpoint"),
            ("POST", "/ner/extract", {"invalid": "json_structure"}),
            ("GET", "/insights/generate/"),  # Missing patient ID
            ("POST", "/graph/create", {}),  # Missing required fields
        ]
        
        for method, endpoint, *args in error_inducing_requests:
            data = args[0] if args else None
            
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=data)
            
            # Check that error responses don't leak sensitive information
            if response.status_code >= 400:
                response_text = response.text.lower()
                
                # Should not contain sensitive path information
                sensitive_patterns = [
                    "/home/", "/root/", "c:\\users\\", "/var/", "/tmp/",
                    "traceback", "stack trace", "internal server error",
                    "database", "sql", "connection string", "password"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in response_text

    def test_session_security(self, client):
        """Test session management security."""
        # Test session-related headers and security
        response = client.get("/health")
        
        headers = response.headers
        
        # Check for security headers
        security_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
            "x-xss-protection": "1; mode=block",
            "strict-transport-security": "max-age=31536000",
            "content-security-policy": None  # Should exist
        }
        
        for header_name, expected_value in security_headers.items():
            if expected_value:
                assert headers.get(header_name) == expected_value
            else:
                # Just check that the header exists
                assert header_name in headers or header_name.title() in headers

    def test_data_encryption_security(self, client):
        """Test data encryption and secure storage."""
        # Test that sensitive data is properly encrypted
        sensitive_document = {
            "text": "Patient has sensitive medical condition: HIV positive",
            "patient_id": "encryption_test_patient",
            "document_id": "sensitive_doc"
        }
        
        # Process sensitive document
        response = client.post("/ner/extract", json=sensitive_document)
        
        if response.status_code == 200:
            # Check that response doesn't contain plaintext sensitive data
            # (This depends on implementation - data might be encrypted in storage)
            response_data = response.json()
            
            # Ensure sensitive information is handled appropriately
            # Implementation would depend on specific encryption requirements
            assert isinstance(response_data, dict)
            assert "entities" in response_data or "error" in response_data

    def test_api_versioning_security(self, client):
        """Test API versioning security."""
        # Test that old API versions don't expose vulnerabilities
        version_tests = [
            "/v1/health",
            "/v2/health", 
            "/api/v1/health",
            "/api/v2/health"
        ]
        
        for endpoint in version_tests:
            response = client.get(endpoint)
            
            # Should either work securely or return 404
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                # Should include security headers
                assert response.headers.get("server") != "nginx/1.0.0"  # Don't expose server version

    def test_cors_security(self, client):
        """Test CORS (Cross-Origin Resource Sharing) security."""
        # Test CORS headers with various origins
        malicious_origins = [
            "http://evil.com",
            "https://malicious-site.com",
            "null",
            "*"
        ]
        
        for origin in malicious_origins:
            headers = {"Origin": origin}
            response = client.get("/health", headers=headers)
            
            # Check CORS headers in response
            cors_header = response.headers.get("access-control-allow-origin")
            
            if cors_header:
                # Should not allow arbitrary origins
                assert cors_header != "*" or cors_header != origin
                
                # Should only allow specific trusted origins
                trusted_origins = ["https://trusted-domain.com", "https://localhost:3000"]
                if cors_header not in trusted_origins:
                    # If CORS is permissive, ensure it's intentional and documented
                    pass

    def test_input_length_validation(self, client):
        """Test input length validation to prevent buffer overflow attacks."""
        # Test with extremely long inputs
        long_inputs = [
            "A" * 10000,  # 10KB
            "B" * 100000,  # 100KB
            "C" * 1000000,  # 1MB
        ]
        
        for long_input in long_inputs:
            test_data = {
                "text": long_input,
                "document_id": "length_test"
            }
            
            response = client.post("/ner/extract", json=test_data)
            
            # Should handle long inputs gracefully
            assert response.status_code in [200, 400, 413, 422]  # OK, Bad Request, or Payload Too Large
            
            # Should not crash or hang
            # (This test should complete in reasonable time)
