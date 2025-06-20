"""
Comprehensive test suite for the OCR + Ingestion Agent.

This module provides complete test coverage for the OCR Agent functionality
including document processing, text extraction, image preprocessing, and medical optimization.

Tests cover:
1. OCR Agent initialization and configuration
2. Document upload and validation
3. Multi-format text extraction (PDF, images, DOCX, Excel)
4. Medical document optimization and preprocessing
5. OCR processing with different engines (Tesseract, Azure Form Recognizer)
6. Confidence scoring and quality assessment
7. Image preprocessing and enhancement
8. Error handling and recovery mechanisms
9. Performance testing and optimization
10. Integration with supervisor workflow
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image

from app.services.ocr_processor import OCRProcessor
from app.services.file_handler import FileHandler
from app.schemas.ocr import (
    OCRRequest, OCRResponse, OCRStatus, OCRConfiguration, 
    DocumentType, OCREngine, ImagePreprocessing, PreprocessingStep
)


class TestOCRAgent:
    """Comprehensive test cases for the OCR + Ingestion Agent."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create an OCR processor instance for testing."""
        return OCRProcessor()
    
    @pytest.fixture
    def file_handler(self):
        """Create a file handler instance for testing."""
        return FileHandler()
    
    @pytest.fixture
    def sample_medical_image(self):
        """Create a sample medical document image for testing."""
        # Create a test image with medical text
        image = Image.new('RGB', (800, 600), color='white')
        return image
    
    @pytest.fixture
    def sample_prescription_image(self):
        """Create a sample prescription image for testing."""
        image = Image.new('RGB', (600, 800), color='white')
        return image
    
    @pytest.fixture
    def lab_report_ocr_config(self):
        """Create OCR configuration for lab reports."""
        return OCRConfiguration(
            engine=OCREngine.TESSERACT,
            document_type=DocumentType.LAB_REPORT,
            preprocessing=ImagePreprocessing.MEDICAL_OPTIMIZED,
            languages=["eng"],
            confidence_threshold=0.8,
            dpi=300,
            enhance_contrast=True,
            correct_skew=True,
            extract_tables=True,
            preserve_layout=True
        )
    
    @pytest.fixture
    def prescription_ocr_config(self):
        """Create OCR configuration for prescriptions."""
        return OCRConfiguration(
            engine=OCREngine.TESSERACT,
            document_type=DocumentType.PRESCRIPTION,
            preprocessing=ImagePreprocessing.MEDICAL_OPTIMIZED,
            languages=["eng"],
            confidence_threshold=0.7,
            dpi=600,
            enhance_contrast=True,
            correct_skew=True,
            extract_handwriting=True,
            medical_terminology_boost=True
        )

    # Test 1: OCR Agent Initialization and Configuration
    async def test_ocr_processor_initialization(self, ocr_processor):
        """Test OCR processor initialization with default configuration."""
        assert ocr_processor is not None
        assert hasattr(ocr_processor, 'tesseract_config')
        assert hasattr(ocr_processor, 'supported_formats')
        
        # Test supported formats
        supported_formats = ocr_processor.get_supported_formats()
        expected_formats = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.docx', '.xlsx'}
        assert expected_formats.issubset(set(supported_formats))

    async def test_ocr_engine_configuration(self, ocr_processor):
        """Test OCR engine configuration and switching."""
        # Test Tesseract configuration
        tesseract_config = OCRConfiguration(
            engine=OCREngine.TESSERACT,
            languages=["eng", "fra"],
            confidence_threshold=0.8
        )
        
        ocr_processor.configure_engine(tesseract_config)
        assert ocr_processor.current_engine == OCREngine.TESSERACT
        assert ocr_processor.confidence_threshold == 0.8
        
        # Test Azure Form Recognizer configuration
        azure_config = OCRConfiguration(
            engine=OCREngine.AZURE_FORM_RECOGNIZER,
            confidence_threshold=0.9,
            extract_tables=True
        )
        
        ocr_processor.configure_engine(azure_config)
        assert ocr_processor.current_engine == OCREngine.AZURE_FORM_RECOGNIZER

    async def test_medical_document_type_detection(self, ocr_processor):
        """Test automatic medical document type detection."""
        # Test lab report detection
        lab_report_text = "LABORATORY REPORT\nPatient: John Doe\nHbA1c: 7.8%\nGlucose: 165 mg/dL"
        doc_type = ocr_processor.detect_document_type(lab_report_text)
        assert doc_type == DocumentType.LAB_REPORT
        
        # Test prescription detection
        prescription_text = "Rx: Metformin 500mg\nSig: Take twice daily\nDr. Smith"
        doc_type = ocr_processor.detect_document_type(prescription_text)
        assert doc_type == DocumentType.PRESCRIPTION
        
        # Test radiology report detection
        radiology_text = "RADIOLOGY REPORT\nCT Scan of Chest\nFindings: No acute abnormalities"
        doc_type = ocr_processor.detect_document_type(radiology_text)
        assert doc_type == DocumentType.RADIOLOGY_REPORT

    # Test 2: Document Upload and Validation
    async def test_file_upload_validation(self, file_handler):
        """Test file upload validation and security checks."""
        # Test valid file types
        valid_files = [
            "test_report.pdf",
            "lab_results.jpg",
            "prescription.png",
            "medical_form.docx"
        ]
        
        for filename in valid_files:
            is_valid = file_handler.validate_file_type(filename)
            assert is_valid, f"File {filename} should be valid"
        
        # Test invalid file types
        invalid_files = [
            "malicious.exe",
            "script.js",
            "document.bat"
        ]
        
        for filename in invalid_files:
            is_valid = file_handler.validate_file_type(filename)
            assert not is_valid, f"File {filename} should be invalid"

    async def test_file_size_validation(self, file_handler):
        """Test file size validation and limits."""
        # Test file size limits
        max_size = 50 * 1024 * 1024  # 50MB
        file_handler.set_max_file_size(max_size)
        
        # Test valid file size
        valid_size = 10 * 1024 * 1024  # 10MB
        assert file_handler.validate_file_size(valid_size)
        
        # Test oversized file
        oversized = 100 * 1024 * 1024  # 100MB
        assert not file_handler.validate_file_size(oversized)

    async def test_file_security_scanning(self, file_handler):
        """Test file security scanning and malware detection."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Create a safe test file
            temp_file.write(b"PDF test content")
            temp_path = temp_file.name
        
        try:
            # Test security scan
            is_safe = await file_handler.security_scan(temp_path)
            assert is_safe
            
            # Test file quarantine for suspicious content
            with patch.object(file_handler, '_detect_malicious_content') as mock_scan:
                mock_scan.return_value = True
                is_safe = await file_handler.security_scan(temp_path)
                assert not is_safe
        finally:
            os.unlink(temp_path)

    # Test 3: Multi-format Text Extraction
    async def test_pdf_text_extraction(self, ocr_processor):
        """Test PDF text extraction with different PDF types."""
        # Mock PDF processing
        with patch('app.services.ocr_processor.pdf2image.convert_from_path') as mock_pdf2image:
            mock_image = Image.new('RGB', (800, 600), color='white')
            mock_pdf2image.return_value = [mock_image]
            
            with patch.object(ocr_processor, '_extract_text_from_image') as mock_extract:
                mock_extract.return_value = {
                    "text": "MEDICAL REPORT\nPatient: John Doe\nDiagnosis: Type 2 Diabetes",
                    "confidence": 0.95,
                    "regions": [{"text": "MEDICAL REPORT", "bbox": [0, 0, 200, 50]}]
                }
                
                # Test PDF extraction
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                    temp_path = temp_pdf.name
                
                try:
                    request = OCRRequest(
                        document_id="test_pdf",
                        file_path=temp_path,
                        document_type=DocumentType.MEDICAL_REPORT
                    )
                    
                    response = await ocr_processor.process_document(request)
                    
                    assert response.status == OCRStatus.COMPLETED
                    assert "MEDICAL REPORT" in response.extracted_text
                    assert response.confidence_score >= 0.9
                    assert response.page_count == 1
                finally:
                    os.unlink(temp_path)

    async def test_image_text_extraction(self, ocr_processor, sample_medical_image):
        """Test text extraction from medical images."""
        with patch.object(ocr_processor, '_extract_text_with_tesseract') as mock_tesseract:
            mock_tesseract.return_value = {
                "text": "Blood Pressure: 140/90 mmHg\nHeart Rate: 78 bpm\nTemperature: 98.6°F",
                "confidence": 0.92,
                "word_confidences": [0.95, 0.90, 0.88, 0.94],
                "regions": [
                    {"text": "Blood Pressure: 140/90 mmHg", "bbox": [10, 20, 300, 40]},
                    {"text": "Heart Rate: 78 bpm", "bbox": [10, 50, 200, 70]}
                ]
            }
            
            # Save test image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                sample_medical_image.save(temp_img.name)
                temp_path = temp_img.name
            
            try:
                request = OCRRequest(
                    document_id="test_image",
                    file_path=temp_path,
                    document_type=DocumentType.VITAL_SIGNS
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert "Blood Pressure" in response.extracted_text
                assert response.confidence_score >= 0.9
                assert len(response.text_regions) >= 2
            finally:
                os.unlink(temp_path)

    async def test_docx_text_extraction(self, ocr_processor):
        """Test text extraction from DOCX medical documents."""
        with patch('docx.Document') as mock_docx:
            # Mock DOCX document
            mock_doc = Mock()
            mock_paragraph = Mock()
            mock_paragraph.text = "PATIENT HISTORY\nJohn Doe, 65 years old\nHistory of diabetes and hypertension"
            mock_doc.paragraphs = [mock_paragraph]
            mock_docx.return_value = mock_doc
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
                temp_path = temp_docx.name
            
            try:
                request = OCRRequest(
                    document_id="test_docx",
                    file_path=temp_path,
                    document_type=DocumentType.PATIENT_HISTORY
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert "PATIENT HISTORY" in response.extracted_text
                assert "John Doe" in response.extracted_text
            finally:
                os.unlink(temp_path)

    async def test_excel_lab_results_extraction(self, ocr_processor):
        """Test extraction of lab results from Excel files."""
        with patch('pandas.read_excel') as mock_pandas:
            # Mock Excel data
            mock_df = Mock()
            mock_df.to_string.return_value = "Test_Name    Result    Reference_Range\nHbA1c        7.8%      <7.0%\nGlucose      165       70-100 mg/dL"
            mock_pandas.return_value = mock_df
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_xlsx:
                temp_path = temp_xlsx.name
            
            try:
                request = OCRRequest(
                    document_id="test_xlsx",
                    file_path=temp_path,
                    document_type=DocumentType.LAB_REPORT
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert "HbA1c" in response.extracted_text
                assert "7.8%" in response.extracted_text
                assert response.table_data is not None
            finally:
                os.unlink(temp_path)

    # Test 4: Medical Document Optimization
    async def test_medical_image_preprocessing(self, ocr_processor, sample_medical_image):
        """Test medical-specific image preprocessing."""
        with patch.object(ocr_processor, '_apply_medical_preprocessing') as mock_preprocess:
            # Mock preprocessing result
            enhanced_image = Image.new('RGB', (800, 600), color='white')
            mock_preprocess.return_value = {
                "image": enhanced_image,
                "preprocessing_steps": [
                    PreprocessingStep.SKEW_CORRECTION,
                    PreprocessingStep.NOISE_REDUCTION,
                    PreprocessingStep.CONTRAST_ENHANCEMENT
                ],
                "quality_score": 0.92
            }
            
            # Test preprocessing
            result = await ocr_processor._preprocess_medical_image(
                sample_medical_image, 
                DocumentType.LAB_REPORT
            )
            
            assert result["quality_score"] >= 0.9
            assert PreprocessingStep.CONTRAST_ENHANCEMENT in result["preprocessing_steps"]

    async def test_prescription_handwriting_optimization(self, ocr_processor, prescription_ocr_config):
        """Test prescription handwriting recognition optimization."""
        with patch.object(ocr_processor, '_optimize_for_handwriting') as mock_handwriting:
            mock_handwriting.return_value = {
                "text": "Metformin 500mg\nTake twice daily with meals\nDr. Smith",
                "confidence": 0.87,
                "handwriting_regions": [
                    {"text": "Dr. Smith", "confidence": 0.82, "is_handwritten": True}
                ]
            }
            
            # Create mock prescription image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                sample_image = Image.new('RGB', (600, 800), color='white')
                sample_image.save(temp_img.name)
                temp_path = temp_img.name
            
            try:
                request = OCRRequest(
                    document_id="test_prescription",
                    file_path=temp_path,
                    configuration=prescription_ocr_config
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert "Metformin" in response.extracted_text
                assert response.handwriting_detected is True
            finally:
                os.unlink(temp_path)

    async def test_lab_report_table_extraction(self, ocr_processor, lab_report_ocr_config):
        """Test table extraction from lab reports."""
        with patch.object(ocr_processor, '_extract_tables') as mock_tables:
            mock_tables.return_value = [
                {
                    "table_id": 1,
                    "headers": ["Test Name", "Result", "Reference Range", "Flag"],
                    "rows": [
                        ["HbA1c", "7.8%", "<7.0%", "HIGH"],
                        ["Glucose", "165 mg/dL", "70-100 mg/dL", "HIGH"],
                        ["Cholesterol", "180 mg/dL", "<200 mg/dL", "NORMAL"]
                    ],
                    "confidence": 0.94
                }
            ]
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_path = temp_pdf.name
            
            try:
                request = OCRRequest(
                    document_id="test_lab_report",
                    file_path=temp_path,
                    configuration=lab_report_ocr_config
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert response.table_data is not None
                assert len(response.table_data) == 1
                assert response.table_data[0]["headers"][0] == "Test Name"
            finally:
                os.unlink(temp_path)

    # Test 5: OCR Engine Performance and Quality
    async def test_tesseract_engine_performance(self, ocr_processor):
        """Test Tesseract OCR engine performance and accuracy."""
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "MEDICAL REPORT\nPatient: Jane Doe\nDate: 2024-01-15"
            
            with patch('pytesseract.image_to_data') as mock_data:
                mock_data.return_value = {
                    'text': ['MEDICAL', 'REPORT', 'Patient:', 'Jane', 'Doe'],
                    'conf': [95, 92, 88, 94, 96]
                }
                
                # Test processing with performance metrics
                start_time = datetime.now()
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                    test_image = Image.new('RGB', (800, 600), color='white')
                    test_image.save(temp_img.name)
                    temp_path = temp_img.name
                
                try:
                    request = OCRRequest(
                        document_id="performance_test",
                        file_path=temp_path,
                        configuration=OCRConfiguration(engine=OCREngine.TESSERACT)
                    )
                    
                    response = await ocr_processor.process_document(request)
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    assert response.status == OCRStatus.COMPLETED
                    assert response.processing_time > 0
                    assert processing_time < 10  # Should complete within 10 seconds
                    assert response.confidence_score >= 0.85
                finally:
                    os.unlink(temp_path)

    async def test_azure_form_recognizer_integration(self, ocr_processor):
        """Test Azure Form Recognizer integration for complex documents."""
        with patch('azure.ai.formrecognizer.DocumentAnalysisClient') as mock_azure:
            # Mock Azure Form Recognizer response
            mock_client = Mock()
            mock_result = Mock()
            mock_result.content = "MEDICAL FORM\nPatient Information\nName: John Smith"
            mock_result.tables = []
            mock_result.pages = [Mock()]
            mock_result.pages[0].lines = [
                Mock(content="MEDICAL FORM"),
                Mock(content="Patient Information"),
                Mock(content="Name: John Smith")
            ]
            
            mock_client.begin_analyze_document.return_value.result.return_value = mock_result
            mock_azure.return_value = mock_client
            
            # Test Azure processing
            azure_config = OCRConfiguration(
                engine=OCREngine.AZURE_FORM_RECOGNIZER,
                confidence_threshold=0.9
            )
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_path = temp_pdf.name
            
            try:
                request = OCRRequest(
                    document_id="azure_test",
                    file_path=temp_path,
                    configuration=azure_config
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.COMPLETED
                assert "MEDICAL FORM" in response.extracted_text
                assert response.confidence_score >= 0.9
            finally:
                os.unlink(temp_path)

    # Test 6: Error Handling and Recovery
    async def test_ocr_processing_failure_recovery(self, ocr_processor):
        """Test OCR processing failure recovery mechanisms."""
        with patch.object(ocr_processor, '_extract_text_with_tesseract') as mock_tesseract:
            # Simulate OCR engine failure
            mock_tesseract.side_effect = Exception("Tesseract engine failed")
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                test_image = Image.new('RGB', (400, 300), color='white')
                test_image.save(temp_img.name)
                temp_path = temp_img.name
            
            try:
                request = OCRRequest(
                    document_id="failure_test",
                    file_path=temp_path,
                    retry_attempts=3
                )
                
                response = await ocr_processor.process_document(request)
                
                assert response.status == OCRStatus.FAILED
                assert "Tesseract engine failed" in response.error_message
                assert response.retry_count == 3
            finally:
                os.unlink(temp_path)

    async def test_corrupted_file_handling(self, ocr_processor):
        """Test handling of corrupted or invalid files."""
        # Create a corrupted file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"This is not a valid PDF file")
            temp_path = temp_file.name
        
        try:
            request = OCRRequest(
                document_id="corrupted_test",
                file_path=temp_path
            )
            
            response = await ocr_processor.process_document(request)
            
            assert response.status == OCRStatus.FAILED
            assert "corrupted" in response.error_message.lower() or "invalid" in response.error_message.lower()
        finally:
            os.unlink(temp_path)

    async def test_memory_management_large_files(self, ocr_processor):
        """Test memory management with large document files."""
        with patch.object(ocr_processor, '_monitor_memory_usage') as mock_memory:
            mock_memory.return_value = {
                "current_usage": "512MB",
                "peak_usage": "768MB",
                "available": "1GB"
            }
            
            # Simulate large file processing
            large_file_config = OCRConfiguration(
                memory_limit="1GB",
                streaming_mode=True,
                chunk_processing=True
            )
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_path = temp_pdf.name
            
            try:
                # Create request with memory limits
                await ocr_processor.process_document(OCRRequest(
                    document_id="large_file_test",
                    file_path=temp_path,
                    configuration=large_file_config
                ))
                
                
                # Verify memory monitoring was called
                mock_memory.assert_called()
            finally:
                os.unlink(temp_path)

    # Test 7: Quality Assessment and Validation
    async def test_text_quality_assessment(self, ocr_processor):
        """Test text quality assessment and confidence scoring."""
        # Test high-quality text
        high_quality_result = {
            "text": "LABORATORY REPORT\nPatient: John Doe\nHbA1c: 7.8%",
            "word_confidences": [0.98, 0.96, 0.94, 0.92, 0.95, 0.89],
            "character_confidences": [0.99, 0.97, 0.95, 0.93, 0.94]
        }
        
        quality_score = ocr_processor._assess_text_quality(high_quality_result)
        assert quality_score >= 0.9
        
        # Test low-quality text
        low_quality_result = {
            "text": "L@B0R@T0RY REP0RT\nP@t1ent: J0hn D0e",
            "word_confidences": [0.65, 0.58, 0.72, 0.64, 0.69],
            "character_confidences": [0.60, 0.55, 0.70, 0.62, 0.58]
        }
        
        quality_score = ocr_processor._assess_text_quality(low_quality_result)
        assert quality_score < 0.7

    async def test_medical_terminology_validation(self, ocr_processor):
        """Test medical terminology validation and correction."""
        with patch.object(ocr_processor, '_validate_medical_terms') as mock_validate:
            mock_validate.return_value = {
                "original_text": "Diabetis Melitus Type 2",
                "corrected_text": "Diabetes Mellitus Type 2",
                "corrections": [
                    {"original": "Diabetis", "corrected": "Diabetes", "confidence": 0.95},
                    {"original": "Melitus", "corrected": "Mellitus", "confidence": 0.92}
                ],
                "medical_terms_found": ["Diabetes Mellitus", "Type 2"]
            }
            
            # Test medical terminology validation
            result = await ocr_processor._validate_and_correct_medical_text(
                "Diabetis Melitus Type 2"
            )
            
            assert result["corrected_text"] == "Diabetes Mellitus Type 2"
            assert len(result["corrections"]) == 2
            assert "Diabetes Mellitus" in result["medical_terms_found"]

    # Test 8: Performance Optimization
    async def test_concurrent_document_processing(self, ocr_processor):
        """Test concurrent processing of multiple documents."""
        # Create multiple test documents
        test_documents = []
        temp_files = []
        
        for i in range(5):
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            test_image = Image.new('RGB', (400, 300), color='white')
            test_image.save(temp_file.name)
            temp_files.append(temp_file.name)
            
            test_documents.append(OCRRequest(
                document_id=f"concurrent_test_{i}",
                file_path=temp_file.name
            ))
        
        try:
            # Mock OCR processing
            with patch.object(ocr_processor, '_extract_text_with_tesseract') as mock_extract:
                mock_extract.return_value = {
                    "text": "Test document text",
                    "confidence": 0.9
                }
                
                # Process documents concurrently
                import asyncio
                tasks = []
                for request in test_documents:
                    task = asyncio.create_task(ocr_processor.process_document(request))
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                
                # Verify all documents were processed
                assert len(responses) == 5
                for response in responses:
                    assert response.status == OCRStatus.COMPLETED
        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

    async def test_batch_processing_optimization(self, ocr_processor):
        """Test batch processing optimization for multiple similar documents."""
        # Create batch of similar documents
        batch_requests = []
        temp_files = []
        
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_files.append(temp_file.name)
            
            batch_requests.append(OCRRequest(
                document_id=f"batch_doc_{i}",
                file_path=temp_file.name,
                document_type=DocumentType.LAB_REPORT
            ))
        
        try:
            with patch.object(ocr_processor, '_process_batch_documents') as mock_batch:
                mock_batch.return_value = [
                    OCRResponse(
                        document_id=f"batch_doc_{i}",
                        status=OCRStatus.COMPLETED,
                        extracted_text=f"Lab report {i} content",
                        confidence_score=0.92
                    ) for i in range(3)
                ]
                
                # Process batch
                responses = await ocr_processor.process_document_batch(batch_requests)
                
                assert len(responses) == 3
                for response in responses:
                    assert response.status == OCRStatus.COMPLETED
                    assert response.confidence_score >= 0.9
        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)

    # Test 9: Integration with Supervisor
    async def test_supervisor_integration(self, ocr_processor):
        """Test integration with supervisor workflow."""
        # Mock supervisor callback
        supervisor_callback = AsyncMock()
        ocr_processor.set_supervisor_callback(supervisor_callback)
        
        with patch.object(ocr_processor, '_extract_text_with_tesseract') as mock_extract:
            mock_extract.return_value = {
                "text": "Integration test document",
                "confidence": 0.94
            }
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                test_image = Image.new('RGB', (400, 300), color='white')
                test_image.save(temp_img.name)
                temp_path = temp_img.name
            
            try:
                request = OCRRequest(
                    document_id="supervisor_integration_test",
                    file_path=temp_path,
                    job_id="supervisor_job_123"
                )
                
                response = await ocr_processor.process_document(request)
                
                # Verify supervisor callback was called
                supervisor_callback.assert_called()
                
                # Verify response contains job context
                assert response.job_id == "supervisor_job_123"
                assert response.status == OCRStatus.COMPLETED
            finally:
                os.unlink(temp_path)

    # Test 10: Configuration and Customization
    async def test_custom_preprocessing_pipeline(self, ocr_processor):
        """Test custom preprocessing pipeline configuration."""
        custom_config = OCRConfiguration(
            preprocessing=ImagePreprocessing.CUSTOM,
            custom_preprocessing_steps=[
                PreprocessingStep.SKEW_CORRECTION,
                PreprocessingStep.NOISE_REDUCTION,
                PreprocessingStep.CONTRAST_ENHANCEMENT,
                PreprocessingStep.SHARPENING,
                PreprocessingStep.MEDICAL_OPTIMIZATION
            ],
            preprocessing_params={
                "contrast_factor": 1.5,
                "noise_reduction_strength": 0.8,
                "sharpening_radius": 2.0
            }
        )
        
        # Test custom preprocessing configuration
        ocr_processor.configure_preprocessing(custom_config)
        
        assert ocr_processor.preprocessing_config.preprocessing == ImagePreprocessing.CUSTOM
        assert len(ocr_processor.preprocessing_config.custom_preprocessing_steps) == 5
        assert ocr_processor.preprocessing_config.preprocessing_params["contrast_factor"] == 1.5

    async def test_medical_specialty_optimization(self, ocr_processor):
        """Test medical specialty-specific optimization."""
        # Test cardiology optimization
        cardiology_config = OCRConfiguration(
            medical_specialty="cardiology",
            specialized_terminology=True,
            boost_medical_terms=["ECG", "EKG", "cardiac", "myocardial", "arrhythmia"]
        )
        
        ocr_processor.configure_medical_specialty(cardiology_config)
        
        assert ocr_processor.medical_specialty == "cardiology"
        assert "ECG" in ocr_processor.boosted_terms
        assert "cardiac" in ocr_processor.boosted_terms
        
        # Test radiology optimization
        radiology_config = OCRConfiguration(
            medical_specialty="radiology",
            specialized_terminology=True,
            boost_medical_terms=["CT", "MRI", "X-ray", "ultrasound", "contrast"]
        )
        
        ocr_processor.configure_medical_specialty(radiology_config)
        
        assert ocr_processor.medical_specialty == "radiology"
        assert "CT" in ocr_processor.boosted_terms
        assert "MRI" in ocr_processor.boosted_terms
