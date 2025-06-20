"""
Comprehensive test suite for common utilities.

This module tests all utility functions in app.common.utils including:
- Logging functionality
- File handling utilities  
- Data processing helpers
- Medical domain specific utilities
- Configuration management
- Error handling utilities
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from app.common.utils import (
    get_logger, ensure_directory, safe_filename, format_file_size,
    parse_duration, create_error_response, load_json_config, save_json_config,
    calculate_confidence_score, truncate_text, batch_items, sanitize_patient_id,
    format_processing_time, sanitize_filename, validate_patient_id,
    get_file_category, create_response_metadata, get_current_timestamp,
    format_timestamp, MEDICAL_DOCUMENT_TYPES, FILE_SIZE_LIMITS,
    SUPPORTED_EXTENSIONS
)


class TestLoggingUtilities:
    """Test cases for logging utility functions."""
    
    def test_get_logger_creates_new_logger(self):
        """Test creation of new logger with proper configuration."""
        logger = get_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) == 1
        assert logger.handlers[0].__class__.__name__ == "StreamHandler"
    
    def test_get_logger_reuses_existing_logger(self):
        """Test that existing loggers are reused."""
        logger1 = get_logger("test_reuse")
        logger2 = get_logger("test_reuse")
        
        assert logger1 is logger2
        assert len(logger1.handlers) == 1  # Should not duplicate handlers
    
    def test_logger_formatter_configuration(self):
        """Test logger formatter is properly configured."""
        logger = get_logger("test_formatter")
        handler = logger.handlers[0]
        formatter = handler.formatter
        
        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt


class TestFileHandlingUtilities:
    """Test cases for file handling utility functions."""
    
    def test_ensure_directory_creates_new_directory(self):
        """Test creation of new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "test_dir" / "nested_dir"
            
            result = ensure_directory(new_dir)
            
            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_ensure_directory_handles_existing_directory(self):
        """Test handling of existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)
            
            result = ensure_directory(existing_dir)
            
            assert result == existing_dir
            assert existing_dir.exists()
    
    def test_safe_filename_removes_unsafe_characters(self):
        """Test removal of unsafe characters from filename."""
        unsafe_filename = "test<>:file|?.txt"
        
        safe_name = safe_filename(unsafe_filename)
        
        assert safe_name == "test___file__.txt"
        assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in safe_name)
    
    def test_safe_filename_handles_dot_prefix(self):
        """Test handling of filenames starting with dot."""
        dot_filename = ".hidden_file"
        
        safe_name = safe_filename(dot_filename)
        
        assert safe_name == "file_.hidden_file"
        assert not safe_name.startswith(".")
    
    def test_format_file_size_various_sizes(self):
        """Test file size formatting for various sizes."""
        test_cases = [
            (0, "0 B"),
            (512, "512.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB"),
            (1099511627776, "1.0 TB")
        ]
        
        for size_bytes, expected in test_cases:
            result = format_file_size(size_bytes)
            assert result == expected
    
    def test_sanitize_filename_comprehensive(self):
        """Test comprehensive filename sanitization."""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file<>:with|bad?chars*.txt", "file_with_bad_chars_.txt"),
            ("file___with___multiple___underscores.txt", "file_with_multiple_underscores.txt"),
            ("___leading_trailing___", "leading_trailing"),
            ("", "unnamed_file"),
            ("file/with\\path:separators", "file_with_path_separators")
        ]
        
        for input_filename, expected in test_cases:
            result = sanitize_filename(input_filename)
            assert result == expected


class TestDataProcessingUtilities:
    """Test cases for data processing utility functions."""
    
    def test_parse_duration_various_formats(self):
        """Test parsing of duration strings in various formats."""
        test_cases = [
            ("30s", 30),
            ("5m", 300),
            ("2h", 7200),
            ("45", 45),  # Assume seconds
            ("invalid", None),
            ("", None)
        ]
        
        for duration_str, expected in test_cases:
            result = parse_duration(duration_str)
            assert result == expected
    
    def test_calculate_confidence_score_empty_list(self):
        """Test confidence score calculation with empty list."""
        result = calculate_confidence_score([])
        assert result == 0.0
    
    def test_calculate_confidence_score_normal_values(self):
        """Test confidence score calculation with normal values."""
        scores = [0.8, 0.9, 0.7, 0.85]
        
        result = calculate_confidence_score(scores)
        
        assert 0.0 <= result <= 1.0
        assert result > 0.7  # Should be reasonably high for good scores
    
    def test_calculate_confidence_score_filters_outliers(self):
        """Test that confidence score filters out low outliers."""
        scores = [0.05, 0.02, 0.8, 0.9, 0.85]  # Low outliers should be filtered
        
        result = calculate_confidence_score(scores)
        
        assert result > 0.8  # Should ignore the low scores
    
    def test_truncate_text_within_limit(self):
        """Test text truncation when within limit."""
        text = "Short text"
        
        result = truncate_text(text, max_length=100)
        
        assert result == text
    
    def test_truncate_text_exceeds_limit(self):
        """Test text truncation when exceeding limit."""
        text = "This is a very long text that exceeds the maximum length limit"
        
        result = truncate_text(text, max_length=20, suffix="...")
        
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a very l..."
    
    def test_batch_items_even_batches(self):
        """Test batching items with even batch sizes."""
        items = list(range(10))
        
        batches = list(batch_items(items, batch_size=3))
        
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]
    
    def test_batch_items_empty_list(self):
        """Test batching with empty list."""
        items = []
        
        batches = list(batch_items(items, batch_size=3))
        
        assert batches == []


class TestMedicalDomainUtilities:
    """Test cases for medical domain specific utilities."""
    
    def test_sanitize_patient_id_valid_input(self):
        """Test patient ID sanitization with valid input."""
        patient_id = "PAT-123_ABC"
        
        result = sanitize_patient_id(patient_id)
        
        assert result == "PAT-123_ABC"
    
    def test_sanitize_patient_id_invalid_characters(self):
        """Test patient ID sanitization with invalid characters."""
        patient_id = "PAT@123#ABC$"
        
        result = sanitize_patient_id(patient_id)
        
        assert result == "PAT123ABC"
    
    def test_sanitize_patient_id_empty_input(self):
        """Test patient ID sanitization with empty input."""
        patient_id = "!@#$%"
        
        result = sanitize_patient_id(patient_id)
        
        assert result == "unknown_patient"
    
    def test_validate_patient_id_valid_cases(self):
        """Test patient ID validation with valid cases."""
        valid_ids = [
            "PAT123",
            "patient-456",
            "USER_789",
            "ABC123DEF456",
            "a"  # Single character
        ]
        
        for patient_id in valid_ids:
            assert validate_patient_id(patient_id) is True
    
    def test_validate_patient_id_invalid_cases(self):
        """Test patient ID validation with invalid cases."""
        invalid_ids = [
            "",  # Empty
            "PAT@123",  # Special characters
            "patient id",  # Space
            "a" * 51,  # Too long
            "PAT/123",  # Forward slash
        ]
        
        for patient_id in invalid_ids:
            assert validate_patient_id(patient_id) is False
    
    def test_medical_document_types_constants(self):
        """Test medical document types constants."""
        assert 'lab_report' in MEDICAL_DOCUMENT_TYPES
        assert 'prescription' in MEDICAL_DOCUMENT_TYPES
        assert MEDICAL_DOCUMENT_TYPES['lab_report'] == 'Laboratory Report'
    
    def test_file_size_limits_constants(self):
        """Test file size limits constants."""
        assert 'image' in FILE_SIZE_LIMITS
        assert 'pdf' in FILE_SIZE_LIMITS
        assert FILE_SIZE_LIMITS['image'] == 10 * 1024 * 1024
    
    def test_supported_extensions_constants(self):
        """Test supported extensions constants."""
        assert '.pdf' in SUPPORTED_EXTENSIONS
        assert '.jpg' in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS['.pdf'] == 'application/pdf'


class TestConfigurationUtilities:
    """Test cases for configuration management utilities."""
    
    def test_load_json_config_valid_file(self):
        """Test loading valid JSON configuration file."""
        config_data = {"key1": "value1", "key2": 42}
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            result = load_json_config(Path("test_config.json"))
            
            assert result == config_data
    
    def test_load_json_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_json_config(Path("nonexistent.json"))
    
    def test_load_json_config_invalid_json(self):
        """Test loading invalid JSON configuration."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with pytest.raises(json.JSONDecodeError):
                load_json_config(Path("invalid.json"))
    
    def test_save_json_config(self):
        """Test saving configuration to JSON file."""
        config_data = {"test": "data", "number": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nested" / "config.json"
            
            save_json_config(config_data, config_path)
            
            assert config_path.exists()
            with open(config_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == config_data


class TestErrorHandlingUtilities:
    """Test cases for error handling utilities."""
    
    def test_create_error_response_basic(self):
        """Test basic error response creation."""
        error = ValueError("Test error message")
        
        response = create_error_response(error)
        
        assert response["error"] == "Test error message"
        assert response["error_type"] == "ValueError"
        assert "timestamp" in response
        assert isinstance(response["timestamp"], str)
    
    def test_create_error_response_with_context(self):
        """Test error response creation with context."""
        error = FileNotFoundError("File not found")
        context = {"file_path": "/test/path", "operation": "read"}
        
        response = create_error_response(error, context)
        
        assert response["error"] == "File not found"
        assert response["error_type"] == "FileNotFoundError"
        assert response["context"] == context
    
    def test_create_response_metadata_basic(self):
        """Test basic response metadata creation."""
        metadata = create_response_metadata()
        
        assert metadata["success"] is True
        assert "timestamp" in metadata
        assert "server_time" in metadata
    
    def test_create_response_metadata_with_processing_time(self):
        """Test response metadata with processing time."""
        metadata = create_response_metadata(
            success=False,
            processing_time=2.5,
            additional_info={"agent": "ocr", "document_count": 3}
        )
        
        assert metadata["success"] is False
        assert "processing_time" in metadata
        assert metadata["agent"] == "ocr"
        assert metadata["document_count"] == 3


class TestTimeUtilities:
    """Test cases for time-related utilities."""
    
    def test_format_processing_time_milliseconds(self):
        """Test formatting time in milliseconds."""
        result = format_processing_time(0.123)
        assert result == "123ms"
    
    def test_format_processing_time_seconds(self):
        """Test formatting time in seconds."""
        result = format_processing_time(5.7)
        assert result == "5.7 seconds"
    
    def test_format_processing_time_minutes(self):
        """Test formatting time in minutes and seconds."""
        result = format_processing_time(125.0)  # 2 minutes 5 seconds
        assert result == "2m 5s"
    
    def test_format_processing_time_hours(self):
        """Test formatting time in hours and minutes."""
        result = format_processing_time(7265.0)  # 2 hours 1 minute
        assert result == "2h 1m"
    
    def test_get_current_timestamp_format(self):
        """Test current timestamp format."""
        timestamp = get_current_timestamp()
        
        # Should be able to parse as ISO format
        datetime.fromisoformat(timestamp.replace('Z', '+00:00') if timestamp.endswith('Z') else timestamp)
        assert isinstance(timestamp, str)
    
    def test_format_timestamp(self):
        """Test datetime formatting to timestamp."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        
        result = format_timestamp(dt)
        
        assert result == "2024-01-15T10:30:45"


class TestFileTypeUtilities:
    """Test cases for file type utilities."""
    
    def test_get_file_category_image(self):
        """Test file category detection for images."""
        test_cases = [
            ("image/jpeg", "image"),
            ("image/png", "image"),
            ("image/tiff", "image")
        ]
        
        for mime_type, expected in test_cases:
            result = get_file_category(mime_type)
            assert result == expected
    
    def test_get_file_category_pdf(self):
        """Test file category detection for PDF."""
        result = get_file_category("application/pdf")
        assert result == "pdf"
    
    def test_get_file_category_office_document(self):
        """Test file category detection for Office documents."""
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        result = get_file_category(mime_type)
        
        assert result == "office_document"
    
    def test_get_file_category_text(self):
        """Test file category detection for text files."""
        result = get_file_category("text/plain")
        assert result == "text"
    
    def test_get_file_category_other(self):
        """Test file category detection for unknown types."""
        result = get_file_category("application/unknown")
        assert result == "other"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
