"""
Common utility functions for the MRIA application.

This module provides shared utilities including logging configuration,
file handling helpers, and other common functionality used across the application.
"""

import logging
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the ensured directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove or replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Ensure it doesn't start with a dot
    if safe_name.startswith('.'):
        safe_name = 'file_' + safe_name
        
    return safe_name


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human-readable file size string
    """
    if size_bytes == 0:
        return "0 B"
        
    size_names = ["B", "KB", "MB", "GB", "TB"]
    
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.1f} {size_names[i]}"


def parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., "5m", "30s", "2h")
        
    Returns:
        Duration in seconds or None if parsing fails
    """
    try:
        if duration_str.endswith('s'):
            return int(duration_str[:-1])
        elif duration_str.endswith('m'):
            return int(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return int(duration_str[:-1]) * 3600
        else:
            return int(duration_str)  # Assume seconds
    except (ValueError, IndexError):
        return None


def create_error_response(
    error: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: Exception that occurred
        context: Optional additional context
        
    Returns:
        Standardized error response dictionary
    """
    error_response = {
        "error": str(error),
        "error_type": type(error).__name__,
        "timestamp": datetime.now().isoformat(),
    }
    
    if context:
        error_response["context"] = context
        
    return error_response


def load_json_config(config_path: Path) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def calculate_confidence_score(scores: list) -> float:
    """
    Calculate overall confidence score from a list of individual scores.
    
    Args:
        scores: List of confidence scores (0.0 to 1.0)
        
    Returns:
        Overall confidence score
    """
    if not scores:
        return 0.0
        
    # Remove outliers (scores below 0.1) for better overall score
    filtered_scores = [s for s in scores if s >= 0.1]
    
    if not filtered_scores:
        return 0.0
        
    # Calculate weighted average (higher scores get more weight)
    weights = [s * s for s in filtered_scores]  # Square for weighting
    weighted_sum = sum(s * w for s, w in zip(filtered_scores, weights))
    weight_total = sum(weights)
    
    if weight_total == 0:
        return 0.0
        
    return weighted_sum / weight_total


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of result
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix


def batch_items(items: list, batch_size: int):
    """
    Yield successive batch_size chunks from items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def sanitize_patient_id(patient_id: str) -> str:
    """
    Sanitize patient ID for safe use in filenames and URLs.
    
    Args:
        patient_id: Raw patient identifier
        
    Returns:
        Sanitized patient ID
    """
    # Keep alphanumeric characters, hyphens, and underscores
    sanitized = ''.join(c for c in patient_id if c.isalnum() or c in '-_')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'unknown_patient'
        
    return sanitized


def measure_execution_time(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger = get_logger(func.__module__)
            logger.info(f"{func.__name__} executed in {duration:.2f} seconds")
    
    return wrapper


def format_processing_time(seconds: float) -> str:
    """
    Format processing time in human-readable format.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Human-readable time string
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Trim underscores from ends
    filename = filename.strip('_')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename


def validate_patient_id(patient_id: str) -> bool:
    """
    Validate patient ID format.
    
    Args:
        patient_id: Patient identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validation - alphanumeric, dashes, underscores allowed
    pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    return bool(re.match(pattern, patient_id))


def get_file_category(mime_type: str) -> str:
    """
    Determine file category based on MIME type.
    
    Args:
        mime_type: MIME type of the file
        
    Returns:
        File category string
    """
    if mime_type.startswith('image/'):
        return 'image'
    elif mime_type == 'application/pdf':
        return 'pdf'
    elif mime_type.startswith('application/vnd.openxmlformats-officedocument'):
        return 'office_document'
    elif mime_type.startswith('text/'):
        return 'text'
    else:
        return 'other'


def create_response_metadata(
    success: bool = True,
    processing_time: Optional[float] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized response metadata.
    
    Args:
        success: Whether the operation was successful
        processing_time: Processing time in seconds
        additional_info: Additional metadata
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'success': success,
        'timestamp': datetime.utcnow().isoformat(),
        'server_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    if processing_time is not None:
        metadata['processing_time'] = format_processing_time(processing_time)
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        Current timestamp as ISO format string
    """
    return datetime.now().isoformat()


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime object to ISO string.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        ISO format timestamp string
    """
    return dt.isoformat()


# Medical-specific constants
MEDICAL_DOCUMENT_TYPES = {
    'lab_report': 'Laboratory Report',
    'prescription': 'Prescription',
    'medical_report': 'Medical Report',
    'radiology': 'Radiology Report',
    'discharge_summary': 'Discharge Summary',
    'clinical_notes': 'Clinical Notes',
    'consent_form': 'Consent Form',
    'insurance_claim': 'Insurance Claim',
    'other': 'Other Medical Document'
}

# File size limits (in bytes)
FILE_SIZE_LIMITS = {
    'image': 10 * 1024 * 1024,  # 10MB for images
    'pdf': 50 * 1024 * 1024,    # 50MB for PDFs
    'office_document': 25 * 1024 * 1024,  # 25MB for Office docs
    'text': 5 * 1024 * 1024,    # 5MB for text files
    'default': 50 * 1024 * 1024  # 50MB default
}

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.txt': 'text/plain'
}
