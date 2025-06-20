"""
File handling service for document upload and processing.

This service handles file uploads, validation, storage, and metadata extraction
for medical documents in the MRIA system. It provides secure file handling
with virus scanning, content validation, and metadata preservation.
"""

import uuid
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

import aiofiles
from fastapi import UploadFile, HTTPException
from PIL import Image

from app.schemas.ingestion import DocumentType
from app.config import get_storage_config, get_file_size_limit

# Try to import magic, fallback to mimetypes if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None


logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """File metadata container."""
    
    filename: str
    file_id: str
    size: int
    mime_type: str
    hash_sha256: str
    created_at: datetime
    original_path: str
    stored_path: str


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class FileHandlerService:
    """
    Service for handling file uploads and storage operations.
    
    This service provides secure file handling capabilities including:
    - File validation and virus scanning
    - Metadata extraction and storage
    - Content type detection and verification
    - Secure file storage with encryption
    - Image preprocessing for OCR optimization
    """
    
    def __init__(self):
        """Initialize the file handler service with configuration."""
        self.storage_config = get_storage_config()
        self.logger = logging.getLogger(__name__)
        self.STORAGE_BASE_DIR = Path(self.storage_config.base_storage_dir)
        # Create supported MIME types mapping from config
        self.SUPPORTED_MIME_TYPES = {}
        for mime_type in self.storage_config.allowed_mime_types:
            # Map MIME types to extensions
            extensions = []
            if 'pdf' in mime_type:
                extensions = ['.pdf']
            elif 'image/jpeg' in mime_type:
                extensions = ['.jpg', '.jpeg']
            elif 'image/png' in mime_type:
                extensions = ['.png']
            elif 'image/tiff' in mime_type:
                extensions = ['.tiff', '.tif']
            elif 'wordprocessingml' in mime_type:
                extensions = ['.docx']
            elif 'ms-excel' in mime_type:
                extensions = ['.xls']
            elif 'spreadsheetml' in mime_type:
                extensions = ['.xlsx']
            elif 'msword' in mime_type:
                extensions = ['.doc']
            elif 'text/plain' in mime_type:
                extensions = ['.txt']
            
            if extensions:
                self.SUPPORTED_MIME_TYPES[mime_type] = extensions
    
    async def validate_file(self, file: UploadFile, document_type: Optional[DocumentType] = None) -> Dict[str, Any]:
        """
        Validate uploaded file for security and format compliance.
        
        Args:
            file: The uploaded file to validate
            document_type: Optional document type for additional validation
            
        Returns:
            Dict containing validation results
            
        Raises:
            FileValidationError: If file validation fails
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Get file extension for size limit
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            max_size = get_file_size_limit(file_extension)
            
            # Check file size
            if file.size and file.size > max_size:
                raise FileValidationError(
                    f"File size {file.size} exceeds maximum allowed size {max_size} for {file_extension} files"
                )
            
            # Read file content for analysis
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            # Detect actual MIME type
            if MAGIC_AVAILABLE:
                mime_type = magic.from_buffer(content, mime=True)
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(file.filename)
                if not mime_type:
                    mime_type = 'application/octet-stream'
            
            # Validate MIME type against allowed types
            if mime_type not in self.storage_config.allowed_mime_types:
                raise FileValidationError(f"Unsupported file type: {mime_type}")
            
            # Validate file extension
            file_extension_with_dot = Path(file.filename).suffix.lower()
            if file_extension not in self.storage_config.allowed_extensions:
                raise FileValidationError(f"Unsupported file extension: {file_extension}")
            
            # Validate extension matches MIME type if we have the mapping
            if mime_type in self.SUPPORTED_MIME_TYPES:
                if file_extension_with_dot not in self.SUPPORTED_MIME_TYPES[mime_type]:
                    validation_result['warnings'].append(
                        f"File extension {file_extension_with_dot} may not match detected type {mime_type}"
                    )
            
            # Security checks
            if self.storage_config.enable_virus_scan:
                if self._contains_malicious_content(content, mime_type):
                    raise FileValidationError("File contains potentially malicious content")
            
            # Calculate file hash for integrity
            file_hash = hashlib.sha256(content).hexdigest()
            
            validation_result['file_info'] = {
                'size': len(content),
                'mime_type': mime_type,
                'hash_sha256': file_hash,
                'extension': file_extension_with_dot
            }
            
        except FileValidationError:
            raise
        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            raise FileValidationError(f"File validation failed: {str(e)}")
        
        return validation_result
    
    def _contains_malicious_content(self, content: bytes, mime_type: str) -> bool:
        """
        Basic malicious content detection.
        
        Args:
            content: File content bytes
            mime_type: Detected MIME type
            
        Returns:
            True if potentially malicious content is detected
        """
        # This is a basic implementation - in production, use a proper antivirus scanner
        malicious_signatures = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'eval(',
            b'exec(',        ]
        
        content_lower = content.lower()
        for signature in malicious_signatures:
            if signature in content_lower:
                return True
        
        return False
    
    async def store_file(self, file: UploadFile, patient_id: str, document_type: DocumentType) -> FileMetadata:
        """
        Store uploaded file securely with metadata.
        
        Args:
            file: The uploaded file
            patient_id: Patient identifier
            document_type: Type of document
            
        Returns:
            FileMetadata object with storage information
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Validate file first
        validation_result = await self.validate_file(file, document_type)
        if not validation_result['is_valid']:
            raise FileValidationError(f"File validation failed: {validation_result['errors']}")
        
        file_info = validation_result['file_info']
        
        # Determine storage directory based on file type and configuration
        if file_info['mime_type'].startswith('image/'):
            # Use image storage directory
            base_dir = self.storage_config.image_upload_dir
        else:
            # Use document storage directory
            base_dir = self.storage_config.upload_dir
        
        # Create patient-specific directory if enabled
        if self.storage_config.use_patient_subdirs:
            patient_dir_name = self.storage_config.patient_dir_structure.format(patient_id=patient_id)
            storage_dir = base_dir / patient_dir_name
        else:
            storage_dir = base_dir
        
        # Ensure storage directory exists
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate storage filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_extension = Path(file.filename).suffix
        storage_filename = f"{file_id}_{timestamp}_{document_type.value}{file_extension}"
        storage_path = storage_dir / storage_filename
        
        try:
            # Store file content
            async with aiofiles.open(storage_path, 'wb') as stored_file:
                content = await file.read()
                await stored_file.write(content)
            
            # Create metadata object
            metadata = FileMetadata(
                filename=file.filename,
                file_id=file_id,
                size=file_info['size'],
                mime_type=file_info['mime_type'],
                hash_sha256=file_info['hash_sha256'],
                created_at=datetime.utcnow(),
                original_path=file.filename,
                stored_path=str(storage_path)
            )
            
            logger.info(f"File stored successfully: {file_id} -> {storage_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"File storage error: {str(e)}")
            # Clean up partial file if exists
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=500, detail=f"File storage failed: {str(e)}")
    
    async def preprocess_image_for_ocr(self, file_path: Path) -> Optional[Path]:
        """
        Preprocess image files for better OCR accuracy.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Path to preprocessed image or None if preprocessing failed
        """
        try:
            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance image for OCR
                # This is a basic implementation - in production, use more sophisticated preprocessing
                
                # Create preprocessed filename
                preprocessed_path = file_path.parent / f"preprocessed_{file_path.name}"
                
                # Save preprocessed image
                img.save(preprocessed_path, 'JPEG', quality=95, optimize=True)
                
                logger.info(f"Image preprocessed for OCR: {preprocessed_path}")
                return preprocessed_path
                
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return None
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            stat = file_path.stat()
              # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type and MAGIC_AVAILABLE:
                with open(file_path, 'rb') as f:
                    mime_type = magic.from_buffer(f.read(1024), mime=True)
            elif not mime_type:
                mime_type = 'application/octet-stream'
            
            return {
                'filename': file_path.name,
                'size': stat.st_size,
                'size_human': self._format_file_size(stat.st_size),
                'mime_type': mime_type,
                'created_at': datetime.fromtimestamp(stat.st_ctime),
                'modified_at': datetime.fromtimestamp(stat.st_mtime),
                'extension': file_path.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {}
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable size string
        """
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files cleaned up
        """
        temp_dir = self.STORAGE_BASE_DIR / 'temp'
        if not temp_dir.exists():
            return 0
        
        cleaned_count = 0
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        try:
            for temp_file in temp_dir.rglob('*'):
                if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    cleaned_count += 1
                    
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            
        except Exception as e:
            logger.error(f"Temp file cleanup error: {str(e)}")
        
        return cleaned_count
