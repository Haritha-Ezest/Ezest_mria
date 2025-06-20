"""
Configuration management for the MRIA application.

This module handles loading and validation of environment variables,
providing typed configuration objects for different components of the system.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class StorageConfig:
    """File storage configuration settings."""
    
    # Base directories
    base_storage_dir: Path
    upload_dir: Path
    temp_dir: Path
    processed_dir: Path
    backup_dir: Path
    image_upload_dir: Path
    image_processed_dir: Path
    
    # Patient directory structure
    patient_dir_structure: str
    use_patient_subdirs: bool
    
    # File size limits (in bytes)
    max_file_size: int
    max_pdf_size: int
    max_image_size: int
    max_office_doc_size: int
    
    # File type restrictions
    allowed_extensions: List[str]
    allowed_mime_types: List[str]
    
    # Retention and cleanup
    file_retention_days: int
    temp_file_retention_hours: int
    auto_cleanup_enabled: bool
    cleanup_schedule: str
    
    # Security settings
    enable_virus_scan: bool
    virus_scan_timeout: int
    enable_file_encryption: bool
    encryption_key_path: Optional[Path]
    
    # Storage quotas (in MB)
    max_storage_per_patient: int
    max_total_storage: int
    storage_warning_threshold: int


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    app_name: str = Field("MRIA Document Processing System", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    reload: bool = Field(False, env="RELOAD")
    
    # Security Configuration
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # File Storage Configuration
    base_storage_dir: str = Field("./storage", env="BASE_STORAGE_DIR")
    upload_dir: str = Field("uploads/documents", env="UPLOAD_DIR")
    temp_dir: str = Field("uploads/temp", env="TEMP_DIR")
    processed_dir: str = Field("uploads/processed", env="PROCESSED_DIR")
    backup_dir: str = Field("uploads/backup", env="BACKUP_DIR")
    image_upload_dir: str = Field("uploads/images", env="IMAGE_UPLOAD_DIR")
    image_processed_dir: str = Field("uploads/images/processed", env="IMAGE_PROCESSED_DIR")
    
    # Patient directory configuration
    patient_dir_structure: str = Field("patient_{patient_id}", env="PATIENT_DIR_STRUCTURE")
    use_patient_subdirs: bool = Field(True, env="USE_PATIENT_SUBDIRS")
    
    # File size limits
    max_file_size: int = Field(52428800, env="MAX_FILE_SIZE")  # 50MB
    max_pdf_size: int = Field(52428800, env="MAX_PDF_SIZE")  # 50MB
    max_image_size: int = Field(10485760, env="MAX_IMAGE_SIZE")  # 10MB
    max_office_doc_size: int = Field(26214400, env="MAX_OFFICE_DOC_SIZE")  # 25MB
    
    # File type restrictions
    allowed_file_extensions: str = Field(
        "pdf,txt,docx,doc,png,jpg,jpeg,tiff,tif,xls,xlsx", 
        env="ALLOWED_FILE_EXTENSIONS"
    )
    allowed_mime_types: str = Field(
        "application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,image/png,image/jpeg,image/tiff,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        env="ALLOWED_MIME_TYPES"
    )
    
    # Retention and cleanup
    file_retention_days: int = Field(30, env="FILE_RETENTION_DAYS")
    temp_file_retention_hours: int = Field(24, env="TEMP_FILE_RETENTION_HOURS")
    auto_cleanup_enabled: bool = Field(True, env="AUTO_CLEANUP_ENABLED")
    cleanup_schedule: str = Field("0 2 * * *", env="CLEANUP_SCHEDULE")
    
    # Security settings
    enable_virus_scan: bool = Field(False, env="ENABLE_VIRUS_SCAN")
    virus_scan_timeout: int = Field(30, env="VIRUS_SCAN_TIMEOUT")
    enable_file_encryption: bool = Field(False, env="ENABLE_FILE_ENCRYPTION")
    encryption_key_path: Optional[str] = Field(None, env="ENCRYPTION_KEY_PATH")
    
    # Storage quotas (in MB)
    max_storage_per_patient: int = Field(1024, env="MAX_STORAGE_PER_PATIENT")
    max_total_storage: int = Field(10240, env="MAX_TOTAL_STORAGE")
    storage_warning_threshold: int = Field(80, env="STORAGE_WARNING_THRESHOLD")
    
    # Database Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")
    neo4j_max_connection_lifetime: int = Field(3600, env="NEO4J_MAX_CONNECTION_LIFETIME")
    neo4j_max_connection_pool_size: int = Field(50, env="NEO4J_MAX_CONNECTION_POOL_SIZE")
    neo4j_connection_acquisition_timeout: int = Field(60, env="NEO4J_CONNECTION_ACQUISITION_TIMEOUT")
      # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field("./storage/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field("medical_documents", env="CHROMA_COLLECTION_NAME")
    chroma_embedding_model: str = Field("all-MiniLM-L6-v2", env="CHROMA_EMBEDDING_MODEL")
    
    # OCR Configuration
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    ocr_language: str = Field("eng", env="OCR_LANGUAGE")
    ocr_dpi: int = Field(300, env="OCR_DPI")
    ocr_confidence_threshold: int = Field(60, env="OCR_CONFIDENCE_THRESHOLD")
    
    @validator("allowed_file_extensions")
    def parse_file_extensions(cls, v):
        """Parse comma-separated file extensions."""
        return [ext.strip().lower() for ext in v.split(",")]
    
    @validator("allowed_mime_types")
    def parse_mime_types(cls, v):
        """Parse comma-separated MIME types."""
        return [mime.strip().lower() for mime in v.split(",")]
    
    @validator("base_storage_dir", "encryption_key_path")
    def validate_paths(cls, v):
        """Validate and convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v)
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration as a structured object."""
        base_dir = Path(self.base_storage_dir)
        
        return StorageConfig(
            # Base directories
            base_storage_dir=base_dir,
            upload_dir=base_dir / self.upload_dir,
            temp_dir=base_dir / self.temp_dir,
            processed_dir=base_dir / self.processed_dir,
            backup_dir=base_dir / self.backup_dir,
            image_upload_dir=base_dir / self.image_upload_dir,
            image_processed_dir=base_dir / self.image_processed_dir,
            
            # Patient directory structure
            patient_dir_structure=self.patient_dir_structure,
            use_patient_subdirs=self.use_patient_subdirs,
            
            # File size limits
            max_file_size=self.max_file_size,
            max_pdf_size=self.max_pdf_size,
            max_image_size=self.max_image_size,
            max_office_doc_size=self.max_office_doc_size,
            
            # File type restrictions
            allowed_extensions=self.allowed_file_extensions,
            allowed_mime_types=self.allowed_mime_types,
            
            # Retention and cleanup
            file_retention_days=self.file_retention_days,
            temp_file_retention_hours=self.temp_file_retention_hours,
            auto_cleanup_enabled=self.auto_cleanup_enabled,
            cleanup_schedule=self.cleanup_schedule,
            
            # Security settings
            enable_virus_scan=self.enable_virus_scan,
            virus_scan_timeout=self.virus_scan_timeout,
            enable_file_encryption=self.enable_file_encryption,
            encryption_key_path=Path(self.encryption_key_path) if self.encryption_key_path else None,
            
            # Storage quotas
            max_storage_per_patient=self.max_storage_per_patient,
            max_total_storage=self.max_total_storage,
            storage_warning_threshold=self.storage_warning_threshold
        )
    
    def ensure_storage_directories(self):
        """Create storage directories if they don't exist."""
        storage_config = self.get_storage_config()
        
        directories = [
            storage_config.base_storage_dir,
            storage_config.upload_dir,
            storage_config.temp_dir,
            storage_config.processed_dir,
            storage_config.backup_dir,
            storage_config.image_upload_dir,
            storage_config.image_processed_dir        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env file


# Global settings instance
settings = Settings()

# Ensure storage directories exist on import
settings.ensure_storage_directories()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    return settings.get_storage_config()


def get_patient_upload_dir(patient_id: str) -> Path:
    """Get the upload directory for a specific patient."""
    storage_config = get_storage_config()
    
    if storage_config.use_patient_subdirs:
        patient_dir_name = storage_config.patient_dir_structure.format(patient_id=patient_id)
        return storage_config.upload_dir / patient_dir_name
    else:
        return storage_config.upload_dir


def get_patient_image_dir(patient_id: str) -> Path:
    """Get the image directory for a specific patient."""
    storage_config = get_storage_config()
    
    if storage_config.use_patient_subdirs:
        patient_dir_name = storage_config.patient_dir_structure.format(patient_id=patient_id)
        return storage_config.image_upload_dir / patient_dir_name
    else:
        return storage_config.image_upload_dir


def get_temp_dir() -> Path:
    """Get the temporary files directory."""
    return get_storage_config().temp_dir


def get_processed_dir() -> Path:
    """Get the processed files directory."""
    return get_storage_config().processed_dir


def get_file_size_limit(file_extension: str) -> int:
    """Get file size limit based on file extension."""
    storage_config = get_storage_config()
    
    # Map file extensions to size limits
    size_limits = {
        'pdf': storage_config.max_pdf_size,
        'png': storage_config.max_image_size,
        'jpg': storage_config.max_image_size,
        'jpeg': storage_config.max_image_size,
        'tiff': storage_config.max_image_size,
        'tif': storage_config.max_image_size,
        'docx': storage_config.max_office_doc_size,
        'doc': storage_config.max_office_doc_size,
        'xls': storage_config.max_office_doc_size,
        'xlsx': storage_config.max_office_doc_size,
    }
    
    return size_limits.get(file_extension.lower(), storage_config.max_file_size)
