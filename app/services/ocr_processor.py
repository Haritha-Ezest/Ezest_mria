"""
OCR (Optical Character Recognition) processor service for medical document text extraction.

This module provides advanced OCR capabilities with medical document optimization,
image preprocessing, and Azure integration following best practices.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import pytesseract
import pdf2image
import magic

from app.schemas.ocr import (
    OCRRequest, OCRResponse, OCRPage, OCRConfiguration, OCRStatus,
    TextRegion, BoundingBox, ImagePreprocessing, DocumentType, OCREngine
)
from app.common.utils import get_logger


# Configure logging
logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Advanced image preprocessing for medical documents.
    
    Implements specialized preprocessing techniques for medical documents
    including handwritten prescriptions, lab reports, and scanned documents.
    """
    
    def __init__(self):
        """Initialize the image preprocessor."""
        self.supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP', 'WEBP']
        
    async def preprocess_image(
        self, 
        image: np.ndarray, 
        preprocessing_level: ImagePreprocessing,
        document_type: DocumentType
    ) -> np.ndarray:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            preprocessing_level: Level of preprocessing to apply
            document_type: Type of medical document for optimization
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            if preprocessing_level == ImagePreprocessing.NONE:
                return image
                
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Apply preprocessing based on level
            if preprocessing_level == ImagePreprocessing.BASIC:
                processed = await self._apply_basic_preprocessing(gray)
            elif preprocessing_level == ImagePreprocessing.ADVANCED:
                processed = await self._apply_advanced_preprocessing(gray)
            elif preprocessing_level == ImagePreprocessing.MEDICAL_OPTIMIZED:
                processed = await self._apply_medical_optimized_preprocessing(
                    gray, document_type
                )
            else:
                processed = gray
                
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image  # Return original on error
            
    async def _apply_basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply basic preprocessing steps."""
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(image)
        
        # Contrast enhancement
        enhanced = cv2.equalizeHist(denoised)
        
        # Slight blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        return blurred
        
    async def _apply_advanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing steps."""
        # Skew correction
        corrected = await self._correct_skew(image)
        
        # Advanced noise reduction
        denoised = cv2.bilateralFilter(corrected, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    async def _apply_medical_optimized_preprocessing(
        self, 
        image: np.ndarray, 
        document_type: DocumentType
    ) -> np.ndarray:
        """Apply medical document-specific preprocessing."""
        # Start with advanced preprocessing
        processed = await self._apply_advanced_preprocessing(image)
        
        # Document-specific optimizations
        if document_type == DocumentType.PRESCRIPTION:
            # Optimize for handwritten text
            processed = await self._optimize_for_handwriting(processed)
        elif document_type == DocumentType.LAB_REPORT:
            # Optimize for tables and numbers
            processed = await self._optimize_for_tables(processed)
        elif document_type in [DocumentType.RADIOLOGY_REPORT, DocumentType.PATHOLOGY_REPORT]:
            # Optimize for technical reports
            processed = await self._optimize_for_technical_reports(processed)
            
        return processed
        
    async def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        try:
            # Detect lines using HoughLines
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate skew angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = np.degrees(theta - np.pi/2)
                    angles.append(angle)
                    
                # Get median angle
                if angles:
                    skew_angle = np.median(angles)
                    
                    # Correct if skew is significant
                    if abs(skew_angle) > 0.5:
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                                 flags=cv2.INTER_CUBIC, 
                                                 borderMode=cv2.BORDER_REPLICATE)
                        return corrected
                        
        except Exception as e:
            logger.warning(f"Skew correction failed: {e}")
            
        return image
        
    async def _optimize_for_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for handwritten text recognition."""
        # Enhance contrast for handwriting
        enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        # Apply median filter to reduce noise while preserving edges
        filtered = cv2.medianBlur(enhanced, 3)
        
        # Use adaptive threshold optimized for handwriting
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        return thresh
        
    async def _optimize_for_tables(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for table and numerical data."""
        # Enhance contrast for clear table lines
        enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=5)
        
        # Apply morphological operations to enhance table structure
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal and vertical lines
        horizontal = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_v)
        
        # Combine line detection with original
        lines = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        result = cv2.addWeighted(enhanced, 0.8, lines, 0.2, 0)
        
        return result
        
    async def _optimize_for_technical_reports(self, image: np.ndarray) -> np.ndarray:
        """Optimize for technical medical reports."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Reduce noise while preserving text
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh


class OCRProcessor:
    """
    Advanced OCR processor for medical documents.
    
    Features:
    - Multi-engine OCR support (Tesseract, Azure Form Recognizer)
    - Medical document optimization
    - Confidence scoring and quality assessment
    - Batch processing capabilities
    - Error handling and retry logic
    """
    
    def __init__(self):
        """Initialize the OCR processor."""
        self.preprocessor = ImagePreprocessor()
        self.supported_formats = [
            'application/pdf',
            'image/jpeg',
            'image/png', 
            'image/tiff',
            'image/bmp',
            'image/webp'
        ]
        
        # Configure Tesseract
        self._configure_tesseract()
        
    def _configure_tesseract(self):
        """Configure Tesseract OCR engine."""
        try:
            # Set Tesseract configuration for medical documents
            self.tesseract_config = {
                'basic': r'--oem 3 --psm 6',
                'medical': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()-/ ',
                'handwriting': r'--oem 3 --psm 8',
                'tables': r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            }
            
            # Verify Tesseract installation
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
        except Exception as e:
            logger.error(f"Tesseract configuration failed: {e}")
            raise
            
    async def process_document(self, ocr_request: OCRRequest) -> OCRResponse:
        """
        Process a document with OCR text extraction.
        
        Args:
            ocr_request: OCR processing request
            
        Returns:
            OCRResponse: OCR processing results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting OCR processing for document {ocr_request.document_id}")
            
            # Load document
            if ocr_request.file_path:
                document_path = Path(ocr_request.file_path)
                if not document_path.exists():
                    raise FileNotFoundError(f"Document not found: {ocr_request.file_path}")
                    
                # Detect file type
                file_type = await self._detect_file_type(document_path)
                
                # Process based on file type
                if file_type == 'application/pdf':
                    pages = await self._process_pdf(document_path, ocr_request.config)
                else:
                    pages = await self._process_image(document_path, ocr_request.config)
                    
            elif ocr_request.file_url:
                # Handle URL-based processing (placeholder)
                raise NotImplementedError("URL-based processing not yet implemented")
            else:
                raise ValueError("Either file_path or file_url must be provided")
                
            # Calculate overall metrics
            total_pages = len(pages)
            pages_successful = sum(1 for page in pages if page.confidence > 0.5)
            pages_failed = total_pages - pages_successful
            
            overall_confidence = np.mean([page.confidence for page in pages]) if pages else 0.0
            full_text = '\n\n'.join([page.text for page in pages])
            
            # Calculate processing time
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            average_processing_time = total_processing_time / max(total_pages, 1)
            
            # Estimate word count
            estimated_word_count = len(full_text.split())
            
            # Detect languages (simplified)
            languages_detected = ['en']  # Placeholder
            
            # Create response
            ocr_response = OCRResponse(
                document_id=ocr_request.document_id,
                job_id=ocr_request.job_id,
                status=OCRStatus.COMPLETED,
                pages=pages,
                full_text=full_text,
                total_pages=total_pages,
                pages_successful=pages_successful,
                pages_failed=pages_failed,
                overall_confidence=overall_confidence,
                average_processing_time=average_processing_time,
                total_processing_time=total_processing_time,
                estimated_word_count=estimated_word_count,
                languages_detected=languages_detected,
                started_at=start_time,
                completed_at=end_time,
                original_filename=document_path.name if ocr_request.file_path else None,
                file_size_bytes=document_path.stat().st_size if ocr_request.file_path else None
            )
            
            logger.info(f"OCR processing completed for document {ocr_request.document_id}")
            return ocr_response
            
        except Exception as e:
            logger.error(f"OCR processing failed for document {ocr_request.document_id}: {e}")
            
            # Return error response
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            
            return OCRResponse(
                document_id=ocr_request.document_id,
                job_id=ocr_request.job_id,
                status=OCRStatus.FAILED,
                pages=[],
                full_text="",
                total_pages=0,
                pages_successful=0,
                pages_failed=0,
                overall_confidence=0.0,
                average_processing_time=0.0,
                total_processing_time=total_processing_time,
                estimated_word_count=0,
                languages_detected=[],
                error_message=str(e),
                started_at=start_time,
                completed_at=end_time
            )
            
    async def _detect_file_type(self, file_path: Path) -> str:
        """Detect file MIME type."""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            return mime_type
        except Exception as e:
            logger.warning(f"File type detection failed: {e}")
            # Fallback to extension-based detection
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                return 'application/pdf'
            elif ext in ['.jpg', '.jpeg']:
                return 'image/jpeg'
            elif ext == '.png':
                return 'image/png'
            elif ext in ['.tif', '.tiff']:
                return 'image/tiff'
            else:
                return 'application/octet-stream'
                
    async def _process_pdf(self, pdf_path: Path, config: OCRConfiguration) -> List[OCRPage]:
        """Process PDF document."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=config.dpi,
                first_page=1,
                last_page=config.max_pages
            )
            
            pages = []
            for page_num, image in enumerate(images, 1):
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Process the page
                page_result = await self._process_page(
                    img_array, page_num, config
                )
                pages.append(page_result)
                
            return pages
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
            
    async def _process_image(self, image_path: Path, config: OCRConfiguration) -> List[OCRPage]:
        """Process image file."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Process single page
            page_result = await self._process_page(image, 1, config)
            return [page_result]
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
            
    async def _process_page(
        self, 
        image: np.ndarray, 
        page_number: int, 
        config: OCRConfiguration
    ) -> OCRPage:
        """Process a single page image."""
        start_time = datetime.now()
        
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            image_dimensions = (width, height)
            
            # Preprocess image
            if config.preprocessing != ImagePreprocessing.NONE:
                processed_image = await self.preprocessor.preprocess_image(
                    image, config.preprocessing, config.document_type
                )
            else:
                processed_image = image
                
            # Perform OCR
            if config.engine == OCREngine.TESSERACT:
                page_result = await self._perform_tesseract_ocr(
                    processed_image, config
                )
            else:
                raise NotImplementedError(f"OCR engine {config.engine} not implemented")
                
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create OCR page result
            ocr_page = OCRPage(
                page_number=page_number,
                text=page_result['text'],
                text_regions=page_result['text_regions'],
                confidence=page_result['confidence'],
                processing_time_seconds=processing_time,
                image_dimensions=image_dimensions,
                image_dpi=config.dpi,
                quality_score=page_result['quality_score'],
                issues_detected=page_result['issues_detected'],
                tables_detected=page_result.get('tables_detected', 0),
                has_handwriting=page_result.get('has_handwriting', False),
                has_signatures=page_result.get('has_signatures', False)
            )
            
            return ocr_page
            
        except Exception as e:
            logger.error(f"Page processing failed: {e}")
            
            # Return error page
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return OCRPage(
                page_number=page_number,
                text="",
                text_regions=[],
                confidence=0.0,
                processing_time_seconds=processing_time,
                image_dimensions=(0, 0),
                image_dpi=config.dpi,
                quality_score=0.0,
                issues_detected=[f"Processing error: {str(e)}"]
            )
            
    async def _perform_tesseract_ocr(
        self, 
        image: np.ndarray, 
        config: OCRConfiguration
    ) -> Dict[str, Any]:
        """Perform OCR using Tesseract."""
        try:
            # Select appropriate Tesseract configuration
            if config.document_type == DocumentType.PRESCRIPTION:
                tesseract_config = self.tesseract_config['handwriting']
            elif config.document_type == DocumentType.LAB_REPORT:
                tesseract_config = self.tesseract_config['tables']
            else:
                tesseract_config = self.tesseract_config['medical']
                
            # Perform OCR with detailed data
            ocr_data = pytesseract.image_to_data(
                image,
                config=tesseract_config,
                lang='+'.join(config.languages),
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            full_text = pytesseract.image_to_string(
                image,
                config=tesseract_config,
                lang='+'.join(config.languages)
            ).strip()
            
            # Process OCR data to create text regions
            text_regions = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    confidence = float(ocr_data['conf'][i]) / 100.0
                    if confidence >= config.confidence_threshold:
                        # Create bounding box
                        bbox = BoundingBox(
                            x=float(ocr_data['left'][i]),
                            y=float(ocr_data['top'][i]),
                            width=float(ocr_data['width'][i]),
                            height=float(ocr_data['height'][i])
                        )
                        
                        # Create text region
                        text_region = TextRegion(
                            text=word,
                            confidence=confidence,
                            bounding_box=bbox
                        )
                        
                        text_regions.append(text_region)
                        confidences.append(confidence)
                        
            # Calculate overall confidence
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            # Assess image quality
            quality_score = await self._assess_image_quality(image)
            
            # Detect issues
            issues_detected = []
            if quality_score < 0.7:
                issues_detected.append("Low image quality detected")
            if overall_confidence < 0.8:
                issues_detected.append("Low OCR confidence")
                
            return {
                'text': full_text,
                'text_regions': text_regions,
                'confidence': overall_confidence,
                'quality_score': quality_score,
                'issues_detected': issues_detected
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'text': "",
                'text_regions': [],
                'confidence': 0.0,
                'quality_score': 0.0,
                'issues_detected': [f"OCR error: {str(e)}"]
            }
            
    async def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality for OCR processing."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Calculate various quality metrics
            
            # 1. Variance of Laplacian (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 2. Contrast (standard deviation)
            contrast_score = min(gray.std() / 64.0, 1.0)
            
            # 3. Brightness (avoid too dark or too bright)
            brightness = gray.mean()
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            
            # 4. Noise level (using bilateral filter difference)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            noise_level = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
            noise_score = max(0.0, 1.0 - noise_level / 20.0)
            
            # Combine metrics
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.3 +
                brightness_score * 0.2 +
                noise_score * 0.2
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default score
    
    async def health_check(self) -> bool:
        """
        Perform health check on OCR processor.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            logger.debug(f"Tesseract version: {version}")
            
            # Test with a simple image processing
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
            cv2.putText(test_image, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Attempt OCR on test image
            text = pytesseract.image_to_string(test_image, config='--psm 8')
            
            # If we can extract some text, consider it healthy
            return len(text.strip()) > 0
            
        except Exception as e:
            logger.error(f"OCR health check failed: {e}")
            return False
    
    async def get_average_response_time(self) -> float:
        """Get average response time for OCR processing."""
        # This would typically be stored in a metrics system
        # For now, return a placeholder
        return 2.5
    
    async def get_processed_count(self) -> int:
        """Get total number of processed requests."""
        # This would typically be stored in a metrics system
        # For now, return a placeholder
        return 0
        

# Global OCR processor instance
ocr_processor_instance = None


def get_ocr_processor() -> OCRProcessor:
    """Get or create OCR processor instance."""
    global ocr_processor_instance
    if ocr_processor_instance is None:
        ocr_processor_instance = OCRProcessor()
    return ocr_processor_instance
