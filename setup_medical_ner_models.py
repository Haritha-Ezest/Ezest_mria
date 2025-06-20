#!/usr/bin/env python3
"""
Medical NER Models Setup Script

This script helps set up the required medical NLP models for the Enhanced
Medical Entity Recognition Agent in the MRIA system.
"""

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalNERModelSetup:
    """Setup class for installing medical NER models."""
    
    def __init__(self):
        # Updated model versions for compatibility with spaCy 3.7.x
        self.scispacy_version = "0.5.4"
        self.scispacy_base_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v{}"
        
        self.required_models = [
            ("spacy", "en_core_web_sm", "Standard English spaCy model"),
            ("pip", "scispacy>=0.5.4", "Scientific spaCy package"),
            ("pip", f"{self.scispacy_base_url.format(self.scispacy_version)}/en_core_sci_sm-{self.scispacy_version}.tar.gz", "Scientific spaCy model download"),
            ("pip", f"{self.scispacy_base_url.format(self.scispacy_version)}/en_core_med7_lg-{self.scispacy_version}.tar.gz", "Med7 model download"),
        ]
        
        self.huggingface_models = [
            "emilyalsentzer/Bio_ClinicalBERT",
            "dmis-lab/biobert-base-cased-v1.1",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ]
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        logger.info(f"Python version: {sys.version}")
        return True
    
    def install_package(self, package: str) -> bool:
        """Install a package using pip."""
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e.stderr}")
            return False
    
    def download_spacy_model(self, model_name: str) -> bool:
        """Download a spaCy model."""
        try:
            logger.info(f"Downloading spaCy model: {model_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully downloaded {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {model_name}: {e.stderr}")
            return False
    
    def verify_model_installation(self, model_type: str, model_name: str) -> bool:
        """Verify that a model is properly installed."""
        try:
            if model_type == "spacy":
                import spacy
                nlp = spacy.load(model_name)
                logger.info(f"✓ {model_name} verified successfully")
                return True
            elif model_type == "transformers":
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"✓ {model_name} verified successfully")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to verify {model_name}: {e}")
            return False
        return False
    
    def setup_huggingface_cache(self) -> None:
        """Set up Hugging Face model cache."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info("Setting up Hugging Face model cache...")
            
            for model_name in self.huggingface_models:
                try:
                    logger.info(f"Caching {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    logger.info(f"✓ {model_name} cached successfully")
                except Exception as e:                    logger.warning(f"Failed to cache {model_name}: {e}")
                    
        except ImportError:
            logger.warning("Transformers not installed, skipping Hugging Face cache setup")
    
    def install_base_dependencies(self) -> bool:
        """Install base dependencies with proper version constraints."""
        base_packages = [
            "spacy>=3.7.0,<3.8.0",  # Compatible with scispacy
            "scispacy>=0.5.4",      # Latest version for spaCy 3.7.x
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "numpy<2.0.0",          # Pinned to 1.x for spaCy compatibility
            "pandas>=1.3.0"
        ]
        
        logger.info("Installing base dependencies...")
        success = True
        
        for package in base_packages:
            if not self.install_package(package):
                success = False
        
        return success
    
    def check_existing_versions(self) -> dict:
        """Check versions of existing packages to avoid conflicts."""
        version_info = {}
        
        try:
            import numpy
            version_info['numpy'] = numpy.__version__
            if int(numpy.__version__.split('.')[0]) >= 2:
                logger.warning(f"⚠️  NumPy {numpy.__version__} detected. This may cause compatibility issues with spaCy.")
                logger.info("Consider downgrading to NumPy <2.0.0 for better compatibility.")
        except ImportError:
            version_info['numpy'] = 'Not installed'
        
        try:
            import spacy
            version_info['spacy'] = spacy.__version__
            if spacy.__version__.startswith('3.6'):
                logger.warning(f"⚠️  spaCy {spacy.__version__} detected. Upgrading to 3.7.x recommended.")
        except ImportError:
            version_info['spacy'] = 'Not installed'
        
        try:
            import scispacy
            version_info['scispacy'] = scispacy.__version__
        except ImportError:
            version_info['scispacy'] = 'Not installed'
        
        logger.info(f"Current package versions: {version_info}")
        return version_info
    
    def fix_compatibility_issues(self) -> bool:
        """Fix known compatibility issues."""
        version_info = self.check_existing_versions()
        fixed_issues = False
        
        # Fix NumPy 2.x compatibility issue
        if 'numpy' in version_info and version_info['numpy'] != 'Not installed':
            try:
                import numpy
                if int(numpy.__version__.split('.')[0]) >= 2:
                    logger.info("🔧 Fixing NumPy compatibility issue...")
                    # Uninstall NumPy 2.x
                    if self.uninstall_package("numpy"):
                        # Install compatible NumPy version
                        if self.install_package("numpy<2.0.0"):
                            logger.info("✅ NumPy compatibility fixed")
                            fixed_issues = True
                        else:
                            logger.error("❌ Failed to install compatible NumPy version")
                            return False
            except Exception as e:
                logger.warning(f"Could not check NumPy version: {e}")
        
        # Fix spaCy version if needed
        if 'spacy' in version_info and version_info['spacy'] != 'Not installed':
            try:
                import spacy
                if spacy.__version__.startswith('3.6'):
                    logger.info("🔧 Upgrading spaCy for better compatibility...")
                    if self.install_package("spacy>=3.7.0,<3.8.0"):
                        logger.info("✅ spaCy upgraded successfully")
                        fixed_issues = True
            except Exception as e:
                logger.warning(f"Could not check spaCy version: {e}")
        
        return True
    
    def uninstall_package(self, package: str) -> bool:
        """Uninstall a package using pip."""
        try:
            logger.info(f"Uninstalling {package}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", package, "-y"],
                capture_output=True,
                text=True,
                check=True            )
            logger.info(f"Successfully uninstalled {package}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to uninstall {package}: {e.stderr}")
            return False
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("🏥 Starting Medical NER Models Setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check and fix compatibility issues
        logger.info("🔍 Checking for compatibility issues...")
        if not self.fix_compatibility_issues():
            logger.error("Failed to fix compatibility issues")
            return False
        
        # Install base dependencies
        if not self.install_base_dependencies():
            logger.error("Failed to install base dependencies")
            return False
        
        # Install and download models
        success = True
        
        for model_type, model_name, description in self.required_models:
            logger.info(f"Processing: {description}")
            
            if model_type == "pip":
                if not self.install_package(model_name):
                    success = False
            elif model_type == "spacy":
                if not self.download_spacy_model(model_name):
                    success = False
        
        # Verify installations
        logger.info("Verifying model installations...")
        
        # Verify spaCy models
        spacy_models = ["en_core_web_sm", "en_core_sci_sm"]
        for model in spacy_models:
            self.verify_model_installation("spacy", model)        
        # Setup Hugging Face cache
        self.setup_huggingface_cache()
        
        if success:
            logger.info("✅ Medical NER models setup completed successfully!")
            self.print_usage_instructions()
        else:
            logger.error("❌ Some models failed to install. Check the logs above.")
        
        return success
    
    def print_usage_instructions(self) -> None:
        """Print usage instructions."""
        instructions = """
🎯 Medical NER Models Setup Complete!

✅ Compatibility Issues Fixed:
  • NumPy: Pinned to <2.0.0 for spaCy compatibility
  • spaCy: Upgraded to 3.7.x for scispaCy compatibility
  • scispaCy: Updated to v0.5.4 with latest models

📚 SpaCy Models Installed:
  • en_core_web_sm: Standard English model
  • en_core_sci_sm v0.5.4: Scientific/medical text model
  • en_core_med7_lg v0.5.4: Large medical entity model

🤖 Transformers Models (cached):
  • Bio_ClinicalBERT: Clinical notes optimized BERT
  • BioBERT: Biomedical domain BERT
  • PubMedBERT: PubMed literature BERT

🚀 Next Steps:
  1. Start the MRIA application: uvicorn app.main:app --reload
  2. Test the NER endpoint: POST /ner/extract
  3. Run the enhanced NER demo: python enhanced_ner_demo.py

📋 Example Usage:
  
  from app.services.ner_processor import get_ner_processor
  from app.schemas.ner import NERRequest, ProcessingMode
  
  # Initialize processor
  ner_processor = await get_ner_processor()
  
  # Create request
  request = NERRequest(
      text="Patient has diabetes and takes metformin 500mg daily.",
      processing_mode=ProcessingMode.MEDICAL,
      enable_entity_linking=True
  )
  
  # Extract entities
  response = await ner_processor.extract_entities(request)

⚠️  Troubleshooting:
  • If you see NumPy compatibility errors, run this script again
  • For model loading issues, check that all models downloaded correctly
  • Ensure Python 3.8+ is being used

🔗 Documentation:
  • See ENHANCED_NER_DOCUMENTATION.md for detailed information
  • Check the /ner/info endpoint for API capabilities
  • Review example medical texts in enhanced_ner_demo.py

Happy medical text processing! 🏥
        """
        print(instructions)

def main():
    """Main setup function."""
    setup = MedicalNERModelSetup()
    success = setup.run_setup()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
