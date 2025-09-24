"""
Configuration management and validation for Welfare Chatbot
"""
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration with validation"""
    google_api_key: str
    max_retries: int = 3
    timeout: int = 30
    log_level: str = "INFO"
    faiss_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    def validate(self):
        """Validate all configuration values"""
        if not self.google_api_key or self.google_api_key == "your-google-api-key-here":
            raise ValueError(
                "GOOGLE_API_KEY is not set or is using placeholder value. "
                "Please set a valid Google API key in the .env file."
            )

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")

        logging.info("Configuration validation passed")

def get_config() -> Config:
    """Get validated configuration instance"""
    return Config(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        timeout=int(os.getenv("TIMEOUT", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        faiss_path=os.getenv("FAISS_PATH")
    )

def setup_logging(config: Config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('welfare_chatbot.log'),
            logging.StreamHandler()
        ]
    )