"""
Health check utilities for Welfare Chatbot
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any

def check_system_health() -> Dict[str, Any]:
    """Perform comprehensive system health check"""
    health_status = {
        "status": "healthy",
        "checks": {},
        "errors": []
    }

    try:
        # Check environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-google-api-key-here":
            health_status["checks"]["api_key"] = "❌ Not configured"
            health_status["errors"].append("Google API key not set")
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["api_key"] = "✅ Configured"

        # Check FAISS database files
        base_dir = Path(__file__).parent.parent
        faiss_path = base_dir / "db" / "faiss_index"
        if faiss_path.exists():
            health_status["checks"]["faiss_db"] = "✅ Available"
        else:
            health_status["checks"]["faiss_db"] = "❌ Missing"
            health_status["errors"].append("FAISS index files not found")
            health_status["status"] = "unhealthy"

        # Check data files
        data_file = base_dir / "data" / "vd_base_v2_refined.json"
        if data_file.exists():
            health_status["checks"]["data_file"] = "✅ Available"
        else:
            health_status["checks"]["data_file"] = "❌ Missing"
            health_status["errors"].append("Main data file not found")
            health_status["status"] = "unhealthy"

        # Check required directories
        required_dirs = ["app", "db", "data"]
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            if dir_path.exists():
                health_status["checks"][f"{dir_name}_dir"] = "✅ Available"
            else:
                health_status["checks"][f"{dir_name}_dir"] = "❌ Missing"
                health_status["errors"].append(f"Required directory '{dir_name}' not found")
                health_status["status"] = "unhealthy"

        logging.info(f"Health check completed: {health_status['status']}")

    except Exception as e:
        logging.error(f"Health check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["errors"].append(f"Health check exception: {str(e)}")

    return health_status

def log_health_status(health_status: Dict[str, Any]):
    """Log health check results"""
    status = health_status["status"]
    if status == "healthy":
        logging.info("🟢 System health check: ALL SYSTEMS OPERATIONAL")
    else:
        logging.warning("🟡 System health check: ISSUES DETECTED")
        for error in health_status["errors"]:
            logging.warning(f"  ⚠️  {error}")

    for check_name, result in health_status["checks"].items():
        logging.info(f"  {check_name}: {result}")