#!/usr/bin/env python3
"""
[DEMO] Environment and Setup Verification Script
Use this to verify that the development environment is properly configured.
"""

import os
import sys
from pathlib import Path

print("\n" + "="*80)
print("VideoMultiAgents Development Environment Check")
print("="*80 + "\n")

# Check directory structure
print("✓ Directory Structure:")
required_dirs = [
    "mock_apis",
    "demo_data/videos",
    "demo_data/captions",
    "demo_data/qa",
    "demo_data/features",
    "tools",
    "utils"
]

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    status = "✓" if dir_path.exists() else "✗"
    print(f"  {status} {dir_name}")

# Check configuration files
print("\n✓ Configuration Files:")
config_files = {
    ".env.demo": "Demo environment configuration",
    "Dockerfile.demo": "Demo Docker configuration",
    "docker-compose.demo.yml": "Demo Docker Compose configuration",
    "mock_apis/__init__.py": "Mock APIs module",
    "mock_apis/mock_openai.py": "Mock OpenAI API",
    "mock_apis/mock_gemini.py": "Mock Gemini API",
    "mock_apis/mock_vision.py": "Mock Vision Extractor",
}

for file_name, description in config_files.items():
    file_path = Path(file_name)
    status = "✓" if file_path.exists() else "✗"
    print(f"  {status} {file_name:<40} - {description}")

# Check demo data
print("\n✓ Demo Data Files:")
demo_files = {
    "demo_data/qa/demo_qa.json": "Demo Q&A data",
    "demo_data/captions/demo_captions.json": "Demo captions",
    "demo_data/features/demo_features.json": "Demo features",
}

for file_name, description in demo_files.items():
    file_path = Path(file_name)
    status = "✓" if file_path.exists() else "✗"
    print(f"  {status} {file_name:<50} - {description}")

# Check Python environment
print("\n✓ Python Environment:")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Python executable: {sys.executable}")

# Check installed packages
print("\n✓ Required Packages:")
required_packages = [
    "openai",
    "google.genai",
    "langchain",
    "langgraph",
    "cv2",
    "pandas",
    "PIL",
    "numpy",
    "dotenv"
]

for package in required_packages:
    try:
        if package == "cv2":
            import cv2
            print(f"  ✓ opencv-python (cv2)")
        elif package == "PIL":
            from PIL import Image
            print(f"  ✓ pillow (PIL)")
        elif package == "dotenv":
            from dotenv import load_dotenv
            print(f"  ✓ python-dotenv")
        elif package == "google.genai":
            import google.genai
            print(f"  ✓ google-genai")
        else:
            __import__(package)
            print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (not installed)")

# Check environment variables from .env.demo
print("\n✓ Environment Configuration (.env.demo):")
env_vars = [
    ("USE_MOCK_API", "Demo mode flag"),
    ("USE_DEMO_DATA", "Demo data flag"),
    ("DATASET", "Dataset selection"),
]

if Path(".env.demo").exists():
    from dotenv import load_dotenv
    load_dotenv(".env.demo")
    
    for var_name, description in env_vars:
        value = os.getenv(var_name)
        status = "✓" if value else "✗"
        print(f"  {status} {var_name:<20} = {value or 'not set':<10} ({description})")
else:
    print("  ✗ .env.demo file not found")

print("\n" + "="*80)
print("Setup Verification Complete!")
print("="*80)
print("\nNext Steps:")
print("1. Check that all items above are marked with ✓")
print("2. For production use:")
print("   - Set USE_MOCK_API=false in .env")
print("   - Provide real API keys: OPENAI_API_KEY, GEMINI_API_KEY")
print("   - Download real datasets and update paths")
print("   - Run: python main.py --dataset=<dataset> --modality=all --agents=multi_report")
print("\n")
