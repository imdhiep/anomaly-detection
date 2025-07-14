#!/usr/bin/env python3
"""
🧭 Navigation Guide for 3D Anomaly Detection Pipeline
====================================================

This guide helps you navigate the organized project structure.
"""

import os
from pathlib import Path

def show_project_structure():
    """Display the organized project structure."""
    
    print("🏗️  ORGANIZED PROJECT STRUCTURE")
    print("=" * 60)
    
    structure = {
        "📦 src/": {
            "description": "All source code organized by function",
            "contents": {
                "🔧 core/": "Core pipeline functionality (pipeline, data loading, etc.)",
                "🧠 models/": "Deep learning models (CNNs, autoencoders, training)",
                "🛠️ utils/": "Utility functions and helpers"
            }
        },
        "📚 examples/": {
            "description": "Usage examples from basic to advanced",
            "contents": {
                "basic/": "Simple demos that work without PyTorch",
                "advanced/": "Full PyTorch implementations"
            }
        },
        "🧪 tests/": {
            "description": "Test suite for validation",
            "contents": {
                "unit/": "Individual component tests",
                "integration/": "Full pipeline tests"
            }
        },
        "📖 docs/": {
            "description": "Complete documentation",
            "contents": {
                "COMPREHENSIVE_README.md": "Detailed usage guide",
                "PYTORCH_SETUP.md": "Installation instructions",
                "api/": "API documentation",
                "guides/": "Tutorial guides"
            }
        },
        "⚙️ config/": {
            "description": "Configuration and requirements",
            "contents": {
                "requirements.txt": "Python dependencies"
            }
        },
        "📊 results/": {
            "description": "Generated outputs and visualizations",
            "contents": {
                "anomaly_detection_results.png": "Sample results"
            }
        }
    }
    
    for folder, info in structure.items():
        print(f"\n{folder}")
        print(f"   {info['description']}")
        for item, desc in info['contents'].items():
            print(f"   ├── {item:<30} # {desc}")

def show_quick_start_options():
    """Show different ways to get started."""
    
    print(f"\n🚀 QUICK START OPTIONS")
    print("=" * 60)
    
    options = [
        {
            "title": "🎨 Visual Demo (No PyTorch needed)",
            "location": "examples/basic/",
            "command": "python demo_pipeline.py",
            "description": "Shows complete pipeline architecture with visualizations"
        },
        {
            "title": "📖 Quick Start Guide",
            "location": "examples/basic/",
            "command": "python quick_start.py", 
            "description": "Interactive guide to using the pipeline"
        },
        {
            "title": "🔬 Full PyTorch Example",
            "location": "examples/advanced/",
            "command": "python example_usage.py",
            "description": "Complete implementation with real deep learning (requires PyTorch)"
        },
        {
            "title": "🧪 Run Tests",
            "location": "tests/integration/",
            "command": "python comprehensive_test.py",
            "description": "Validate all components are working"
        },
        {
            "title": "📚 Read Documentation",
            "location": "docs/",
            "command": "open COMPREHENSIVE_README.md",
            "description": "Complete usage guide and API reference"
        }
    ]
    
    for i, option in enumerate(options, 1):
        print(f"\n{i}. {option['title']}")
        print(f"   📁 Location: {option['location']}")
        print(f"   💻 Command:  {option['command']}")
        print(f"   📝 Purpose:  {option['description']}")

def show_import_examples():
    """Show how to import from the organized structure."""
    
    print(f"\n🔗 IMPORT EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "use_case": "Basic Pipeline Usage",
            "code": """
# Import main pipeline
from detection.src.core import AnomalyDetectionPipeline
from detection.src.core import load_and_normalize_volume

# Create and use pipeline
pipeline = AnomalyDetectionPipeline()
volume = load_and_normalize_volume('scan.nii.gz')
scores = pipeline.score_map(volume)
"""
        },
        {
            "use_case": "Custom Model Building", 
            "code": """
# Import specific models
from detection.src.models import ResNet3D, VariationalAutoencoder
from detection.src.models import train_autoencoder

# Build custom architecture
encoder = ResNet3D(layers=[2, 2, 2, 2])
autoencoder = VariationalAutoencoder(input_dim=512)
"""
        },
        {
            "use_case": "Data Processing",
            "code": """
# Import data utilities
from detection.src.core import extract_patches_3d
from detection.src.core import comprehensive_evaluation

# Process data
patches = extract_patches_3d(volume, patch_size=(8, 8, 8))
metrics = comprehensive_evaluation(predictions, ground_truth)
"""
        }
    ]
    
    for example in examples:
        print(f"\n📌 {example['use_case']}:")
        print("```python" + example['code'] + "```")

def main():
    """Main navigation guide."""
    
    print("🧭 3D ANOMALY DETECTION PIPELINE - NAVIGATION GUIDE")
    print("=" * 80)
    
    show_project_structure()
    show_quick_start_options()
    show_import_examples()
    
    print(f"\n🎯 RECOMMENDED FIRST STEPS")
    print("=" * 60)
    print("1. 🎨 Run the visual demo: cd examples/basic && python demo_pipeline.py")
    print("2. 📖 Read the main README: open README.md")
    print("3. 🔧 Setup PyTorch: follow docs/PYTORCH_SETUP.md")
    print("4. 🧪 Validate setup: cd tests/integration && python comprehensive_test.py")
    print("5. 🚀 Start building: explore src/ modules")
    
    print(f"\n✨ BENEFITS OF ORGANIZED STRUCTURE")
    print("=" * 60)
    print("✅ Easy to find specific functionality")
    print("✅ Clear separation of concerns") 
    print("✅ Logical import paths")
    print("✅ Scalable architecture")
    print("✅ Professional project layout")
    print("✅ Simple navigation")
    
    print(f"\n🏆 YOUR PIPELINE IS READY!")
    print("=" * 60)
    print("The complete 3D anomaly detection pipeline is now organized")
    print("in a user-friendly structure. Start with the examples and")
    print("explore the modular components to build your solution!")

if __name__ == "__main__":
    main()
