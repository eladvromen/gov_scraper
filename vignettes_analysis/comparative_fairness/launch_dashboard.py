#!/usr/bin/env python3
"""
🚀 Ethical AI Assessment Dashboard Launcher

Quick launcher for the comprehensive ethical AI assessment dashboard.
Automatically configures paths and launches the Streamlit application.
"""

import subprocess
import sys
from pathlib import Path
import os

def main():
    """Launch the Ethical AI Assessment Dashboard"""
    
    print("🔬 Ethical AI Assessment Dashboard Launcher")
    print("=" * 60)
    
    # Get the dashboard path
    dashboard_path = Path(__file__).parent / "ethical_ai_assessment_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Error: Dashboard file not found!")
        print(f"Expected: {dashboard_path}")
        return
    
    # Set working directory to the comparative fairness folder
    os.chdir(Path(__file__).parent)
    
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🎯 Dashboard file: {dashboard_path.name}")
    print("🌐 Starting Streamlit server...")
    print("\n💡 The dashboard will open in your default browser.")
    print("📊 Features available:")
    print("   • Fairness divergence analysis")
    print("   • Normative alignment assessment") 
    print("   • Geometric bias vector visualization")
    print("   • Statistical significance patterns")
    print("   • Topic-attribute intersectional analysis")
    print("   • Individual case exploration")
    print("   • Automated scientific insights")
    print("\n🔧 Use Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {str(e)}")
        print("\n🔧 Try running manually:")
        print(f"   cd {Path(__file__).parent}")
        print(f"   streamlit run {dashboard_path.name}")

if __name__ == "__main__":
    main() 