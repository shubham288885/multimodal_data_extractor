"""
A wrapper around PyMuPDF (fitz) to avoid import conflicts.
This acts as a buffer between our code and PyMuPDF imports.
"""

import sys
import os

# Try to ensure our PyMuPDF is imported correctly
def get_fitz():
    """
    Safe import of PyMuPDF to avoid conflicts with other packages.
    """
    try:
        # Direct import
        import fitz
        return fitz
    except ImportError:
        print("PyMuPDF not found in direct import path. Trying alternative...")
    
    # If direct import fails, try to find it in site-packages
    try:
        import site
        site_packages = site.getsitepackages()
        for site_pkg in site_packages:
            pymupdf_path = os.path.join(site_pkg, 'fitz')
            if os.path.exists(pymupdf_path) and pymupdf_path not in sys.path:
                sys.path.insert(0, os.path.dirname(pymupdf_path))
                break
        
        import fitz
        return fitz
    except (ImportError, Exception) as e:
        print(f"Error importing PyMuPDF: {e}")
        raise

# Make fitz available when this module is imported
fitz = get_fitz()

__all__ = ['fitz'] 