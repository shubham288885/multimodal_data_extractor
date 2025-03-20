import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleDemoApp:
    def __init__(self):
        st.set_page_config(
            page_title="Multimodal PDF Extractor Demo",
            page_icon="🔍",
            layout="wide"
        )

    def render_sidebar(self):
        with st.sidebar:
            st.title("📚 Navigation")
            
            # Tabs
            tabs = {
                "search": "🔍 Document Search",
                "emissions": "🌍 Emissions Calculator"
            }
            
            # Create tab buttons
            st.markdown("### Main Features")
            for tab_id, tab_name in tabs.items():
                st.button(tab_name, key=f"tab_{tab_id}")
            
            st.markdown("---")
            st.markdown("### Status")
            st.success("✅ Environment setup successful!")

    def render_main_interface(self):
        st.title("🔍 Document Search & Emissions Calculator - Demo")
        
        st.markdown("""
        ## 🎉 Success!
        
        This simplified demo confirms that your environment is working correctly. 
        
        The actual application couldn't load because of module import conflicts. To fix this:
        
        1. **Check project structure**: Ensure all imports in the original Streamlit app use the correct paths
        2. **Install dependencies**: Make sure all required dependencies are installed
        3. **PYTHONPATH**: Consider adding the project root to your PYTHONPATH environment variable
        
        You can now proceed to debug the main application.
        """)
        
        st.info("This is a simplified demo version that confirms the Streamlit environment is working correctly.")
        
        # Document Upload Section for demonstration
        st.markdown("## 📤 Document Upload Demo")
        uploaded_files = st.file_uploader(
            "Upload PDF documents (demo only - not functional)", 
            type=['pdf'],
            accept_multiple_files=True
        )

    def run(self):
        self.render_sidebar()
        self.render_main_interface()

if __name__ == "__main__":
    app = SimpleDemoApp()
    app.run() 