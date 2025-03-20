# Running the Multimodal PDF Extractor

## Current Status
The simplified demo app is now running at: **http://localhost:8501**

You should be able to access it by opening your web browser and navigating to this URL.

## If the App is Not Running

If you need to restart the application, follow these steps:

1. Open your terminal/command prompt
2. Navigate to the project directory:
   ```
   cd C:\Users\Admin\Desktop\Carbon6_main\multimodal_pdf_extractor\multimodal_data_extractor
   ```
3. Make sure your virtual environment is activated:
   ```
   ..\..\fresh_env\Scripts\activate
   ```
4. Run the Streamlit app:
   ```
   streamlit run app/frontend/simple_app.py
   ```

## Running the Full Application

The full application has import path issues. To fix them:

1. Create Python package structure:
   - Add empty `__init__.py` files to all directories in the project

2. OR use this command to set PYTHONPATH before running:
   ```
   $env:PYTHONPATH="C:\Users\Admin\Desktop\Carbon6_main\multimodal_pdf_extractor\multimodal_data_extractor"; streamlit run app/frontend/streamlit_app.py
   ```

## Troubleshooting
If you encounter issues:
1. Check that all dependencies are installed
2. Ensure your virtual environment is activated
3. Try running with Python directly to see error messages:
   ```
   python app/frontend/simple_app.py
   ```

For detailed error logs, check the Streamlit logs in the terminal. 