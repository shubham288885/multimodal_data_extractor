import numpy as np
from PIL import Image
import requests
import base64
import os
import io

class OCRProcessor:
    def __init__(self):
        self.api_key = os.getenv('NVIDIA_PADDLEOCR_KEY')
        self.paddleocr_endpoint = os.getenv('NVIDIA_PADDLEOCR_ENDPOINT')
        
    def process_image(self, image):
        """Extract text from image using NVIDIA PaddleOCR"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        payload = {
            "input": [{
                "type": "image_url",
                "url": f"data:image/png;base64,{img_base64}"
            }]
        }
        
        try:
            response = requests.post(
                self.paddleocr_endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return self._extract_text_from_response(result)
        except Exception as e:
            raise Exception(f"OCR API request failed: {str(e)}")
    
    def process_table(self, table_image):
        """Special processing for table images using NVIDIA PaddleOCR"""
        # Convert image to base64
        img_byte_arr = io.BytesIO()
        table_image.save(img_byte_arr, format='PNG')
        image_b64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        payload = {
            "input": [{
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }]
        }
        
        response = requests.post(
            self.paddleocr_endpoint,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            # Convert OCR results to structured table format
            table_data = self._structure_table_data(result)
            return table_data
        else:
            raise Exception(f"PaddleOCR API request failed: {response.text}")
    
    def _extract_text_from_response(self, response):
        """Extract text from PaddleOCR response"""
        text_blocks = []
        for result in response.get('results', []):
            for text_line in result.get('text_lines', []):
                text_blocks.append(text_line.get('text', ''))
        return '\n'.join(text_blocks)
    
    def _structure_table_data(self, response):
        """Convert NVIDIA PaddleOCR response to structured table format"""
        try:
            # Get all text lines with their positions
            text_lines = []
            for result in response.get('results', []):
                for text_line in result.get('text_lines', []):
                    text = text_line.get('text', '')
                    box = text_line.get('box', [])
                    if box and text:
                        # Calculate average y-coordinate for the text line
                        y_coord = sum(point[1] for point in box) / len(box)
                        x_coord = box[0][0]  # Use left-most x-coordinate
                        text_lines.append((text, x_coord, y_coord))
            
            # Group text lines by similar y-coordinates (within 10 pixels)
            rows = {}
            for text, x_coord, y_coord in text_lines:
                row_key = int(y_coord / 10)
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append((x_coord, text))
            
            # Sort rows by y-coordinate and sort cells within each row by x-coordinate
            structured_data = []
            for row_key in sorted(rows.keys()):
                row = sorted(rows[row_key], key=lambda x: x[0])  # Sort by x-coordinate
                structured_data.append([cell[1] for cell in row])  # Extract just the text
            
            return structured_data
            
        except Exception as e:
            raise Exception(f"Error structuring table data: {e}") 