from .fitz_wrapper import fitz  # Safe import of PyMuPDF
import cv2
import numpy as np
from PIL import Image
import requests
import os
import io
import base64

class DocumentExtractor:
    def __init__(self):
        self.yolox_key = os.getenv('NVIDIA_YOLOX_KEY')
        self.deplot_key = os.getenv('NVIDIA_DEPLOT_KEY')
        self.yolox_endpoint = os.getenv('NVIDIA_YOLOX_ENDPOINT')
        self.deplot_endpoint = os.getenv('NVIDIA_DEPLOT_ENDPOINT')
        
    def _detect_objects(self, image):
        """Detect page elements using NVIDIA YOLOX"""
        # Convert image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {self.yolox_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Prepare payload according to NVIDIA API format
        payload = {
            "input": [{
                "type": "image_url",
                "url": f"data:image/png;base64,{img_base64}"
            }]
        }
        
        response = requests.post(
            self.yolox_endpoint,
            headers=headers,
            json=payload  # Send as JSON instead of raw data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Transform the response to match your expected format
            predictions = []
            for detection in result.get('results', []):
                for box in detection.get('boxes', []):
                    predictions.append({
                        'label': box.get('label'),
                        'box': box.get('coordinates'),  # [x1, y1, x2, y2]
                        'confidence': box.get('confidence', 0.0)
                    })
            return predictions
        else:
            raise Exception(f"YOLOX API request failed: {response.status_code} - {response.text}")
    
    def _process_chart(self, image, bbox):
        """Process chart using Google-DePlot"""
        chart_img = image.crop(bbox)
        img_byte_arr = io.BytesIO()
        chart_img.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        headers = {
            "Authorization": f"Bearer {self.deplot_key}",
            "Accept": "text/event-stream"
        }
        
        payload = {
            "messages": [{
                "role": "user",
                "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{img_base64}" />'
            }],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.20,
            "stream": True
        }
        
        try:
            response = requests.post(
                self.deplot_endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Process streaming response
            chart_data = []
            for line in response.iter_lines():
                if line:
                    chart_data.append(line.decode("utf-8"))
            
            return {
                'bbox': bbox,
                'data': '\n'.join(chart_data)
            }
        except Exception as e:
            raise Exception(f"DePlot API request failed: {str(e)}")
    
    def _process_table(self, image, bbox):
        """Process table region and return structured data"""
        # Crop table region
        table_img = image.crop(bbox)
        return {
            'bbox': bbox,
            'image': table_img  # This will be processed by OCR later
        }
    
    def extract_from_pdf(self, pdf_path):
        """Extract content from PDF including text, tables, and charts"""
        doc = fitz.open(pdf_path)
        extracted_content = {
            'text': [],
            'tables': [],
            'charts': []
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page as image for object detection
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            try:
                # Detect objects using YOLOX
                objects = self._detect_objects(img)
                
                # Process each detected object
                for obj in objects:
                    if obj['confidence'] < 0.5:  # Skip low confidence detections
                        continue
                        
                    if obj['label'] == 'chart':
                        try:
                            chart_data = self._process_chart(img, obj['box'])
                            extracted_content['charts'].append(chart_data)
                        except Exception as e:
                            print(f"Error processing chart on page {page_num}: {str(e)}")
                            
                    elif obj['label'] == 'table':
                        try:
                            table_data = self._process_table(img, obj['box'])
                            extracted_content['tables'].append(table_data)
                        except Exception as e:
                            print(f"Error processing table on page {page_num}: {str(e)}")
            
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
            
            # Extract text content
            text = page.get_text()
            extracted_content['text'].append({
                'page_num': page_num,
                'content': text
            })
            
        return extracted_content 