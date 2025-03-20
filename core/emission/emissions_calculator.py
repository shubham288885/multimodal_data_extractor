import os
from typing import Dict, Any, List
from openai import OpenAI
from .emission_factor_client import EmissionFactorClient

class EmissionsCalculator:
    """
    Calculate greenhouse gas emissions based on document content
    using DeepSeek-R1 LLM via NVIDIA NIMS
    """
    
    def __init__(self, emission_factor_client=None):
        """
        Initialize the emissions calculator
        
        Args:
            emission_factor_client: Optional client for emission factor lookup
        """
        self.client = OpenAI(
            base_url=os.getenv('NVIDIA_LLM_ENDPOINT'),
            api_key=os.getenv('NVIDIA_LLM_KEY')
        )
        self.model = "deepseek-ai/deepseek-r1"
        self.emission_factor_client = emission_factor_client or EmissionFactorClient()
        
    def extract_activities(self, document_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract emission-relevant activities from document content
        
        Args:
            document_content: List of document segments with text and metadata
            
        Returns:
            List of activity descriptions with details
        """
        # Format the document content for the prompt
        formatted_content = self._format_document_content(document_content)
        
        # Create the prompt for activity extraction
        prompt = self._create_activity_extraction_prompt(formatted_content)
        
        # Call the LLM to extract activities
        try:
            print("Sending request to LLM for activity extraction...")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_activity_extraction_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.7,
                max_tokens=1024
            )
            
            # Parse the response to get activities
            activities_json = completion.choices[0].message.content
            print("Received LLM response for activity extraction:")
            print(f"Response starts with: {activities_json[:200]}...")
            
            # Try to parse the JSON response
            try:
                import json
                import re
                
                # Try to extract JSON from the response if it's not valid JSON already
                # This helps when the LLM wraps JSON in markdown or adds extra text
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                json_match = re.search(json_pattern, activities_json)
                
                if json_match:
                    # Extract JSON from code block
                    extracted_json = json_match.group(1).strip()
                    activities = json.loads(extracted_json)
                else:
                    # Try to parse directly
                    activities = json.loads(activities_json)
                
                if 'activities' not in activities:
                    print("Response doesn't contain 'activities' key. Raw response:")
                    print(activities)
                    # Try to build a valid structure if possible
                    if isinstance(activities, list):
                        return activities  # Assume it's a list of activities
                    else:
                        return []  # No activities found
                
                activities_list = activities.get('activities', [])
                print(f"Successfully extracted {len(activities_list)} activities")
                return activities_list
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse activity extraction response as JSON: {e}")
                print("Attempting to build activities from non-JSON response...")
                
                # Try to extract structured information from non-JSON response
                activities = self._extract_activities_from_text(activities_json)
                if activities:
                    print(f"Extracted {len(activities)} activities from text response")
                    return activities
                return []
                
        except Exception as e:
            print(f"Error extracting activities: {str(e)}")
            return []
    
    def _extract_activities_from_text(self, text):
        """Extract activities from non-JSON text response"""
        import re
        
        activities = []
        
        # Look for "Activity" or "Description" patterns
        activity_patterns = [
            r'(?:Activity|Description):\s*(.*?)(?:\n|$)',
            r'(\d+\.\s*.*?)(?:\n|$)',
            r'- (.*?)(?:\n|$)'
        ]
        
        for pattern in activity_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if len(match.strip()) > 5:  # Avoid very short matches
                        activities.append({
                            "description": match.strip(),
                            "details": {"source": "text extraction fallback"}
                        })
        
        return activities
    
    def calculate_emissions(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate greenhouse gas emissions for the extracted activities
        
        Args:
            activities: List of activities extracted from the document
            
        Returns:
            Structured emission calculation results
        """
        # Fetch emission factors for each activity
        activities_with_factors = []
        
        for activity in activities:
            try:
                # Get appropriate emission factor for the activity
                factor = self.emission_factor_client.get_appropriate_emission_factor(
                    activity['description'],
                    activity.get('details', {})
                )
                
                # Add the emission factor to the activity
                activity['emission_factor'] = factor
                activities_with_factors.append(activity)
                
            except Exception as e:
                print(f"Error getting emission factor for activity '{activity['description']}': {str(e)}")
                # Still include the activity even without a factor
                activities_with_factors.append(activity)
        
        # Create the prompt for emissions calculation
        prompt = self._create_emissions_calculation_prompt(activities_with_factors)
        
        # Call the LLM to calculate emissions
        try:
            print("Sending request to LLM for emissions calculation...")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_emissions_calculation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.7,
                max_tokens=1500
            )
            
            # Parse the LLM response
            emissions_json = completion.choices[0].message.content
            print("Received LLM response for emissions calculation:")
            print(f"Response starts with: {emissions_json[:200]}...")
            
            # Try to parse the JSON response
            try:
                import json
                import re
                
                # Try to extract JSON from the response if it's not valid JSON already
                # This helps when the LLM wraps JSON in markdown or adds extra text
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_match = re.search(json_pattern, emissions_json)
                
                if json_match:
                    # Extract JSON from code block
                    extracted_json = json_match.group(1).strip()
                    print("Extracted JSON from markdown code block")
                    emissions_results = json.loads(extracted_json)
                else:
                    # Try to parse directly
                    emissions_results = json.loads(emissions_json)
                
                print("Successfully parsed emissions calculation result")
                return emissions_results
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse emissions calculation response as JSON: {e}")
                print("Creating fallback emissions calculation response...")
                
                # Create a simplified response with the raw text
                return self._create_fallback_emissions_result(emissions_json, activities_with_factors)
                
        except Exception as e:
            print(f"Error calculating emissions: {str(e)}")
            return {"error": f"Error calculating emissions: {str(e)}"}
            
    def _create_fallback_emissions_result(self, raw_text, activities_with_factors):
        """Create a fallback structured emissions result when JSON parsing fails"""
        # Build a simple structured response
        result = {
            "error": "Failed to parse structured calculation, showing text results",
            "raw_calculation": raw_text[:1000],  # Truncate to avoid excessive text
            "emission_sources": []
        }
        
        # Extract total emissions from text if available
        import re
        total_match = re.search(r'total.*emissions:?\s*(\d+\.?\d*)', raw_text, re.IGNORECASE)
        if total_match:
            result["total_scope_3_emissions"] = float(total_match.group(1))
            
        # Create basic emission sources based on activities
        for activity in activities_with_factors:
            if 'emission_factor' in activity:
                factor = activity['emission_factor']
                quantity = "1"  # Default quantity
                
                # Try to extract quantity from activity details
                if 'details' in activity and 'quantity' in activity['details']:
                    quantity = activity['details']['quantity']
                    
                # Try to extract a quantity from the text using regex
                description = activity['description'].lower()
                quantity_match = re.search(r'(\d+\.?\d*)\s*(kwh|kw|mwh|therms|liters|gallons)', description)
                if quantity_match:
                    quantity = quantity_match.group(0)
                
                # Calculate emissions using the factor value (with fallback to 1.0)
                factor_value = float(factor.get('value', 1.0))
                quantity_value = 1.0  # Default if we can't parse
                
                # Try to extract numeric value from quantity
                quantity_numeric_match = re.search(r'(\d+\.?\d*)', str(quantity))
                if quantity_numeric_match:
                    try:
                        quantity_value = float(quantity_numeric_match.group(1))
                    except:
                        pass
                
                total = factor_value * quantity_value
                
                # Add to emission sources
                source = {
                    "source": activity['description'],
                    "processes": [{
                        "name": factor.get('description', 'electricity_consumption'),
                        "description": f"Emissions from {activity['description']}",
                        "parameters": {
                            "quantity": str(quantity),
                            "emission_factor": f"{factor.get('value', 'unknown')} {factor.get('unit', 'kg CO2e')}",
                            "calculation": f"{quantity} × {factor.get('value', 'unknown')} = {total} kg CO2e",
                            "total_emissions": total
                        }
                    }],
                    "total_emissions": total
                }
                result["emission_sources"].append(source)
                
        # Calculate the total emissions
        total_emissions = sum(source["total_emissions"] for source in result["emission_sources"])
        if "total_scope_3_emissions" not in result:
            result["total_scope_3_emissions"] = total_emissions
            
        # Add data sources
        result["data_sources"] = ["Fallback emission factors when API response could not be parsed as JSON"]
        
        return result
    
    def _format_document_content(self, document_content: List[Dict[str, Any]]) -> str:
        """Format document content for the prompt"""
        formatted_docs = []
        
        for i, doc in enumerate(document_content):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Format metadata for better context
            meta_str = ""
            if metadata:
                source = metadata.get('document_path', 'Unknown source')
                page = metadata.get('page_num', 'Unknown page')
                meta_str = f"[Source: {source}, Page: {page}]"
            
            formatted_docs.append(f"Document {i+1} {meta_str}:\n{text}\n")
        
        return "\n".join(formatted_docs)
    
    def _create_activity_extraction_prompt(self, document_content: str) -> str:
        """Create the prompt for activity extraction"""
        return f"""Please analyze the following document content and extract all activities that could generate greenhouse gas emissions. This document may be a utility bill (electricity, gas, water), transportation receipt, purchase invoice, or similar document. Pay special attention to:

1. Electricity usage (kWh, MWh)
2. Natural gas consumption (therms, cubic meters, BTU)
3. Water consumption 
4. Fuel purchases (gallons, liters)
5. Transportation details (miles, km)
6. Purchased goods with quantities
7. Waste disposal information

For utility bills, consider the total energy consumption as an activity, even if the document just shows the billing amount. Look for consumption units like kWh, therms, etc.

Document Content:
{document_content}

Extract and list ALL activities that could generate greenhouse gas emissions in JSON format. Include as much detail as possible about quantities, regions, transport modes, etc. Even if the document only shows billing information, try to identify the underlying consumption activity."""
    
    def _create_emissions_calculation_prompt(self, activities_with_factors: List[Dict[str, Any]]) -> str:
        """Create the prompt for emissions calculation"""
        activities_str = ""
        
        for i, activity in enumerate(activities_with_factors):
            # Format the activity details
            details_str = ""
            if 'details' in activity:
                for key, value in activity['details'].items():
                    details_str += f"\n    {key}: {value}"
            
            # Format the emission factor if available
            factor_str = ""
            if 'emission_factor' in activity:
                factor = activity['emission_factor']
                factor_str = f"\nEmission Factor: {factor.get('description', 'Unknown')} - Value: {factor.get('value', 'Unknown')} {factor.get('unit', '')}"
            
            activities_str += f"""Activity {i+1}: {activity['description']}
Details:{details_str or ' None provided'}
{factor_str}

"""
        
        return f"""Please calculate the greenhouse gas emissions (Scope 1, 2, and 3) for the following activities:

{activities_str}

For each activity, break down the emission sources and processes, and provide detailed calculations. Return the results in a structured JSON format as described in your instructions."""
    
    def _get_activity_extraction_system_prompt(self) -> str:
        """Get the system prompt for activity extraction"""
        return """You are an advanced carbon accounting AI designed to extract emission-relevant activities from documents. Your task is to identify all activities mentioned in the provided document that could generate greenhouse gas emissions, particularly focusing on:

1. Utility bills (electricity, natural gas, water)
2. Transportation (air travel, road transport, shipping)
3. Energy consumption (electricity, fuel, heating)
4. Material procurement and usage (paper, plastics, electronics)
5. Waste generation and disposal
6. Manufacturing processes
7. Construction activities
8. Agricultural activities
9. Service provision

For utility bills:
- If you see an electricity bill, the electricity consumption is an emission-relevant activity
- If you see a natural gas bill, the gas consumption is an emission-relevant activity
- Even if only monetary amounts are shown, assume there is underlying energy consumption
- Look for billing periods and total consumption amounts (kWh, therms, etc.)
- If a specific consumption value isn't provided, note this in the details

For each activity, extract:
- A clear description of the activity
- Quantities or amounts mentioned (if any)
- Regions or locations (if mentioned)
- Time periods (if relevant)
- Any other details that would help calculate the emissions

Respond in JSON format with a list of activities, like this:
{
  "activities": [
    {
      "description": "Electricity consumption",
      "details": {
        "quantity": "500 kWh",
        "region": "California, USA",
        "time_period": "January 2025",
        "bill_amount": "$75.50"
      }
    },
    {
      "description": "Natural gas usage for heating",
      "details": {
        "quantity": "50 therms",
        "region": "Northeast USA",
        "time_period": "December 2024"
      }
    }
  ]
}

IMPORTANT: For utility bills, if specific consumption values aren't provided but you can see it's an electricity or gas bill, still include it as an activity with whatever information is available. Make reasonable assumptions based on the document context."""
    
    def _get_emissions_calculation_system_prompt(self) -> str:
        """Get the system prompt for emissions calculation"""
        return """You are an advanced carbon accounting AI designed to calculate Scope 1, 2, and 3 greenhouse gas (GHG) emissions for any user-defined activity, product, or supply chain transaction. Your outputs align with the GHG Protocol, ISO 14064, and IPCC guidelines, using regionally and industry-specific emission factors from verified databases (e.g., EPA, DEFRA, Ecoinvent, IPCC 2023).

Core Functionality:
- *Process Identification*: Break down activities into emission-generating stages.
- *Emission Factor Selection*: Prioritize region-specific, industry-specific, or global average emission factors.
- *Calculation & Validation*: Compute emissions, validate against benchmarks, and highlight uncertainties.
- *Output Requirements*: Return structured JSON with granular data and plain-language summaries.
- **only calculate for transportation emission if you are provided with the details.
- **always provide the units.
- **dont give recommendations to cut down emission your outputs will be given to another model to guide the emission mitigation.

Example Output Structure:

{
  "activity_description": "Purchase of 20 kg plastic bags (20 km transport)",
  "emission_sources": [
    {
      "source": "production",
      "processes": [
        {
          "name": "polyethylene_production",
          "description": "Crude oil refining, polymerization, and bag manufacturing",
          "parameters": {
            "quantity": "20 kg",
            "emission_factor": "1.5 kg CO2e/kg (IPCC 2023, Plastics Manufacturing)",
            "calculation": "20 kg × 1.5 kg CO2e/kg = 30 kg CO2e",
            "total_emissions": 30.0
          }
        },
        {
          "name": "raw_material_extraction",
          "description": "Petroleum extraction and refining",
          "parameters": {
            "quantity": "20 kg",
            "emission_factor": "0.115 kg CO2e/kg (Ecoinvent 2023, Crude Oil)",
            "calculation": "20 kg × 0.115 kg CO2e/kg = 2.3 kg CO2e",
            "total_emissions": 2.3
          }
        }
      ],
      "total_emissions": 32.3
    },
    {
      "source": "transportation",
      "processes": [
        {
          "name": "road_freight",
          "description": "Round-trip diesel vehicle transport",
          "parameters": {
            "distance": "40 km (20 km × 2)",
            "vehicle_type": "light-duty truck",
            "emission_factor": "0.27 kg CO2e/km (EPA 2023)",
            "calculation": "40 km × 0.27 kg CO2e/km = 10.8 kg CO2e",
            "total_emissions": 10.8
          }
        }
      ],
      "total_emissions": 10.8
    }
  ],
  "total_scope_3_emissions": 43.1,
  "assumptions": [
    "Defaulted to landfill disposal (emission factor: 0.1 kg CO2e/kg).",
    "Vehicle type inferred as light-duty truck (user did not specify)."
  ],
  "data_sources": [
    "IPCC 2023: Plastics Production Emission Factors",
    "EPA 2023: Transportation Emission Factors"
  ]
}

Use the emission factors provided when available, and make reasonable assumptions when needed. Be transparent about all assumptions made.""" 