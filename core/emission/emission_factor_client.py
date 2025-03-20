import requests
import os
from typing import List, Dict, Any
import json
from ..embedding.embedder import DocumentEmbedder

class EmissionFactorClient:
    """Client for accessing emission factor data through the Emission Factors API"""
    
    def __init__(self, api_url=None):
        """
        Initialize the emission factor client
        
        Args:
            api_url: The URL of the emission factors API (defaults to environment variable)
        """
        self.api_url = api_url or os.getenv('EMISSION_FACTORS_API_URL', "http://localhost:8000/api/v1/emission-factors/search")
        self.embedder = DocumentEmbedder()  # Use the existing document embedder
        self.fallback_factors = self._load_fallback_factors()
        
    def _load_fallback_factors(self):
        """Load fallback emission factors to use when API is not available"""
        return {
            "electricity": {
                "description": "Electricity generation - average grid mix (global average)",
                "value": 0.475,
                "unit": "kg CO2e/kWh"
            },
            "natural_gas": {
                "description": "Natural gas combustion for heating",
                "value": 0.198,
                "unit": "kg CO2e/kWh"
            },
            "fuel_oil": {
                "description": "Fuel oil combustion",
                "value": 2.68,
                "unit": "kg CO2e/liter"
            },
            "vehicle_gasoline": {
                "description": "Gasoline vehicle emissions",
                "value": 2.31,
                "unit": "kg CO2e/liter"
            },
            "vehicle_diesel": {
                "description": "Diesel vehicle emissions",
                "value": 2.68,
                "unit": "kg CO2e/liter"
            },
            "air_travel_short": {
                "description": "Air travel - short haul (<500 km)",
                "value": 0.18,
                "unit": "kg CO2e/km/passenger"
            },
            "air_travel_medium": {
                "description": "Air travel - medium haul (500-1500 km)",
                "value": 0.13,
                "unit": "kg CO2e/km/passenger"
            },
            "air_travel_long": {
                "description": "Air travel - long haul (>1500 km)",
                "value": 0.11,
                "unit": "kg CO2e/km/passenger"
            }
        }
        
    def search_emission_factors(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search emission factors semantically similar to the query
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            print(f"Searching emission factors for: {query}")
            # Prepare the request payload
            payload = {
                "query": query,
                "top_k": top_k
            }
            
            # Make the API request with timeout
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=5  # 5 second timeout
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse and return the response
            result = response.json()
            print(f"Found {len(result.get('results', []))} emission factors")
            return result
            
        except requests.RequestException as e:
            error_message = f"Error searching emission factors: {str(e)}"
            print(error_message)
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            
            # If it's a connection error, the API might not be running
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                print("Emission Factors API appears to be unavailable. Using fallback emission factors.")
                return self._get_fallback_results(query)
                
            raise Exception(f"Failed to search emission factors: {str(e)}")
            
    def _get_fallback_results(self, query: str) -> Dict[str, Any]:
        """Provide fallback emission factors when API is unavailable"""
        lower_query = query.lower()
        results = []
        
        # Match keywords in query to fallback factors
        if "electricity" in lower_query:
            results.append(self.fallback_factors["electricity"])
        elif "gas" in lower_query:
            results.append(self.fallback_factors["natural_gas"])
        elif "fuel" in lower_query or "oil" in lower_query:
            results.append(self.fallback_factors["fuel_oil"])
        elif "gasoline" in lower_query or "petrol" in lower_query:
            results.append(self.fallback_factors["vehicle_gasoline"])
        elif "diesel" in lower_query:
            results.append(self.fallback_factors["vehicle_diesel"])
        elif "air" in lower_query or "flight" in lower_query or "plane" in lower_query:
            results.append(self.fallback_factors["air_travel_medium"])
        else:
            # Default to electricity as it's commonly needed
            results.append(self.fallback_factors["electricity"])
            
        print(f"Using fallback emission factor: {results[0]['description']}")
        return {"results": results}
            
    def get_appropriate_emission_factor(self, activity_description: str, 
                                       details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find the most appropriate emission factor for a given activity
        
        Args:
            activity_description: Description of the activity
            details: Additional details about the activity (e.g., region, type)
            
        Returns:
            Dictionary with the best matching emission factor
        """
        # Generate a search query based on the activity and details
        search_query = activity_description
        
        if details:
            # Add relevant details to the search query
            if 'region' in details:
                search_query += f" {details['region']}"
            if 'type' in details:
                search_query += f" {details['type']}"
            if 'category' in details:
                search_query += f" {details['category']}"
        
        try:
            # Search for emission factors
            search_results = self.search_emission_factors(search_query)
            
            if not search_results or 'results' not in search_results or not search_results['results']:
                print(f"No emission factors found for activity: {activity_description}")
                
                # Use a fallback if available
                activity_lower = activity_description.lower()
                if "electricity" in activity_lower:
                    return self.fallback_factors["electricity"]
                elif "gas" in activity_lower:
                    return self.fallback_factors["natural_gas"]
                
                raise ValueError(f"No emission factors found for activity: {activity_description}")
            
            # Return the top result
            return search_results['results'][0]
            
        except Exception as e:
            print(f"Error getting emission factor, using fallback: {str(e)}")
            # Use a generic electricity factor as fallback
            return self.fallback_factors["electricity"] 