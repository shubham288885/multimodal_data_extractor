from openai import OpenAI
import os
from typing import List, Dict, Any

class LLMAnswerGenerator:
    def __init__(self):
        """Initialize the LLM Answer Generator with DeepSeek-R1 model via NVIDIA NIMS"""
        self.client = OpenAI(
            base_url=os.getenv('NVIDIA_LLM_ENDPOINT'),
            api_key=os.getenv('NVIDIA_LLM_KEY')
        )
        self.model = "deepseek-ai/deepseek-r1"
        
    def generate_answer(self, query: str, context: List[Dict[str, Any]], stream: bool = False) -> str:
        """
        Generate an answer to a query based on retrieved context documents
        
        Args:
            query: The user's question
            context: List of retrieved documents with text and metadata
            stream: Whether to stream the response (for UI)
            
        Returns:
            Generated answer as a string
        """
        # Format the context into a string
        formatted_context = self._format_context(context)
        
        # Create the prompt with the query and context
        prompt = self._create_prompt(query, formatted_context)
        
        print(f"Generating answer with DeepSeek-R1 for query: '{query[:50]}...'")
        
        # Define the system prompt for carbon accounting
        system_prompt = """You are an advanced carbon accounting AI designed to calculate Scope 3 greenhouse gas (GHG) emissions for any user-defined activity, product, or supply chain transaction. Your outputs align with the GHG Protocol, ISO 14064, and IPCC guidelines, using regionally and industry-specific emission factors from verified databases (e.g., EPA, DEFRA, Ecoinvent, IPCC 2023).

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

User Interaction Guidelines:
- Ask for missing parameters (e.g., "Specify transport mode: air, road, rail?").
- Be transparent about uncertainties.
- Scale to complex supply chains."""
        
        try:
            # Call the LLM
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.7,
                max_tokens=1024,
                stream=stream
            )
            
            # Handle streaming or non-streaming response
            if stream:
                return completion  # Return the stream object for the UI to consume
            else:
                # For non-streaming, collect the full response
                answer = completion.choices[0].message.content
                return answer
                
        except Exception as e:
            print(f"Error generating answer with LLM: {str(e)}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format the retrieved documents into a string for the prompt"""
        formatted_docs = []
        
        for i, doc in enumerate(context):
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
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the LLM"""
        return f"""Please answer the following question based only on the provided context documents:

Question: {query}

Context Documents:
{context}

Answer:"""
