# Travel-Agent-Code
The multi-agent approach creates more detailed and specialized travel plans, while the Pydantic models ensure consistency and validation throughout the application.
"""
Travel Planning Agent using OpenAI's API

This code creates a travel planning agent that leverages structured prompting 
and system design to handle travel-related requests.
"""

# Import necessary libraries
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Configuration for OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client with proper error handling
try:
    client = AsyncOpenAI(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# Define message types for structured communication
class SystemMessage:
    def __init__(self, content: str, source: str = "system"):
        self.content = content
        self.source = source
        
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.source, "content": self.content}

class UserMessage:
    def __init__(self, content: str, source: str = "user"):
        self.content = content
        self.source = source
        
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.source, "content": self.content}

class AssistantMessage:
    def __init__(self, content: Union[str, Dict], source: str = "assistant"):
        self.content = content
        self.source = source
        
    def to_dict(self) -> Dict[str, Union[str, Dict]]:
        return {"role": self.source, "content": self.content}

# Define structured data models for travel planning
class Destination(BaseModel):
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    timezone: str = Field(..., description="Timezone of the destination")
    local_currency: str = Field(..., description="Local currency")
    best_areas_to_stay: List[str] = Field(..., description="Recommended areas to stay")
    
class TravelDates(BaseModel):
    departure_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: str = Field(..., description="Return date (YYYY-MM-DD)")
    duration_days: int = Field(..., description="Total duration of the trip in days")

class TravelParty(BaseModel):
    adults: int = Field(..., description="Number of adults")
    children: int = Field(..., description="Number of children")
    children_ages: Optional[List[int]] = Field(None, description="Ages of children, if applicable")

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    description: str = Field(..., description="Brief description")
    suitable_for_children: bool = Field(..., description="Whether suitable for children")
    duration_hours: float = Field(..., description="Approximate duration in hours")
    estimated_cost: str = Field(..., description="Estimated cost per person")

class DailyPlan(BaseModel):
    day: int = Field(..., description="Day number of the trip")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    activities: List[Activity] = Field(..., description="Activities planned for the day")
    accommodation: str = Field(..., description="Where to stay this night")
    transportation: str = Field(..., description="Transportation methods for this day")

class TravelPlan(BaseModel):
    from_destination: Destination = Field(..., description="Origin location")
    to_destination: Destination = Field(..., description="Destination location")
    travel_dates: TravelDates = Field(..., description="Travel dates information")
    travel_party: TravelParty = Field(..., description="Information about the travelers")
    daily_plans: List[DailyPlan] = Field(..., description="Day-by-day breakdown of the trip")
    estimated_total_budget: str = Field(..., description="Estimated total budget for the trip")
    packing_suggestions: List[str] = Field(..., description="Suggestions for what to pack")
    travel_tips: List[str] = Field(..., description="Helpful tips for this specific trip")

# Enhanced error handling for API calls
async def get_travel_plan(query: str) -> Dict[str, Any]:
    """
    Get a travel plan based on the user query.
    
    Args:
        query: User's travel request
        
    Returns:
        A structured travel plan
    """
    try:
        # Define system message with specialized agents
        system_message = SystemMessage("""
        You are an expert travel planning AI with the following specialized agents:
        - FlightExpert: For finding and recommending flights
        - AccommodationSpecialist: For recommending hotels and places to stay
        - ActivitiesPlanner: For suggesting age-appropriate activities
        - BudgetAdvisor: For providing cost estimates and budget advice
        - DestinationInfo: For providing information about destinations
        - DefaultAgent: For handling general requests
        
        For families with children, ensure all recommendations are family-friendly.
        Always include:
        1. Day-by-day itinerary
        2. Estimated costs
        3. Weather-appropriate packing suggestions
        4. Local transportation options
        5. Child-friendly activities and accommodations
        
        Respond with a complete travel plan in JSON format matching the TravelPlan schema.
        """)
        
        # Create message list
        messages = [
            system_message.to_dict(),
            {"role": "user", "content": query}
        ]
        
        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=4000
                )
                
                # Parse and validate the response
                response_content = response.choices[0].message.content
                travel_plan = json.loads(response_content)
                
                # Validate against our schema
                validated_plan = TravelPlan(**travel_plan)
                return validated_plan.dict()
                
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise ValueError("Failed to parse response as JSON")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
    except Exception as e:
        raise RuntimeError(f"Failed to generate travel plan: {e}")

# Function to calculate travel dates with flexible duration
def calculate_travel_dates(departure_date_str: Optional[str] = None, 
                           duration_days: int = 7) -> Dict[str, str]:
    """
    Calculate travel dates based on departure date and duration.
    
    Args:
        departure_date_str: Optional departure date string (YYYY-MM-DD)
        duration_days: Duration of the trip in days
        
    Returns:
        Dictionary with departure_date, return_date, and duration_days
    """
    # If no departure date specified, default to 2 weeks from today
    if not departure_date_str:
        departure_date = datetime.now() + timedelta(days=14)
    else:
        departure_date = datetime.strptime(departure_date_str, '%Y-%m-%d')
    
    return_date = departure_date + timedelta(days=duration_days)
    
    return {
        "departure_date": departure_date.strftime('%Y-%m-%d'),
        "return_date": return_date.strftime('%Y-%m-%d'),
        "duration_days": duration_days
    }

# Example usage of the travel planning agent
async def main():
    try:
        # Example query with a family traveling from Singapore to Melbourne
        query = "Create a travel plan for a family with 2 kids (ages 5 and 8) from Singapore to Melbourne for 10 days starting next month"
        
        # Get the travel plan
        travel_plan = await get_travel_plan(query)
        
        # Pretty print the result
        print(json.dumps(travel_plan, indent=2))
        
    except Exception as e:
        print(f"Error generating travel plan: {e}")

if __name__ == "__main__":
    asyncio.run(main())
