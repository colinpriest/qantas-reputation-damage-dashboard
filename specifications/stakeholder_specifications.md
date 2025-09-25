"""
SimulatedStakeholder Class Specifications
=========================================

This module defines specifications for a stakeholder simulation system that uses
psychological profiles and GPT-4o-mini to predict stakeholder reactions to news events.

Requirements:
- Python 3.8+
- instructor library
- pydantic
- openai
- python-dotenv
- asyncio for concurrent processing
"""

from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import hashlib
import json


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ReactionType(str, Enum):
    """Enumeration of possible stakeholder reactions."""
    DO_NOTHING = "Do Nothing"
    SHARE_OPINION = "Share Opinion with Others"
    SWITCH_TO_COMPETITOR = "Switch to Competitor"
    LEGAL_REGULATORY_ACTION = "Legal/Regulatory Action"
    SHAREHOLDER_ACTIVISM = "Shareholder Activism"
    DEVALUE_SHARES = "Devalue Shares"


# ============================================================================
# PYDANTIC MODELS FOR API INTERACTION
# ============================================================================

class NewsAnalysisRequest(BaseModel):
    """
    Model for a single news story analysis request to GPT-4o-mini.
    
    The system prompt will be the psychological profile.
    The user prompt will combine instructions and the news story.
    """
    news_story: str = Field(..., description="The news story text to analyze")
    psychological_profile: str = Field(..., description="The stakeholder's psychological profile and n-shot history")


class ReactionProbability(BaseModel):
    """Model for a single reaction and its probability."""
    reaction: ReactionType = Field(..., description="The type of reaction")
    probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Probability of this reaction occurring (0.0 to 1.0)"
    )
    
    @validator('probability')
    def validate_probability(cls, v):
        """Ensure probability is a valid percentage."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability must be between 0.0 and 1.0')
        return v


class GPTAnalysisResponse(BaseModel):
    """
    Model for the structured response from GPT-4o-mini for a single news story.
    
    This is what the instructor library will extract from the GPT response.
    """
    reputation_impact_score: int = Field(
        ..., 
        ge=-5, 
        le=5,
        description="Change in reputation/brand impact score (-5 to +5)"
    )
    reaction_probabilities: List[ReactionProbability] = Field(
        ...,
        description="List of potential reactions and their probabilities"
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="Optional reasoning for the scores (for debugging/transparency)"
    )
    
    @validator('reputation_impact_score')
    def validate_reputation_score(cls, v):
        """Ensure reputation score is within valid range."""
        if not -5 <= v <= 5:
            raise ValueError('Reputation impact score must be between -5 and 5')
        return v
    
    @validator('reaction_probabilities')
    def validate_reactions(cls, v):
        """
        Validate that all reaction types are included and probabilities are valid.
        Note: Probabilities do NOT need to sum to 1.0 as reactions are not mutually exclusive.
        """
        reaction_types_present = {rp.reaction for rp in v}
        all_reaction_types = set(ReactionType)
        
        # Ensure all reaction types are present (even if with 0 probability)
        if reaction_types_present != all_reaction_types:
            missing = all_reaction_types - reaction_types_present
            raise ValueError(f"Missing reaction types: {missing}")
        
        return v


# ============================================================================
# STAKEHOLDER REACTION CLASS
# ============================================================================

class StakeholderReaction:
    """
    Aggregates multiple GPTAnalysisResponse objects to provide overall stakeholder reaction metrics.
    
    Attributes:
        delta_reputation (float): Mean reputation/brand image score across all analyses
        most_likely_reaction (str): The reaction with highest mean probability
    """
    
    def __init__(self, gpt_responses: List[GPTAnalysisResponse]):
        """
        Initialize StakeholderReaction from a list of GPT analysis responses.
        
        Args:
            gpt_responses: List of GPTAnalysisResponse objects from analyzing multiple news stories
        
        Raises:
            ValueError: If gpt_responses is empty
        """
        if not gpt_responses:
            raise ValueError("At least one GPT response is required")
        
        self._responses = gpt_responses
        self._reputation_scores = [r.reputation_impact_score for r in gpt_responses]
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate all aggregate metrics from the responses."""
        # Calculate reputation metrics
        self._reputation_range = (min(self._reputation_scores), max(self._reputation_scores))
        self._mean_reputation = sum(self._reputation_scores) / len(self._reputation_scores)
        self._mode_reputation = self._calculate_mode(self._reputation_scores)
        
        # Calculate reaction probabilities
        self._mean_reaction_probabilities = self._calculate_mean_reaction_probabilities()
        self._most_likely_reaction_value = self._determine_most_likely_reaction()
    
    def _calculate_mode(self, values: List[int]) -> int:
        """
        Calculate the mode of a list of integers.
        
        If multiple modes exist, returns the one closest to the median.
        If no mode exists (all unique), returns the median.
        
        Args:
            values: List of integer values
            
        Returns:
            The mode value
        """
        from collections import Counter
        from statistics import median
        
        counter = Counter(values)
        max_count = max(counter.values())
        
        # Find all modes (values with max count)
        modes = [val for val, count in counter.items() if count == max_count]
        
        if len(modes) == 1:
            return modes[0]
        
        # Multiple modes or all unique - return closest to median
        median_val = median(values)
        return min(modes, key=lambda x: abs(x - median_val))
    
    def _calculate_mean_reaction_probabilities(self) -> Dict[ReactionType, float]:
        """
        Calculate mean probability for each reaction type across all responses.
        
        Returns:
            Dictionary mapping reaction types to mean probabilities
        """
        reaction_sums = {rt: 0.0 for rt in ReactionType}
        
        for response in self._responses:
            for rp in response.reaction_probabilities:
                reaction_sums[rp.reaction] += rp.probability
        
        num_responses = len(self._responses)
        return {
            rt: total / num_responses 
            for rt, total in reaction_sums.items()
        }
    
    def _determine_most_likely_reaction(self) -> ReactionType:
        """
        Determine the reaction with the highest mean probability.
        
        Returns:
            The ReactionType with highest mean probability
        """
        return max(
            self._mean_reaction_probabilities.items(),
            key=lambda x: x[1]
        )[0]
    
    # Public properties
    @property
    def delta_reputation(self) -> float:
        """Returns the mean reputation/brand image score."""
        return self._mean_reputation
    
    @property
    def most_likely_reaction(self) -> str:
        """Returns the most likely response/reaction as a string."""
        return self._most_likely_reaction_value.value
    
    # Additional public methods for detailed metrics
    def get_reputation_range(self) -> Tuple[int, int]:
        """Returns the range (min, max) of reputation scores."""
        return self._reputation_range
    
    def get_reputation_mode(self) -> int:
        """Returns the mode of reputation scores."""
        return self._mode_reputation
    
    def get_mean_reaction_probabilities(self) -> Dict[str, float]:
        """Returns mean probabilities for all reaction types."""
        return {
            rt.value: prob 
            for rt, prob in self._mean_reaction_probabilities.items()
        }
    
    def get_summary(self) -> Dict:
        """
        Returns a comprehensive summary of the stakeholder reaction.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        return {
            "reputation_metrics": {
                "mean": self.delta_reputation,
                "mode": self._mode_reputation,
                "range": self._reputation_range,
                "all_scores": self._reputation_scores
            },
            "reaction_metrics": {
                "most_likely": self.most_likely_reaction,
                "mean_probabilities": self.get_mean_reaction_probabilities()
            },
            "analysis_count": len(self._responses)
        }


# ============================================================================
# CACHE MANAGER FOR API RESPONSES
# ============================================================================

class APIResponseCache:
    """
    Manages caching of GPT API responses to avoid duplicate calls.
    
    Uses a hash of the prompt content as the cache key.
    """
    
    def __init__(self, cache_file: str = ".stakeholder_cache.json"):
        """
        Initialize the cache manager.
        
        Args:
            cache_file: Path to the JSON file for persistent cache storage
        """
        self.cache_file = cache_file
        self._cache: Dict[str, Dict] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file if it exists."""
        import os
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except IOError:
            pass  # Fail silently if can't save cache
    
    def _generate_key(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a unique cache key from prompts.
        
        Args:
            system_prompt: The system prompt (psychological profile)
            user_prompt: The user prompt (instructions + news story)
            
        Returns:
            SHA-256 hash of the combined prompts
        """
        combined = f"{system_prompt}|||{user_prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, system_prompt: str, user_prompt: str) -> Optional[GPTAnalysisResponse]:
        """
        Retrieve cached response if available.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            
        Returns:
            Cached GPTAnalysisResponse or None if not cached
        """
        key = self._generate_key(system_prompt, user_prompt)
        if key in self._cache:
            try:
                return GPTAnalysisResponse(**self._cache[key])
            except Exception:
                # Invalid cached data, remove it
                del self._cache[key]
                return None
        return None
    
    def set(self, system_prompt: str, user_prompt: str, response: GPTAnalysisResponse) -> None:
        """
        Cache a response.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            response: The GPTAnalysisResponse to cache
        """
        key = self._generate_key(system_prompt, user_prompt)
        self._cache[key] = response.dict()
        self._save_cache()


# ============================================================================
# MAIN SIMULATED STAKEHOLDER CLASS
# ============================================================================

class SimulatedStakeholder:
    """
    Simulates a stakeholder's reactions to news stories using psychological profiles and GPT-4o-mini.
    
    This class manages the interaction with OpenAI's API to analyze how a stakeholder
    with a specific psychological profile would react to various news stories.
    
    Attributes:
        name (str): The name of the stakeholder
        profile_path (str): Path to the psychological profile file
        
    Implementation Notes:
        - Uses instructor library for structured output from GPT
        - Implements concurrent processing with up to 10 threads
        - Handles rate limiting with exponential backoff
        - Caches API responses to avoid duplicate calls
        - Validates all inputs according to specifications
    """
    
    def __init__(self, name: str, profile_path: str):
        """
        Initialize the SimulatedStakeholder.
        
        Args:
            name: The name of the stakeholder
            profile_path: File path to text file containing psychological profile and n-shot history
            
        Raises:
            FileNotFoundError: If the profile file doesn't exist
            ValueError: If the profile file is empty or invalid
        """
        self.name = name
        self.profile_path = profile_path
        
        # Validate and load profile
        self._load_profile()
        
        # Initialize API client (will be set up in implementation)
        self._setup_api_client()
        
        # Initialize cache
        self._cache = APIResponseCache()
        
        # Logging setup
        self._setup_logging()
    
    def _load_profile(self) -> None:
        """
        Load and validate the psychological profile from file.
        
        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If profile file is empty
        """
        import os
        
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Psychological profile file not found: {self.profile_path}")
        
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            self.psychological_profile = f.read().strip()
        
        if not self.psychological_profile:
            raise ValueError(f"Psychological profile file is empty: {self.profile_path}")
    
    def _setup_api_client(self) -> None:
        """
        Set up the OpenAI API client with instructor.
        
        Loads API key from .env file.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not found in environment
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        # Client setup will be implemented with instructor
        # self.client = instructor.from_openai(OpenAI(api_key=api_key))
    
    def _setup_logging(self) -> None:
        """Set up logging for error tracking and debugging."""
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'stakeholder_{self.name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"SimulatedStakeholder.{self.name}")
    
    def reaction(self, news_stories: List[str]) -> StakeholderReaction:
        """
        Analyze stakeholder reactions to a list of news stories.
        
        This method:
        1. Validates all input news stories
        2. Sends each story to GPT-4o-mini for analysis (up to 10 concurrent)
        3. Handles rate limiting and retries
        4. Aggregates responses into a StakeholderReaction
        
        Args:
            news_stories: List of news story texts to analyze
            
        Returns:
            StakeholderReaction containing aggregated analysis results
            
        Raises:
            ValueError: If news_stories is empty, contains None, empty strings, or non-strings
            RuntimeError: If API calls fail after retries or token limit is exceeded
        """
        # Validate input
        self._validate_news_stories(news_stories)
        
        # Process stories (implementation will handle concurrency)
        gpt_responses = self._process_news_stories(news_stories)
        
        # Create and return aggregated reaction
        return StakeholderReaction(gpt_responses)
    
    def _validate_news_stories(self, news_stories: List[str]) -> None:
        """
        Validate the news stories input.
        
        Args:
            news_stories: List of news stories to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not news_stories:
            raise ValueError("News stories list must contain at least one element")
        
        for i, story in enumerate(news_stories):
            if story is None:
                raise ValueError(f"News story at index {i} is None")
            if not isinstance(story, str):
                raise ValueError(f"News story at index {i} is not a string: {type(story)}")
            if not story.strip():
                raise ValueError(f"News story at index {i} is empty")
    
    def _process_news_stories(self, news_stories: List[str]) -> List[GPTAnalysisResponse]:
        """
        Process multiple news stories with concurrent API calls.
        
        Implements:
        - Concurrent processing with up to 10 threads
        - Rate limiting handling with exponential backoff
        - Response caching
        - Error logging
        
        Args:
            news_stories: List of validated news stories
            
        Returns:
            List of GPTAnalysisResponse objects
            
        Raises:
            RuntimeError: If processing fails
        """
        # This will be implemented with asyncio/threading
        # Placeholder for specification
        responses = []
        
        # Process each story (actual implementation will be concurrent)
        for story in news_stories:
            response = self._analyze_single_story(story)
            if response:
                responses.append(response)
        
        if not responses:
            raise RuntimeError("Failed to get any valid responses from API")
        
        return responses
    
    def _analyze_single_story(self, news_story: str) -> Optional[GPTAnalysisResponse]:
        """
        Analyze a single news story with retry logic.
        
        Args:
            news_story: The news story to analyze
            
        Returns:
            GPTAnalysisResponse or None if all retries failed
        """
        # Check cache first
        user_prompt = self._create_user_prompt(news_story)
        cached = self._cache.get(self.psychological_profile, user_prompt)
        if cached:
            self.logger.info(f"Using cached response for news story")
            return cached
        
        # Implement retry logic (up to 3 attempts)
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Make API call (implementation detail)
                response = self._call_gpt_api(news_story)
                
                # Cache successful response
                self._cache.set(self.psychological_profile, user_prompt, response)
                
                return response
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"All retries failed for news story")
                    return None
    
    def _create_user_prompt(self, news_story: str) -> str:
        """
        Create the user prompt for GPT analysis.
        
        Args:
            news_story: The news story to analyze
            
        Returns:
            Formatted user prompt with instructions and news story
        """
        instructions = """
        Analyze the following news story and provide:
        
        1. A reputation/brand impact score from -5 (most negative) to +5 (most positive)
        2. Probabilities (0.0 to 1.0) for each of these potential stakeholder reactions:
           - Do Nothing
           - Share Opinion with Others
           - Switch to Competitor
           - Legal/Regulatory Action
           - Shareholder Activism
           - Devalue Shares
        
        Note: Reaction probabilities do not need to sum to 1.0 as multiple reactions may occur.
        Set probabilities to 0.0 for reactions that don't apply to this stakeholder type.
        
        News Story:
        """
        
        return f"{instructions}\n\n{news_story}"
    
    def _call_gpt_api(self, news_story: str) -> GPTAnalysisResponse:
        """
        Make the actual API call to GPT-4o-mini.
        
        This is a placeholder for the actual implementation using instructor.
        
        Args:
            news_story: The news story to analyze
            
        Returns:
            Structured GPTAnalysisResponse
            
        Raises:
            Various API-related exceptions
        """
        # This will be implemented with instructor library
        # Example structure:
        # response = self.client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": self.psychological_profile},
        #         {"role": "user", "content": self._create_user_prompt(news_story)}
        #     ],
        #     response_model=GPTAnalysisResponse
        # )
        pass


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use the SimulatedStakeholder class.
    """
    # Initialize stakeholder with profile
    stakeholder = SimulatedStakeholder(
        name="Alan Joyce",
        profile_path="profiles/alan_joyce.txt"
    )
    
    # Prepare news stories
    news_stories = [
        "Qantas announces record profits despite customer complaints...",
        "New safety concerns raised about Qantas maintenance procedures...",
        "Qantas introduces new loyalty program benefits..."
    ]
    
    # Get stakeholder reaction
    reaction = stakeholder.reaction(news_stories)
    
    # Access results
    print(f"Stakeholder: {stakeholder.name}")
    print(f"Mean Reputation Impact: {reaction.delta_reputation}")
    print(f"Most Likely Reaction: {reaction.most_likely_reaction}")
    print(f"All Reaction Probabilities: {reaction.get_mean_reaction_probabilities()}")
    print(f"Full Summary: {reaction.get_summary()}")


# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================

"""
Implementation Requirements:

1. Environment Setup:
   - Create .env file with OPENAI_API_KEY
   - Install dependencies: pip install instructor pydantic openai python-dotenv

2. Concurrent Processing:
   - Use asyncio or concurrent.futures.ThreadPoolExecutor
   - Limit to 10 concurrent API calls
   - Implement proper error handling for each thread

3. Rate Limiting:
   - Catch rate limit exceptions from OpenAI
   - Implement exponential backoff (5 seconds initial, up to 2 retries)
   - Log all rate limiting events

4. Error Handling:
   - Validate profile file existence before any processing
   - Stop execution if token limit exceeded
   - Log all API errors with timestamps
   - Provide meaningful error messages

5. Caching:
   - Cache key: hash of (psychological_profile + user_prompt)
   - Persist cache to disk for reuse across sessions
   - Validate cached responses before using

6. Stakeholder-Specific Logic:
   - Customers: Set "Devalue Shares" probability to 0
   - Investors: Set "Switch to Competitor" probability to 0
   - Employees: Context-specific probabilities

7. Testing Considerations:
   - Mock API calls for unit tests
   - Test with various stakeholder profiles from the document
   - Validate edge cases (empty lists, invalid responses)
   - Test concurrent processing limits
"""