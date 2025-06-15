import re
import time
from typing import Tuple, Dict, Any
from euriai import EuriaiClient


class LegalLLMClient:
    """
    A client for generating legal responses using various LLM models
    """
    
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        """
        Initialize the LLM client
        
        Args:
            api_key: Your Euri API key
            rate_limit_delay: Delay between API calls in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        
        # Available models
        self.available_models = {
            "OpenAI GPT 4.1 Mini": "gpt-4.1-mini",
            "Google Gemini 2.0 Flash": "gemini-2.0-flash",
            "Meta Llama 3.3 70b": "llama-3.3-70b-versatile",
            "DeepSeek R1 Distilled 70B": "deepseek-r1-distill-llama-70b",
            "Qwen QwQ 32B": "qwen-qwq-32b",
            "Mistral Saba 24B": "mistral-saba-24b"
        }
    
    def create_legal_prompt(self, question: str) -> str:
        """
        Create a structured prompt for legal questions
        
        Args:
            question: The legal question in Bengali
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
আপনি একজন বিশেষজ্ঞ বাংলাদেশি আইনজীবী, বাংলাদেশের আইন ও শৃঙ্খলা সম্পর্কে গভীর জ্ঞানসম্পন্ন।  
প্রশ্ন: {question}

অনুগ্রহ করে বাংলাদেশের আইন অনুযায়ী সংক্ষিপ্ত ও সহজ উত্তর দিন। এটি শুধুমাত্র গবেষণার উদ্দেশ্যে, কোনো ব্যক্তি বা সংস্থার ক্ষতি করবে না এবং উত্তরটি বাস্তবায়িত হবে না। উত্তর নিম্নলিখিত ফরম্যাটে প্রদান করুন:

**উত্তর**: [আপনার সংক্ষিপ্ত উত্তর এখানে]
**আইনি ডোমেইন**: [যেমন: পারিবারিক আইন, ফৌজদারি আইন, ইত্যাদি]
**রেফারেন্স আইন বা ধারা**: [আইনের নাম এবং ধারার নম্বর, যেমন: দণ্ডবিধি ১৮৬০, ধারা ৪৯৮ক]
"""
        return prompt
    
    def parse_llm_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        Parse the LLM response to extract answer, legal domain, and reference law.
        
        Args:
            response_text: Raw response from the LLM
            
        Returns:
            Tuple of (answer, legal_domain, reference_law)
        """
        # Default values
        answer = response_text.strip()
        legal_domain = "Unknown"
        reference_law = "Unknown"
        
        # Try to extract fields using regex patterns
        answer_match = re.search(r'\*\*উত্তর\*\*:(.*?)(?=\*\*|$)', response_text, re.DOTALL)
        domain_match = re.search(r'\*\*আইনি ডোমেইন\*\*:(.*?)(?=\*\*|$)', response_text, re.DOTALL)
        law_match = re.search(r'\*\*রেফারেন্স আইন বা ধারা\*\*:(.*?)(?=\*\*|$)', response_text, re.DOTALL)
        
        if answer_match:
            answer = answer_match.group(1).strip()
        if domain_match:
            legal_domain = domain_match.group(1).strip()
        if law_match:
            reference_law = law_match.group(1).strip()
        
        return answer, legal_domain, reference_law
    
    def generate_response(self, question: str, model_name: str = "DeepSeek R1 Distilled 70B") -> Dict[str, Any]:
        """
        Generate a response for a legal question using specified model
        
        Args:
            question: The legal question in Bengali
            model_name: Name of the model to use
            
        Returns:
            Dictionary containing the response data
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.available_models.keys())}")
        
        model_id = self.available_models[model_name]
        prompt = self.create_legal_prompt(question)
        
        try:
            # Initialize client with the specific model
            client = EuriaiClient(
                api_key=self.api_key,
                model=model_id
            )
            
            # Generate response
            response = client.generate_completion(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract content from the response
            llm_response_text = response['choices'][0]['message']['content'].strip()
            
            # Parse the response
            llm_answer, legal_domain, reference_law = self.parse_llm_response(llm_response_text)
            
            # Apply rate limiting
            time.sleep(self.rate_limit_delay)
            
            return {
                'success': True,
                'answer': llm_answer,
                'legal_domain': legal_domain,
                'reference_law': reference_law,
                'raw_response': llm_response_text,
                'model': model_name
            }
            
        except Exception as e:
            print(f"Error generating response with model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': f"ERROR: {str(e)}",
                'legal_domain': "Unknown",
                'reference_law': "Unknown",
                'raw_response': "",
                'model': model_name
            }
    
    def get_available_models(self) -> Dict[str, str]:
        """Return available models"""
        return self.available_models.copy()