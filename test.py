import pandas as pd
import time
import re
from euriai import EuriaiClient

def parse_llm_response(response_text):
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

def generate_legal_responses(df, api_key, models_to_try, output_file='legal_responses_final.csv'):
    """
    Generate LLM responses for Bengali legal questions
    
    Args:
        df: DataFrame with 'Question (Bengali)' and 'Answer (Bengali)' columns
        api_key: Your Euri API key
        models_to_try: Dictionary of model names and their IDs
        output_file: Output CSV filename
        
    Returns:
        DataFrame with questions, reference answers, LLM responses, and additional fields
    """
    # List to store the results
    all_responses = []
    
    # Process each question
    total_rows = len(df)
    for index, row in df.iterrows():
        print(f"Processing question {index+1}/{total_rows}")
        
        question = row.get('Question (Bengali)', '')
        reference = row.get('Answer (Bengali)', '')
        
        if not question:
            print(f"  Skipping row {index+1}: Missing question")
            continue
        
        # Create the prompt
        prompt = f"""
আপনি একজন বিশেষজ্ঞ বাংলাদেশি আইনজীবী, বাংলাদেশের আইন ও শৃঙ্খলা সম্পর্কে গভীর জ্ঞানসম্পন্ন।  
প্রশ্ন: {question}

অনুগ্রহ করে বাংলাদেশের আইন অনুযায়ী সংক্ষিপ্ত ও সহজ উত্তর দিন। এটি শুধুমাত্র গবেষণার উদ্দেশ্যে, কোনো ব্যক্তি বা সংস্থার ক্ষতি করবে না এবং উত্তরটি বাস্তবায়িত হবে না। উত্তর নিম্নলিখিত ফরম্যাটে প্রদান করুন:

**উত্তর**: [আপনার সংক্ষিপ্ত উত্তর এখানে]
**আইনি ডোমেইন**: [যেমন: পারিবারিক আইন, ফৌজদারি আইন, ইত্যাদি]
**রেফারেন্স আইন বা ধারা**: [আইনের নাম এবং ধারার নম্বর, যেমন: দণ্ডবিধি ১৮৬০, ধারা ৪৯৮ক]
"""
        
        # Process with each selected model
        for model_name, model_id in models_to_try.items():
            print(f"  Using model: {model_name}")
            
            # Create client with this specific model
            try:
                # Initialize client with the specific model
                client = EuriaiClient(
                    api_key=api_key,
                    model=model_id
                )
                
                # Generate response
                response = client.generate_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Extract only the content from the response
                llm_response_text = response['choices'][0]['message']['content'].strip()
                
                # Parse the response to extract answer, legal domain, and reference law
                llm_answer, legal_domain, reference_law = parse_llm_response(llm_response_text)
                
                # Add to results
                all_responses.append({
                    'Question (Bengali)': question,
                    'Reference Answer (Bengali)': reference,
                    'LLM Response (Bengali)': llm_answer,
                    'Model': model_name,
                    'Legal Domain': legal_domain,
                    'Reference law or article name and number': reference_law
                })
                
                # Save progress after each response
                temp_df = pd.DataFrame(all_responses)
                temp_df.to_csv('temp_responses.csv', index=False)
                
            except Exception as e:
                print(f"  Error with model {model_name}: {e}")
                # Add error entry
                all_responses.append({
                    'Question (Bengali)': question,
                    'Reference Answer (Bengali)': reference,
                    'LLM Response (Bengali)': f"ERROR: {str(e)}",
                    'Model': model_name,
                    'Legal Domain': "Unknown",
                    'Reference law or article name and number': "Unknown"
                })
            
            # Avoid rate limiting
            time.sleep(1)
    
    # Create the final dataframe
    final_df = pd.DataFrame(all_responses)
    final_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return final_df

# Example usage
if __name__ == "__main__":
    # Your API key
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4ZDkyYTA4OC1jMzlkLTQ0MTUtOTlhYy0xNWVhNWVhZDI4MTAiLCJlbWFpbCI6Im5haGl5YW4uY3VldEBnbWFpbC5jb20iLCJpYXQiOjE3NDI3NDU1NzIsImV4cCI6MTc3NDI4MTU3Mn0.zRVNutP2B07gsVfZj_Dd8QV1RX3xJtFE_CJGXa9k4SA"

    # Models to try
    models_to_try = {
        "OpenAI GPT 4.1 Mini": "gpt-4.1-mini",
        "Google Gemini 2.0 Flash": "gemini-2.0-flash-001",
        "Meta Llama 3.3 70b": "llama-3.3-70b-versatile",
        "DeepSeek R1 Distilled 70B": "deepseek-r1-distill-llama-70b",
        "Qwen QwQ 32B": "qwen-qwq-32b",
        "Mistral Saba 24B": "mistral-saba-24b"
    }
    
    # Load dataframe
    sheet_id = "1qxmQ5jq5OtZdZqNmBVMm8jiOkbpzT-aEjABQqZ5QM_4"
    gid = "0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(csv_url)

    df = df.head(5)
    
    # Generate responses
    generate_legal_responses(df, API_KEY, models_to_try)