import pandas as pd
import asyncio
import aiohttp
import json
from typing import Dict, List, Tuple
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


class ClinicalSummaryEvaluator:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # Evaluation criteria from your rubric
        self.criteria = {
            "factual_accuracy": {
                "description": "Accuracy of medical facts and information",
                "scale": {
                    1: "Many errors/hallucinations",
                    2: "Some errors present", 
                    3: "Minor detail errors",
                    4: "Only trivial issues",
                    5: "Perfectly faithful"
                }
            },
            "completeness": {
                "description": "Coverage of critical information",
                "scale": {
                    1: "<50% critical info",
                    2: "Misses several key points",
                    3: "Minor omissions",
                    4: "Nearly complete",
                    5: "Fully complete"
                }
            },
            "conciseness": {
                "description": "Brevity and focus",
                "scale": {
                    1: "Very verbose or cryptic",
                    2: "Verbosity or compression interferes",
                    3: "Slightly wordy/terse",
                    4: "Good balance",
                    5: "Optimal length"
                }
            },
            "clarity": {
                "description": "Readability and understanding",
                "scale": {
                    1: "Hard to understand",
                    2: "Partially understandable",
                    3: "Generally clear",
                    4: "Very clear",
                    5: "Crystal-clear professional"
                }
            }
        }
        
    def create_evaluation_prompt(self, original_note: str, summary: str, summary_type: str) -> str:
        """Create evaluation prompt for Claude"""
        
        prompt = f"""You are an expert clinical evaluator. Please evaluate the following {summary_type} summary against the original clinical note using the provided rubric.

ORIGINAL CLINICAL NOTE:
{original_note}

{summary_type.upper()} SUMMARY TO EVALUATE:
{summary}

EVALUATION RUBRIC:
Please score each criterion on a scale of 1-5:

1. FACTUAL ACCURACY (1=Many errors/hallucinations, 2=Some errors present, 3=Minor detail errors, 4=Only trivial issues, 5=Perfectly faithful)
2. COMPLETENESS (1=<50% critical info, 2=Misses several key points, 3=Minor omissions, 4=Nearly complete, 5=Fully complete)
3. CONCISENESS (1=Very verbose or cryptic, 2=Verbosity or compression interferes, 3=Slightly wordy/terse, 4=Good balance, 5=Optimal length)
4. CLARITY (1=Hard to understand, 2=Partially understandable, 3=Generally clear, 4=Very clear, 5=Crystal-clear professional)

Please provide your evaluation in the following JSON format:
{{
    "factual_accuracy": {{
        "score": [1-5],
        "reasoning": "Brief explanation of score"
    }},
    "completeness": {{
        "score": [1-5],
        "reasoning": "Brief explanation of score"
    }},
    "conciseness": {{
        "score": [1-5],
        "reasoning": "Brief explanation of score"
    }},
    "clarity": {{
        "score": [1-5],
        "reasoning": "Brief explanation of score"
    }},
    "overall_score": [average of all scores],
    "key_strengths": ["strength1", "strength2"],
    "key_weaknesses": ["weakness1", "weakness2"],
    "clinical_appropriateness": "Assessment of clinical relevance and appropriateness"
}}

Focus on clinical accuracy, completeness of critical information, and appropriateness for the target audience ({summary_type}).
"""
        return prompt

    async def evaluate_summary(self, session: aiohttp.ClientSession, original_note: str, 
                             summary: str, summary_type: str) -> Dict:
        """Evaluate a single summary using Claude API"""
        
        prompt = self.create_evaluation_prompt(original_note, summary, summary_type)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1  # Low temperature for consistent evaluation
        }
        
        try:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['content'][0]['text']
                    
                    # Extract JSON from response
                    try:
                        # Find JSON in the response
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        json_str = content[json_start:json_end]
                        evaluation = json.loads(json_str)
                        return evaluation
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON from response: {content}")
                        return None
                else:
                    print(f"API request failed with status {response.status}")
                    return None
                    
        except Exception as e:
            print(f"Error evaluating summary: {e}")
            return None

    async def evaluate_all_summaries(self, df: pd.DataFrame, max_concurrent: int = 5) -> pd.DataFrame:
        """Evaluate all summaries in the dataset"""
        
        # Define the summary columns to evaluate
        summary_columns = [
            ('llama_3_8b_physician_summary', 'physician'),
            ('llama_3_8b_patient_summary', 'patient'),
            ('titan_text_express_physician_summary', 'physician'),
            ('titan_text_express_patient_summary', 'patient'),
            ('claude_haiku_physician_summary', 'physician'),
            ('claude_haiku_patient_summary', 'patient')
        ]
        
        results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for idx, row in df.iterrows():
                original_note = row['cleaned_text']  # Use cleaned_text as the reference
                
                for summary_col, summary_type in summary_columns:
                    if pd.notna(row[summary_col]) and row[summary_col].strip():
                        task = self.evaluate_with_semaphore(
                            semaphore, session, original_note, 
                            row[summary_col], summary_type, 
                            row['note_id'], summary_col
                        )
                        tasks.append(task)
            
            # Execute all evaluations
            print(f"Starting evaluation of {len(tasks)} summaries...")
            evaluations = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for eval_result in evaluations:
                if isinstance(eval_result, Exception):
                    print(f"Evaluation failed: {eval_result}")
                elif eval_result is not None:
                    results.append(eval_result)
        
        # Convert results to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return pd.DataFrame()

    async def evaluate_with_semaphore(self, semaphore: asyncio.Semaphore, session: aiohttp.ClientSession,
                                    original_note: str, summary: str, summary_type: str, 
                                    note_id: str, model_name: str) -> Dict:
        """Evaluate with concurrency control"""
        
        async with semaphore:
            evaluation = await self.evaluate_summary(session, original_note, summary, summary_type)
            
            if evaluation:
                # Add metadata to the evaluation
                evaluation['note_id'] = note_id
                evaluation['model'] = model_name
                evaluation['summary_type'] = summary_type
                evaluation['timestamp'] = datetime.now().isoformat()
                
                return evaluation
            
            return None

    def generate_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive evaluation report"""
        
        if results_df.empty:
            return {"error": "No evaluation results available"}
        
        report = {
            "overall_statistics": {},
            "model_comparison": {},
            "summary_type_comparison": {},
            "detailed_analysis": {}
        }
        
        # Overall statistics
        criteria_cols = ['factual_accuracy', 'completeness', 'conciseness', 'clarity']
        
        for criterion in criteria_cols:
            scores = [row[criterion]['score'] for _, row in results_df.iterrows() if criterion in row]
            if scores:
                report["overall_statistics"][criterion] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Model comparison
        models = results_df['model'].unique()
        for model in models:
            model_data = results_df[results_df['model'] == model]
            model_scores = {}
            
            for criterion in criteria_cols:
                scores = [row[criterion]['score'] for _, row in model_data.iterrows() if criterion in row]
                if scores:
                    model_scores[criterion] = sum(scores) / len(scores)
            
            if model_scores:
                model_scores['overall_average'] = sum(model_scores.values()) / len(model_scores)
                report["model_comparison"][model] = model_scores
        
        # Summary type comparison (physician vs patient)
        summary_types = results_df['summary_type'].unique()
        for summary_type in summary_types:
            type_data = results_df[results_df['summary_type'] == summary_type]
            type_scores = {}
            
            for criterion in criteria_cols:
                scores = [row[criterion]['score'] for _, row in type_data.iterrows() if criterion in row]
                if scores:
                    type_scores[criterion] = sum(scores) / len(scores)
            
            if type_scores:
                type_scores['overall_average'] = sum(type_scores.values()) / len(type_scores)
                report["summary_type_comparison"][summary_type] = type_scores
        
        return report

    def save_results(self, results_df: pd.DataFrame, report: Dict, 
                    results_file: str = "evaluation_results.csv", 
                    report_file: str = "evaluation_report.json"):
        """Save evaluation results and report"""
        
        # Save detailed results
        if not results_df.empty:
            # Flatten the nested JSON structure for CSV
            flattened_results = []
            for _, row in results_df.iterrows():
                flat_row = {
                    'note_id': row['note_id'],
                    'model': row['model'],
                    'summary_type': row['summary_type'],
                    'timestamp': row['timestamp'],
                    'overall_score': row.get('overall_score', 0)
                }
                
                # Flatten criteria scores
                criteria_cols = ['factual_accuracy', 'completeness', 'conciseness', 'clarity']
                for criterion in criteria_cols:
                    if criterion in row:
                        flat_row[f'{criterion}_score'] = row[criterion]['score']
                        flat_row[f'{criterion}_reasoning'] = row[criterion]['reasoning']
                
                # Add other fields
                flat_row['key_strengths'] = str(row.get('key_strengths', []))
                flat_row['key_weaknesses'] = str(row.get('key_weaknesses', []))
                flat_row['clinical_appropriateness'] = row.get('clinical_appropriateness', '')
                
                flattened_results.append(flat_row)
            
            flattened_df = pd.DataFrame(flattened_results)
            flattened_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_file}")

# Example usage function
async def main():
    # Initialize evaluator
    evaluator = ClinicalSummaryEvaluator(
        api_key=CLAUDE_API_KEY,
        model="claude-sonnet-4-20250514"
    )
    
    # Load your CSV file
    df = pd.read_csv("clinical_notes_with_summaries.csv")
    
    # Optional: Evaluate a subset for testing
    df = df.head(1)  
    
    # Run evaluation
    results_df = await evaluator.evaluate_all_summaries(df, max_concurrent=3)
    
    # Generate report
    report = evaluator.generate_report(results_df)
    
    # Save results
    evaluator.save_results(results_df, report)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total evaluations completed: {len(results_df)}")
    
    if "model_comparison" in report:
        print("\nModel Rankings (by overall average):")
        model_scores = [(model, scores.get('overall_average', 0)) 
                       for model, scores in report["model_comparison"].items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, score) in enumerate(model_scores, 1):
            print(f"{i}. {model}: {score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())