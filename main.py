import pandas as pd
import time
from typing import List, Dict, Any
from src.sheets_handler import GoogleSheetsHandler
from src.llm_client import LegalLLMClient

class LegalQuestionProcessor:
    """
    Main processor for handling legal questions and sheet updates
    """
    
    def __init__(self, api_key: str, credentials_path: str = None, rate_limit_delay: float = 2.0):
        """
        Initialize the processor
        
        Args:
            api_key: Your Euri API key
            credentials_path: Path to Google Sheets service account credentials
            rate_limit_delay: Delay between API calls in seconds
        """
        self.llm_client = LegalLLMClient(api_key, rate_limit_delay)
        self.sheets_handler = GoogleSheetsHandler(credentials_path)
        self.rate_limit_delay = rate_limit_delay
    
    def process_questions_by_condition(self, spreadsheet_id: str, 
                                     question_column: str = "Question (Bengali)",
                                     condition_column: str = "Currently Evaluating",
                                     condition_value: str = "ishmam",
                                     target_column: str = "deepseek_r1",
                                     model_name: str = "OpenAI GPT 4.1 Mini",
                                     worksheet_name: str = "Sheet1",
                                     max_questions: int = None):
        """
        Process questions that match a specific condition and update the target column
        
        Args:
            spreadsheet_id: Google Sheets ID
            question_column: Column containing questions
            condition_column: Column to check for condition
            condition_value: Value to match in condition column
            target_column: Column to update with answers
            model_name: LLM model to use
            worksheet_name: Name of the worksheet to update
            max_questions: Maximum number of questions to process (None for all)
        """
        print(f"Starting processing for spreadsheet: {spreadsheet_id}")
        print(f"Looking for rows where {condition_column} = {condition_value}")
        print(f"Will update {target_column} column with answers from {model_name}")
        
        # Read the current sheet data
        try:
            df = self.sheets_handler.read_sheet(spreadsheet_id)
            print(f"Successfully loaded sheet with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading sheet: {e}")
            return
        
        # Validate required columns exist
        required_columns = [question_column, condition_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
        
        # Find rows that match the condition
        matching_rows = self.sheets_handler.find_rows_by_condition(
            df, condition_column, condition_value
        )
        
        if len(matching_rows) == 0:
            print("No matching rows found. Exiting.")
            return
        
        # Limit the number of questions if specified
        if max_questions and len(matching_rows) > max_questions:
            matching_rows = matching_rows.head(max_questions)
            print(f"Limited to processing {max_questions} questions")
        
        # Check if target column exists
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found in sheet")
            print("Available columns:", df.columns.tolist())
            print("Proceeding anyway - you may need to add the column to your sheet")
        
        # Process each matching question
        answers = []
        processed_count = 0
        
        for idx, (df_idx, row) in enumerate(matching_rows.iterrows()):
            question = row.get(question_column, "")
            
            if not question or pd.isna(question):
                print(f"  Skipping row {df_idx + 2}: Empty question")
                answers.append("")
                continue
            
            print(f"\nProcessing question {idx + 1}/{len(matching_rows)} (Sheet row {df_idx + 2})")
            print(f"  Question: {question[:100]}...")
            
            # Generate response using LLM
            response = self.llm_client.generate_response(question, model_name)
            
            if response['success']:
                answer = response['answer']
                answers.append(answer)
                processed_count += 1
                print(f"  ✓ Generated answer: {answer[:100]}...")
                
                # Optional: Update sheet immediately after each response
                if not self.sheets_handler.use_csv_export:
                    try:
                        sheet_row = df_idx + 2  # +2 for 1-indexing and header
                        target_col_index = df.columns.get_loc(target_column) + 1 if target_column in df.columns else len(df.columns) + 1
                        
                        self.sheets_handler.update_cell(
                            spreadsheet_id, worksheet_name, 
                            sheet_row, target_col_index, answer
                        )
                        print(f"  ✓ Updated sheet row {sheet_row}")
                    except Exception as e:
                        print(f"  ✗ Error updating sheet: {e}")
                
            else:
                error_msg = f"ERROR: {response['error']}"
                answers.append(error_msg)
                print(f"  ✗ Error: {response['error']}")
            
            # Rate limiting
            if idx < len(matching_rows) - 1:  # Don't sleep after the last item
                print(f"  Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {processed_count}/{len(matching_rows)} questions")
        
        # If we're using CSV export method, print the results for manual updating
        if self.sheets_handler.use_csv_export:
            print("\n=== Results (for manual sheet update) ===")
            for idx, (df_idx, row) in enumerate(matching_rows.iterrows()):
                print(f"Row {df_idx + 2}: {answers[idx]}")
        
        return {
            'processed_count': processed_count,
            'total_questions': len(matching_rows),
            'answers': answers,
            'matching_rows': matching_rows
        }
    
    def preview_questions(self, spreadsheet_id: str, 
                         question_column: str = "Question (Bengali)",
                         condition_column: str = "Currently Evaluating",
                         condition_value: str = "ishmam",
                         max_preview: int = 5):
        """
        Preview questions that would be processed
        """
        print(f"Previewing questions from spreadsheet: {spreadsheet_id}")
        
        try:
            df = self.sheets_handler.read_sheet(spreadsheet_id)
            matching_rows = self.sheets_handler.find_rows_by_condition(
                df, condition_column, condition_value
            )
            
            if len(matching_rows) == 0:
                print("No matching rows found.")
                return
            
            print(f"\nPreview of first {min(max_preview, len(matching_rows))} questions:")
            print("=" * 80)
            
            for idx, (df_idx, row) in enumerate(matching_rows.head(max_preview).iterrows()):
                question = row.get(question_column, "")
                print(f"\nRow {df_idx + 2}: {question}")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error previewing questions: {e}")


def main():
    """Main execution function"""
    
    # Configuration
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4ZDkyYTA4OC1jMzlkLTQ0MTUtOTlhYy0xNWVhNWVhZDI4MTAiLCJlbWFpbCI6Im5haGl5YW4uY3VldEBnbWFpbC5jb20iLCJpYXQiOjE3NDI3NDU1NzIsImV4cCI6MTc3NDI4MTU3Mn0.zRVNutP2B07gsVfZj_Dd8QV1RX3xJtFE_CJGXa9k4SA"
    SPREADSHEET_ID = "1qxmQ5jq5OtZdZqNmBVMm8jiOkbpzT-aEjABQqZ5QM_4"
    
    # Optional: Path to Google Sheets service account credentials
    # CREDENTIALS_PATH = "path/to/your/service-account-key.json"
    CREDENTIALS_PATH = "credentials.json"  # Set to None to use CSV export method
    
    # Processing parameters
    QUESTION_COLUMN = "Question (Bengali)"
    CONDITION_COLUMN = "Currently Evaluating"
    CONDITION_VALUE = "ishmam"
    TARGET_COLUMN = "Gemini 2.0 Flash"
    MODEL_NAME = 'Google Gemini 2.0 Flash'
    WORKSHEET_NAME = "Sheet1"  # Adjust if needed
    RATE_LIMIT_DELAY = 2.0  # seconds between API calls
    MAX_QUESTIONS = 10  # Set to None to process all matching questions
    
    # Initialize processor
    processor = LegalQuestionProcessor(
        api_key=API_KEY,
        credentials_path=CREDENTIALS_PATH,
        rate_limit_delay=RATE_LIMIT_DELAY
    )
    
    # Preview questions first (optional)
    print("=== PREVIEW MODE ===")
    processor.preview_questions(
        spreadsheet_id=SPREADSHEET_ID,
        question_column=QUESTION_COLUMN,
        condition_column=CONDITION_COLUMN,
        condition_value=CONDITION_VALUE
    )
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with processing? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        return
    
    # Process questions
    print("\n=== PROCESSING MODE ===")
    results = processor.process_questions_by_condition(
        spreadsheet_id=SPREADSHEET_ID,
        question_column=QUESTION_COLUMN,
        condition_column=CONDITION_COLUMN,
        condition_value=CONDITION_VALUE,
        target_column=TARGET_COLUMN,
        model_name=MODEL_NAME,
        worksheet_name=WORKSHEET_NAME,
        max_questions=MAX_QUESTIONS
    )
    
    if results:
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Processed: {results['processed_count']}/{results['total_questions']} questions")
        
        if processor.sheets_handler.use_csv_export:
            print(f"\nNote: Using CSV export method. Sheet was not automatically updated.")
            print(f"Please manually update the '{TARGET_COLUMN}' column with the results shown above.")


if __name__ == "__main__":
    main()