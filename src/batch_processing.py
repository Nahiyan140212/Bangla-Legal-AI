"""DELETE THIS FILE LATER"""
import pandas as pd
import time

class LegalQuestionProcessor:
    def __init__(self, sheets_handler, llm_client, rate_limit_delay=1.0):
        self.sheets_handler = sheets_handler
        self.llm_client = llm_client
        self.rate_limit_delay = rate_limit_delay

    def fill_model_answers(self, spreadsheet_id: str, worksheet_name: str = "Sheet1", num_rows: int = 50):
        """
        Fill answers for the first num_rows using available models if the corresponding columns are empty.
        
        Args:
            spreadsheet_id: Google Sheets ID
            worksheet_name: Name of the worksheet to update
            num_rows: Number of rows to process
        """
        target_columns = ['OpenAI 4.1 Mini', 'Gemini 2.0 Flash', 'Meta Llama 3, 70b', 'deepseek_r1']
        column_to_model = {
            'OpenAI 4.1 Mini': 'OpenAI GPT 4.1 Mini',
            'Gemini 2.0 Flash': 'Google Gemini 2.0 Flash',
            'Meta Llama 3, 70b': 'Meta Llama 3.3 70b',
            'deepseek_r1': 'DeepSeek R1 Distilled 70B'
        }
        question_column = "Question (Bengali)"
        
        print(f"Starting to fill model answers for the first {num_rows} rows in spreadsheet: {spreadsheet_id}")
        
        # Read the sheet
        try:
            df = self.sheets_handler.read_sheet(spreadsheet_id, worksheet_name=worksheet_name)
            print(f"Successfully loaded sheet with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading sheet: {e}")
            return
        
        # Take the first num_rows
        df_subset = df.head(num_rows)
        
        # Initialize list for updates
        updates = []
        
        for df_idx, row in df_subset.iterrows():
            sheet_row = df_idx + 2  # 1-based with header
            question = row.get(question_column, "")
            
            if not question or pd.isna(question):
                print(f"  Skipping row {sheet_row}: Empty question")
                continue
            
            for column in target_columns:
                if pd.isna(row[column]) or row[column] == "":
                    model_name = column_to_model[column]
                    print(f"  Generating answer for row {sheet_row}, column '{column}' using model '{model_name}'")
                    
                    # Generate response
                    response = self.llm_client.generate_response(question, model_name)
                    
                    if response['success']:
                        answer = response['answer']
                        print(f"    ✓ Generated answer: {answer[:50]}...")
                    else:
                        answer = f"ERROR: {response['error']}"
                        print(f"    ✗ Error: {response['error']}")
                    
                    # Get column index
                    col_index = df.columns.get_loc(column) + 1  # 1-based
                    
                    # Add to updates
                    updates.append({'row': sheet_row, 'col': col_index, 'value': answer})
                    
                    # Update local DataFrame
                    df.at[df_idx, column] = answer
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
        
        # Perform batch update
        if updates and not self.sheets_handler.use_csv_export:
            try:
                self.sheets_handler.batch_update_cells(spreadsheet_id, worksheet_name, updates)
                print(f"  ✓ Batch updated {len(updates)} cells")
            except Exception as e:
                print(f"  ✗ Error during batch update: {e}")
        
        # Save the first num_rows to CSV
        output_file = f"legal_questions_first_{num_rows}_{int(time.time())}.csv"
        df_subset.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Saved first {num_rows} rows to {output_file}")
        
        return {
            'updated_cells': len(updates),
            'output_file': output_file
        }

# Example usage in main.py
if __name__ == "__main__":
    from sheets_handler import GoogleSheetsHandler as SheetsHandler
    from llm_client import LegalLLMClient
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4ZDkyYTA4OC1jMzlkLTQ0MTUtOTlhYy0xNWVhNWVhZDI4MTAiLCJlbWFpbCI6Im5haGl5YW4uY3VldEBnbWFpbC5jb20iLCJpYXQiOjE3NDI3NDU1NzIsImV4cCI6MTc3NDI4MTU3Mn0.zRVNutP2B07gsVfZj_Dd8QV1RX3xJtFE_CJGXa9k4SA"
    sheets_handler = SheetsHandler(use_csv_export=False)  # Set to True if no Google Sheets auth
    llm_client = LegalLLMClient(api_key=api_key)
    processor = LegalQuestionProcessor(sheets_handler, llm_client, rate_limit_delay=1.0)
    
    SPREADSHEET_ID = "1qxmQ5jq5OtZdZqNmBVMm8jiOkbpzT-aEjABQqZ5QM_4"
    WORKSHEET_NAME = "Sheet1"
    
    results = processor.fill_model_answers(
        spreadsheet_id=SPREADSHEET_ID,
        worksheet_name=WORKSHEET_NAME,
        num_rows=35
    )
    
    if results:
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Updated {results['updated_cells']} cells")
        print(f"Results saved to: {results['output_file']}")