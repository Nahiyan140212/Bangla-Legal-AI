import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict, Any, Optional
import time
import os


class GoogleSheetsHandler:
    """
    Handler for Google Sheets operations with improved error handling
    """
    
    def __init__(self, credentials_path: str = None, use_csv_export: bool = False):
        """
        Initialize the Google Sheets handler
        
        Args:
            credentials_path: Path to service account credentials JSON file
            use_csv_export: If True, uses CSV export method (no auth required)
        """
        self.use_csv_export = use_csv_export
        self.credentials_path = credentials_path or "credentials.json"
        self.gc = None
        
        # Try to setup authenticated client if not using CSV export
        if not use_csv_export:
            self._setup_authenticated_client()
    
    def _setup_authenticated_client(self):
        """Setup authenticated Google Sheets client with detailed error handling"""
        try:
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
            
            print(f"📄 Using credentials file: {self.credentials_path}")
            
            # Define the required scopes
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials
            creds = Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scope
            )
            
            # Authorize the client
            self.gc = gspread.authorize(creds)
            self.use_csv_export = False
            
            print("✅ Authenticated Google Sheets client setup successful")
            
            # Test the connection
            try:
                # Try to list spreadsheets to verify connection
                print("🔍 Testing Google Sheets API connection...")
                # This will throw an exception if authentication fails
                
            except Exception as test_error:
                print(f"⚠️ Authentication test failed: {test_error}")
                raise
                
        except FileNotFoundError as e:
            print(f"❌ Credentials file error: {e}")
            print("💡 Make sure 'credentials.json' exists in the project root")
            self.use_csv_export = True
            raise
        except Exception as e:
            print(f"❌ Failed to setup authenticated client: {e}")
            print("💡 Common issues:")
            print("   - Service account email not added to the spreadsheet")
            print("   - Incorrect permissions in credentials.json")
            print("   - Google Sheets API not enabled in Google Cloud Console")
            print("   - Invalid JSON format in credentials file")
            self.use_csv_export = True
            raise
    
    def read_sheet(self, spreadsheet_id: str, gid: str = "0", worksheet_name: str = None) -> pd.DataFrame:
        """
        Read data from Google Sheets
        
        Args:
            spreadsheet_id: The Google Sheets ID
            gid: The sheet GID (default is "0" for first sheet)
            worksheet_name: Name of the worksheet (for authenticated method)
            
        Returns:
            DataFrame with the sheet data
        """
        if self.use_csv_export:
            return self._read_sheet_csv(spreadsheet_id, gid)
        else:
            return self._read_sheet_authenticated(spreadsheet_id, worksheet_name)
    
    def _read_sheet_csv(self, spreadsheet_id: str, gid: str = "0") -> pd.DataFrame:
        """Read sheet using CSV export method (no authentication required)"""
        csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
        try:
            print(f"📥 Reading sheet via CSV export: {csv_url}")
            df = pd.read_csv(csv_url)
            print(f"✅ Successfully read {len(df)} rows from sheet")
            return df
        except Exception as e:
            print(f"❌ Error reading sheet via CSV: {e}")
            print("💡 Make sure the spreadsheet is publicly readable or use authenticated method")
            raise
    
    def _read_sheet_authenticated(self, spreadsheet_id: str, worksheet_name: str = None) -> pd.DataFrame:
        """Read sheet using authenticated method"""
        try:
            print(f"📥 Reading sheet via authenticated method: {spreadsheet_id}")
            sheet = self.gc.open_by_key(spreadsheet_id)
            
            if worksheet_name:
                worksheet = sheet.worksheet(worksheet_name)
                print(f"📋 Using worksheet: {worksheet_name}")
            else:
                worksheet = sheet.get_worksheet(0)  # First worksheet
                print(f"📋 Using first worksheet: {worksheet.title}")
            
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            print(f"✅ Successfully read {len(df)} rows from authenticated sheet")
            return df
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"❌ Spreadsheet not found: {spreadsheet_id}")
            print("💡 Make sure:")
            print("   - The spreadsheet ID is correct")
            print("   - The service account has access to the spreadsheet")
            raise
        except gspread.exceptions.WorksheetNotFound:
            print(f"❌ Worksheet not found: {worksheet_name}")
            available_sheets = [ws.title for ws in self.gc.open_by_key(spreadsheet_id).worksheets()]
            print(f"💡 Available worksheets: {available_sheets}")
            raise
        except Exception as e:
            print(f"❌ Error reading authenticated sheet: {e}")
            raise
    
    def update_cell(self, spreadsheet_id: str, worksheet_name: str, row: int, col: int, value: str):
        """
        Update a single cell in the sheet
        
        Args:
            spreadsheet_id: The Google Sheets ID
            worksheet_name: Name of the worksheet
            row: Row number (1-indexed)
            col: Column number (1-indexed)
            value: Value to insert
        """
        if self.use_csv_export:
            raise ValueError("❌ Cannot update cells without authentication. Please provide valid credentials.")
        
        if not self.gc:
            raise ValueError("❌ Google Sheets client not initialized. Check your credentials.")
        
        try:
            print(f"📝 Updating cell ({row}, {col}) in worksheet '{worksheet_name}'")
            sheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = sheet.worksheet(worksheet_name)
            worksheet.update_cell(row, col, value)
            print(f"✅ Updated cell ({row}, {col}) with value: {str(value)[:50]}...")
            time.sleep(1)  # Rate limiting to avoid API quota issues
        except gspread.exceptions.APIError as e:
            print(f"❌ Google Sheets API Error: {e}")
            if "PERMISSION_DENIED" in str(e):
                print("💡 The service account doesn't have edit permissions for this spreadsheet")
                print("   Add the service account email as an editor to the Google Sheet")
            elif "RATE_LIMIT_EXCEEDED" in str(e):
                print("💡 Rate limit exceeded. Waiting longer between requests...")
                time.sleep(5)
            raise
        except Exception as e:
            print(f"❌ Error updating cell: {e}")
            raise
    
    def update_column_by_condition(self, spreadsheet_id: str, worksheet_name: str, 
                                 df: pd.DataFrame, condition_column: str, 
                                 condition_value: str, target_column: str, 
                                 new_values: List[str]):
        """
        Update a column based on a condition
        
        Args:
            spreadsheet_id: The Google Sheets ID
            worksheet_name: Name of the worksheet
            df: DataFrame with current data
            condition_column: Column to check condition
            condition_value: Value to match in condition column
            target_column: Column to update
            new_values: List of new values to insert
        """
        if self.use_csv_export:
            raise ValueError("❌ Cannot update cells without authentication. Please provide valid credentials.")
        
        if not self.gc:
            raise ValueError("❌ Google Sheets client not initialized. Check your credentials.")
        
        try:
            print(f"🔍 Looking for rows where {condition_column} = {condition_value}")
            
            # Find rows that match the condition
            matching_rows = df[df[condition_column] == condition_value]
            
            if len(matching_rows) == 0:
                print(f"⚠️ No rows found with {condition_column} = {condition_value}")
                return
            
            print(f"📋 Found {len(matching_rows)} matching rows")
            
            # Get column index for target column
            if target_column not in df.columns:
                print(f"❌ Target column '{target_column}' not found in sheet")
                print(f"💡 Available columns: {list(df.columns)}")
                return
            
            target_col_index = df.columns.get_loc(target_column) + 1  # 1-indexed for sheets
            print(f"📊 Target column '{target_column}' is at index {target_col_index}")
            
            # Update each matching row
            sheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = sheet.worksheet(worksheet_name)
            
            for idx, (df_idx, row) in enumerate(matching_rows.iterrows()):
                if idx < len(new_values):
                    sheet_row = df_idx + 2  # +2 because sheets are 1-indexed and we have header
                    print(f"📝 Updating row {sheet_row}, column {target_col_index}")
                    worksheet.update_cell(sheet_row, target_col_index, new_values[idx])
                    print(f"✅ Updated row {sheet_row}, column {target_column} with: {str(new_values[idx])[:50]}...")
                    time.sleep(1.5)  # Increased rate limiting
                    
            print(f"✅ Successfully updated {min(len(matching_rows), len(new_values))} rows")
                    
        except Exception as e:
            print(f"❌ Error updating column: {e}")
            raise
    
    def batch_update_cells(self, spreadsheet_id: str, worksheet_name: str, updates: List[Dict]):
        """
        Batch update multiple cells at once (more efficient than individual updates)
        
        Args:
            spreadsheet_id: The Google Sheets ID
            worksheet_name: Name of the worksheet
            updates: List of dicts with 'row', 'col', 'value' keys
        """
        if self.use_csv_export:
            raise ValueError("❌ Cannot update cells without authentication.")
        
        if not self.gc:
            raise ValueError("❌ Google Sheets client not initialized.")
        
        try:
            print(f"📦 Batch updating {len(updates)} cells")
            sheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = sheet.worksheet(worksheet_name)
            
            # Prepare batch update data
            cells_to_update = []
            for update in updates:
                cell = worksheet.cell(update['row'], update['col'])
                cell.value = update['value']
                cells_to_update.append(cell)
            
            # Perform batch update
            worksheet.update_cells(cells_to_update)
            print(f"✅ Batch update completed for {len(updates)} cells")
            
        except Exception as e:
            print(f"❌ Error in batch update: {e}")
            raise
    
    def get_service_account_email(self) -> Optional[str]:
        """Get the service account email from credentials"""
        try:
            if os.path.exists(self.credentials_path):
                import json
                with open(self.credentials_path, 'r') as f:
                    creds_data = json.load(f)
                    return creds_data.get('client_email')
        except Exception as e:
            print(f"❌ Error reading service account email: {e}")
        return None
    
    def test_permissions(self, spreadsheet_id: str) -> bool:
        """Test if the service account has permissions to access the spreadsheet"""
        try:
            if self.use_csv_export:
                print("📋 Testing CSV export access...")
                df = self._read_sheet_csv(spreadsheet_id)
                return len(df) >= 0
            else:
                print("📋 Testing authenticated access...")
                sheet = self.gc.open_by_key(spreadsheet_id) 
                worksheets = sheet.worksheets()
                print(f"✅ Access confirmed. Found {len(worksheets)} worksheets")
                return True
        except Exception as e:
            print(f"❌ Permission test failed: {e}")
            return False
    
    def get_column_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of column names from DataFrame"""
        return df.columns.tolist()
    
    def find_rows_by_condition(self, df: pd.DataFrame, condition_column: str, 
                              condition_value: str) -> pd.DataFrame:
        """Find rows that match a specific condition"""
        if condition_column not in df.columns:
            print(f"❌ Column '{condition_column}' not found in DataFrame")
            print(f"💡 Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        matching_rows = df[df[condition_column] == condition_value]
        print(f"🔍 Found {len(matching_rows)} rows where {condition_column} = {condition_value}")
        return matching_rows