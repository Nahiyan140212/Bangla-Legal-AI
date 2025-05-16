# Bangla-Legal-AI
Bengali Legal Response Generator
Overview
This project is a Python script designed to generate legal responses in Bengali for questions sourced from a Google Sheets CSV file. The script leverages the EuriaiClient API to interact with large language models (LLMs) and produces structured responses based on Bangladeshi law. The responses are intended solely for research purposes, will not be implemented, and are not meant to cause harm to any person or entity.
The script processes a dataset of legal questions, generates responses using specified LLMs, and saves the output to a CSV file with the following columns:

Question (Bengali)
Reference Answer (Bengali)
LLM Response (Bengali)
Model
Legal Domain
Reference law or article name and number

The Legal Domain and Reference law or article name and number are extracted from the LLM's structured response.
Features

Fetches questions from a Google Sheets CSV export.
Generates concise legal responses in Bengali, grounded in Bangladeshi law.
Supports multiple LLM models (e.g., Qwen QwQ 32B, Mistral Saba 24B).
Saves intermediate progress to temp_responses.csv and final output to legal_responses_final.csv.
Handles errors gracefully, logging issues without stopping execution.
Ensures responses include legal domain and reference law, as specified by the prompt.

Prerequisites

Python: Version 3.8 or higher.
Dependencies:
pandas: For handling CSV data and DataFrames.
euriai: Custom library for interacting with the Euriai API (assumed proprietary or custom-built).
re: Built-in Python library for regular expressions (used for response parsing).


API Key: A valid Euriai API key for accessing the LLMs.
Google Sheets Access: A publicly accessible Google Sheet with legal questions and reference answers in Bengali.

Installation

Clone the Repository (or download the script):git clone <repository-url>
cd <repository-directory>


Install Dependencies:pip install pandas

Note: The euriai library is assumed to be proprietary. Ensure it is installed or available in your environment. Contact the API provider for access if needed.
Set Up the Google Sheet:
Ensure the Google Sheet is publicly accessible or shared appropriately.
Update the sheet_id and gid in the script to match your Google Sheet's ID and tab ID.



Usage

Prepare the Input Data:

The Google Sheet should have at least two columns: Question (Bengali) and Answer (Bengali).
The script fetches the data from the CSV export URL: https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}.


Configure the Script:

Update the API_KEY in the script with your Euriai API key.
Modify the models_to_try dictionary if you want to use different LLMs supported by the Euriai API.
Example configuration in the script:API_KEY = "your-api-key-here"
models_to_try = {
    "OpenAI GPT 4.1 Mini": "gpt-4.1-mini",
    "Google Gemini 2.0 Flash": "gemini-2.0-flash-001",
    "Meta Llama 3.3 70b": "llama-3.3-70b-versatile",
    "DeepSeek R1 Distilled 70B": "deepseek-r1-distill-llama-70b",
    "Qwen QwQ 32B": "qwen-qwq-32b",
    "Mistral Saba 24B": "mistral-saba-24b"
}
sheet_id = "1qxmQ5jq5OtZdZqNmBVMm8jiOkbpzT-aEjABQqZ5QM_4"
gid = "0"




Run the Script:
python generate_legal_responses.py


The script will process each question, query the LLMs, and save results to temp_responses.csv (for progress) and legal_responses_final.csv (final output).
Progress and errors are printed to the console.



Output

Temporary File: temp_responses.csv is updated after each response to save progress.

Final File: legal_responses_final.csv contains the complete results with the following columns:

Question (Bengali): The input legal question.
Reference Answer (Bengali): The reference answer from the Google Sheet.
LLM Response (Bengali): The LLM-generated answer.
Model: The LLM model used (e.g., Qwen QwQ 32B).
Legal Domain: The legal domain (e.g., পারিবারিক আইন, ফৌজদারি আইন), extracted from the LLM response.
Reference law or article name and number: The cited law or article (e.g., দণ্ডবিধি ১৮৬০, ধারা ৪৯৮ক), extracted from the LLM response.


Example Row:



Question (Bengali)
Reference Answer (Bengali)
LLM Response (Bengali)
Model
Legal Domain
Reference law or article name and number



পিতার সম্পত্তিতে সন্তানের অধিকার কী?
সন্তানের অধিকার আছে...
পিতার সম্পত্তিতে সন্তানের অধিকার রয়েছে...
Qwen QwQ 32B
উত্তরাধিকার আইন
মুসলিম পারিবারিক আইন অধ্যাদেশ ১৯৬১




Notes

API Dependency: The script relies on the EuriaiClient library, which is assumed to be a custom or proprietary API client. Ensure you have access to this library and a valid API key.
Rate Limiting: The script includes a time.sleep(1) to avoid API rate limits. Adjust this value based on the API provider's guidelines.
LLM Response Format: The script expects the LLM to return responses in a structured format with sections labeled **উত্তর**, **আইনি ডোমেইন**, and **রেফারেন্স আইন বা ধারা**. If the LLM deviates, the parsing logic may need adjustment.
Error Handling: Errors (e.g., API failures) are logged in the CSV with "ERROR: {message}" in the LLM Response (Bengali) column and "Unknown" in the Legal Domain and Reference law or article name and number columns.
Limitations:
The accuracy of legal responses depends on the LLM's knowledge of Bangladeshi law.
The script assumes the Google Sheet is accessible and correctly formatted.
Responses are for research only and should not be used for legal advice or implementation.



Troubleshooting

API Errors: Check your API key and network connection. Ensure the euriai library is correctly installed.
Missing Fields: If Legal Domain or Reference law or article name and number are "Unknown," the LLM may not have followed the structured format. Inspect the raw LLM response and adjust the parse_llm_response function if needed.
Google Sheet Access: If the CSV download fails, verify the sheet_id and gid, and ensure the sheet is publicly accessible or shared appropriately.

Contributing
This project is for research purposes and not intended for production use. Contributions are welcome for improving the parsing logic, error handling, or documentation. Please submit a pull request or open an issue to discuss changes.
License
This project is provided as-is for research purposes. No license is specified, as the code is intended for academic use only. Ensure compliance with the Euriai API's terms of service when using the script.
