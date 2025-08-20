# attendance.py
import os

import gspread
from pprint import pprint

# Path to your service account JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, "google-sheets-credentials.json")

# Google Sheet URL
SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1dqwB_jHAwzj5NmJdbe9SytnnqEpaDZfbGjkkwcIDtE0/edit'

# Name of the sheet/tab you want to scrape
SHEET_NAME = 'Test'

def main():
    # Authenticate with Google Sheets
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)

    # Open the spreadsheet by URL
    sh = gc.open_by_url(SPREADSHEET_URL)

    # Select the worksheet/tab
    worksheet = sh.worksheet(SHEET_NAME)

    # Get all values from the sheet
    data = worksheet.get_all_values()

    # Print the scraped data
    return data


def parse_attendance(data):
    present = []
    absent = []

    for row in data:
        for i, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue

            # Case 1: "TRUE"/"FALSE" then a name in the next cell
            if cell.upper() in ("TRUE", "FALSE"):
                if i + 1 < len(row) and row[i + 1].strip():
                    name = row[i + 1].strip()
                    if cell.upper() == "FALSE":
                        present.append(name)
                    else:
                        absent.append(name)

            # Case 2: Name followed by "TRUE"/"FALSE"
            elif i + 1 < len(row) and row[i + 1].strip().upper() in ("TRUE", "FALSE"):
                name = cell
                status = row[i + 1].strip().upper()
                if status == "FALSE":
                    present.append(name)
                else:
                    absent.append(name)

    return present, absent


if __name__ == "__main__":
    # Paste the data list you printed here
    from pprint import pprint

    data = main()

    present, absent = parse_attendance(data)

    print("✅ Present:")
    pprint(present)

    print("\n❌ Absent:")
    pprint(absent)


