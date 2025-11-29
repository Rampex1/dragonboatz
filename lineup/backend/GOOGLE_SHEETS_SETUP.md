# Google Sheets Integration Setup

The "Scrape Google Sheets Attendance" feature requires Google Sheets API credentials to work.

## Setup Instructions

1. **Create a Google Cloud Project** (if you don't have one):
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Google Sheets API**:
   - In the Google Cloud Console, go to "APIs & Services" > "Library"
   - Search for "Google Sheets API" and enable it

3. **Create a Service Account**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the service account details
   - Click "Create and Continue"
   - Skip the optional steps and click "Done"

4. **Generate Service Account Key**:
   - In the Credentials page, find your service account
   - Click on the service account email
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format and download the key file

5. **Set up the credentials file**:
   - Rename the downloaded JSON file to `google-sheets-credentials.json`
   - Place it in the `backend/` directory
   - Or copy the contents to `google-sheets-credentials.json` using the template provided

6. **Share your Google Sheet**:
   - Open your Google Sheet
   - Click "Share" button
   - Add the service account email (from the credentials file) as an editor
   - The email will look like: `your-service-account@your-project-id.iam.gserviceaccount.com`

## Current Configuration

The application is configured to work with:
- **Attendance Sheet**: `https://docs.google.com/spreadsheets/d/1dqwB_jHAwzj5NmJdbe9SytnnqEpaDZfbGjkkwcIDtE0/edit`
- **Sheet Name**: `Test`
- **Lineup Sheet**: `https://docs.google.com/spreadsheets/d/1SZMIH8hu3vMTnZfVpRkCJLnR-Jsbr8JMiNn0L1MRWoI`

Make sure your service account has access to both sheets.

## Troubleshooting

- If you get "No such file or directory" error, make sure `google-sheets-credentials.json` exists in the backend directory
- If you get permission errors, make sure the service account email is shared with your Google Sheets
- If you get API errors, make sure the Google Sheets API is enabled in your Google Cloud project
