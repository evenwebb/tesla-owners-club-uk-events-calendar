# Tesla Owners UK Events Calendar

This repository contains a Python script that scrapes upcoming events from [Tesla Owners UK](https://teslaowners.org.uk/events) and converts them into an iCalendar (`.ics`) feed plus a web page with subscribe links. The script runs via GitHub Actions and the output is served with GitHub Pages.

Running the scraper generates a `docs/` folder: `tocuk.ics` (the calendar file) and `index.html` (a page with subscribe instructions). Subscribe in Google Calendar, Apple Calendar, Outlook, or any iCalendar-compatible app.

## Features

* **Single source**: Scrapes all upcoming events from the official Tesla Owners UK events page
* **Rich event data**: Title, description, location, start/end times, event URL, and ticket booking links
* **Timezone-aware**: Events with timezone info (e.g. Giga Texas in Austin) are correctly converted to UTC
* **Standards-compliant iCal**: RFC 5545 compliant output
* **Optional reminders**: Configurable calendar notifications (day before, day of, etc.)

## Requirements

* Python 3.10 or newer
* `requests`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

From the repository root run:

```bash
python tocuk_scraper.py
```

The script will fetch the latest events, write `docs/tocuk.ics` and `docs/index.html`, and print a summary. Any errors are logged to `tocuk_log.txt`.

## GitHub Actions

The workflow (`.github/workflows/scrape_tocuk.yml`) runs the scraper weekly on Mondays at 09:00 UTC. You can also trigger it manually from the **Actions** tab.

The workflow will:
1. Install Python and dependencies
2. Run the scraper (writes `docs/tocuk.ics` and `docs/index.html`)
3. Commit and push the `docs/` folder if there are changes

**GitHub Pages**  
Set **Settings → Pages** to **Deploy from a branch**, branch **main**, folder **/docs**. The site will serve the index page and the `tocuk.ics` calendar feed.

## Configuration

### Notifications

Optional calendar reminders can be enabled in `tocuk_scraper.py`:

```python
NOTIFICATIONS = {
    "enabled": True,
    "alarms": [
        {"days_before": 1, "description": "Event tomorrow"},
        {"days_before": 0, "description": "Event today"}
    ]
}
```

## How it Works

1. **Fetch**: Requests the events page at https://teslaowners.org.uk/events
2. **Parse**: Extracts event data from the `__NEXT_DATA__` JSON embedded in the page (Next.js)
3. **Filter**: Keeps only future events (start date/time in the future)
4. **Generate**: Converts each event to an iCalendar VEVENT with description, location, URL, and correct UTC times

## Example Output

```
✓ Created docs/ with tocuk.ics (7 events) and index.html

  • Giga Texas 2026 – 16 March 2026 11:30 @ Austin, Texas
  • Annual General Meeting – 28 March 2026 10:30 @ Online
  • Everything Electric - North – 08 May 2026 10:00 @ Yorkshire Event Centre
  ...
```

## License

This project is provided as-is for personal use. Not affiliated with Tesla Owners UK Limited.
