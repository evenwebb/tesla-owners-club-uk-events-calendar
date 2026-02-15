"""Tesla Owners Club UK Events Calendar Scraper.

Scrapes upcoming events from Tesla Owners UK (https://teslaowners.org.uk/events)
and generates an iCalendar file. Fetches each event's detail page for richer
descriptions, map coordinates, and additional metadata.
"""
import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ============================================================================
# CONSTANTS
# ============================================================================
EVENTS_URL = "https://teslaowners.org.uk/events"
BASE_URL = "https://teslaowners.org.uk"
HTTP_TIMEOUT = 60
HTTP_RETRIES = 3
HTTP_RETRY_DELAY = 1
HTTP_RETRY_MULTIPLIER = 2
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)

ICAL_LINE_LENGTH = 75
OUTPUT_DIR = "docs"
LOG_FILE = "tocuk_log.txt"

# Cache event details to avoid re-scraping unchanged data
CACHE_FILE = ".tocuk_event_cache.json"
CACHE_EXPIRY_DAYS = 7
FETCH_DELAY_SEC = 0.5  # Delay between detail page requests

# State file to detect new events (skip full scrape when no new events)
STATE_FILE = "docs/.last_upcoming.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
error_handler = logging.FileHandler(LOG_FILE)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)


# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================
NOTIFICATION_TIME = "09:00"
NOTIFICATIONS = {
    "enabled": False,
    "alarms": [],
}


def fetch_with_retries(
    url: str,
    retries: int = HTTP_RETRIES,
    timeout: int = HTTP_TIMEOUT,
) -> requests.Response:
    """Return HTTP response, retrying with exponential backoff on errors."""
    headers = {"User-Agent": USER_AGENT}
    delay = HTTP_RETRY_DELAY

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= HTTP_RETRY_MULTIPLIER

    raise requests.RequestException("All retries exhausted")


def extract_events_from_page(html: str) -> List[Dict[str, Any]]:
    """Extract events from __NEXT_DATA__ JSON embedded in the page.

    Returns:
        List of event dicts with title, start_at, end_at, location, description, url, slug.
    """
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        logger.error("Could not find __NEXT_DATA__ in page")
        return []

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error("Failed to parse __NEXT_DATA__ JSON: %s", e)
        return []

    events_data = data.get("props", {}).get("pageProps", {}).get("events", {})
    upcoming = events_data.get("upcoming", [])
    past = events_data.get("past", [])
    # Merge and deduplicate by slug (upcoming takes precedence)
    seen = set()
    merged = []
    for e in upcoming + past:
        slug = e.get("slug")
        if slug and slug not in seen:
            seen.add(slug)
            merged.append(e)
        elif not slug:
            merged.append(e)
    return merged


def load_cache() -> Dict[str, dict]:
    """Load event detail cache from disk. Drops expired entries."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        cutoff = (
            datetime.datetime.now() - datetime.timedelta(days=CACHE_EXPIRY_DAYS)
        ).isoformat()
        cache = {k: v for k, v in cache.items() if v.get("cached_at", "") > cutoff}
        logger.info("Loaded event cache with %d entries", len(cache))
        return cache
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache load failed: %s", e)
        return {}


def save_cache(cache: Dict[str, dict]) -> None:
    """Save event detail cache to disk."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info("Saved event cache with %d entries", len(cache))
    except OSError as e:
        logger.warning("Cache save failed: %s", e)


def load_last_upcoming_slugs() -> set:
    """Load the set of upcoming event slugs from last run. Empty if no state file."""
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        slugs = data.get("slugs", [])
        return set(slugs) if isinstance(slugs, list) else set()
    except (json.JSONDecodeError, OSError):
        return set()


def save_last_upcoming_slugs(slugs: List[str]) -> None:
    """Save the current upcoming event slugs for next run comparison."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"slugs": slugs, "updated": datetime.datetime.now().isoformat()},
                f,
                indent=2,
            )
        logger.info("Saved state with %d upcoming slugs", len(slugs))
    except OSError as e:
        logger.warning("State save failed: %s", e)


def has_new_events(current_slugs: set, previous_slugs: set) -> bool:
    """True if there is at least one event in current that was not in previous.

    Events moving from upcoming to past (in previous but not current) do NOT count.
    """
    return bool(current_slugs - previous_slugs)


def extract_event_from_detail_page(html: str) -> Optional[Dict[str, Any]]:
    """Extract full event data from __NEXT_DATA__ on an event detail page.

    Returns:
        Event dict from pageProps.event, or None if not found.
    """
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        event = data.get("props", {}).get("pageProps", {}).get("event")
        return event if isinstance(event, dict) else None
    except json.JSONDecodeError:
        return None


def fetch_event_detail(slug: str, cache: Dict[str, dict]) -> Optional[Dict[str, Any]]:
    """Fetch event detail from its page. Uses cache if available and not expired.

    Args:
        slug: Event slug (e.g. 'giga-texas-2026')
        cache: Cache dict to read/write

    Returns:
        Merged event dict with detail page data, or None on failure.
    """
    if not slug:
        return None

    if slug in cache:
        entry = {k: v for k, v in cache[slug].items() if k != "cached_at"}
        logger.info("Using cached detail for: %s", slug)
        return entry

    url = f"{BASE_URL}/events/{slug}"
    try:
        time.sleep(FETCH_DELAY_SEC)
        response = fetch_with_retries(url)
        event = extract_event_from_detail_page(response.text)
        if event:
            cache[slug] = {**event, "cached_at": datetime.datetime.now().isoformat()}
            logger.info("Fetched detail for: %s", slug)
            return event
    except requests.RequestException as e:
        logger.warning("Failed to fetch event detail %s: %s", slug, e)
    return None


def merge_event_detail(
    list_event: Dict[str, Any], detail_event: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge detail page data into list event. Detail overrides where richer."""
    if not detail_event:
        return list_event
    merged = dict(list_event)
    # Prefer detail page for description (often fuller)
    if detail_event.get("description"):
        merged["description"] = detail_event["description"]
    # Prefer detail location if list is empty
    if detail_event.get("location") and not merged.get("location"):
        merged["location"] = detail_event["location"]
    # Map coordinates and link
    if detail_event.get("map_latitude") is not None:
        merged["map_latitude"] = detail_event["map_latitude"]
    if detail_event.get("map_longitude") is not None:
        merged["map_longitude"] = detail_event["map_longitude"]
    if detail_event.get("map_link"):
        merged["map_link"] = detail_event["map_link"]
    # Additional info
    if detail_event.get("additional_info"):
        merged["additional_info"] = detail_event["additional_info"]
    # Metadata for iCal
    if detail_event.get("id") is not None:
        merged["id"] = detail_event["id"]
    if detail_event.get("created_at"):
        merged["created_at"] = detail_event["created_at"]
    if detail_event.get("updated_at"):
        merged["updated_at"] = detail_event["updated_at"]
    if detail_event.get("timezone"):
        merged["timezone"] = detail_event["timezone"]
    if detail_event.get("date_or_range"):
        merged["date_or_range"] = detail_event["date_or_range"]
    if detail_event.get("banner_url"):
        merged["banner_url"] = detail_event["banner_url"]
    if detail_event.get("email_address"):
        merged["email_address"] = detail_event["email_address"]
    if detail_event.get("reply_to_email"):
        merged["reply_to_email"] = detail_event["reply_to_email"]
    if detail_event.get("ticket_success_message"):
        merged["ticket_success_message"] = detail_event["ticket_success_message"]
    if detail_event.get("registrations_count") is not None:
        merged["registrations_count"] = detail_event["registrations_count"]
    if detail_event.get("tickets_count") is not None:
        merged["tickets_count"] = detail_event["tickets_count"]
    if detail_event.get("releases"):
        merged["releases"] = detail_event["releases"]
    if detail_event.get("homepage_url"):
        merged["homepage_url"] = detail_event["homepage_url"]
    return merged


def strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    if not html:
        return ""
    # Remove markdown-style links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", html)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_iso_datetime(iso_str: str) -> Optional[datetime.datetime]:
    """Parse ISO 8601 datetime string to datetime (timezone-aware or naive)."""
    if not iso_str:
        return None
    try:
        # Handle formats like 2026-03-16T11:30:00.000-06:00 or 2026-03-28T10:30:00.000Z
        dt = datetime.datetime.fromisoformat(
            iso_str.replace("Z", "+00:00").replace(".000", "")
        )
        return dt
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse datetime %r: %s", iso_str, e)
        return None


def escape_and_fold_ical_text(text: str, prefix: str = "") -> str:
    """Escape and fold text for iCalendar format per RFC 5545."""
    escaped = text.replace("\\", "\\\\").replace("\n", "\\n")
    full_line = prefix + escaped

    if len(full_line) <= ICAL_LINE_LENGTH:
        return full_line

    result = [full_line[:ICAL_LINE_LENGTH]]
    remaining = full_line[ICAL_LINE_LENGTH:]
    while remaining:
        result.append(" " + remaining[: ICAL_LINE_LENGTH - 1])
        remaining = remaining[ICAL_LINE_LENGTH - 1 :]
    return "\n".join(result)


def generate_alarm(alarm_config: Dict[str, Any], event_start: datetime.datetime) -> str:
    """Generate a VALARM component for iCalendar using RFC 5545 duration format."""
    # Use RFC 5545 duration format for TRIGGER (e.g., -P1D = 1 day before)
    days = alarm_config.get("days_before", 1)

    # Convert to ISO 8601 duration format
    if days == 0:
        # On event day - use hours before if time is specified
        time_str = alarm_config.get("time", NOTIFICATION_TIME)
        time_parts = time_str.split(":")
        hours = int(time_parts[0])
        minutes = int(time_parts[1]) if len(time_parts) > 1 else 0

        # Calculate hours before event
        event_hour = event_start.hour if hasattr(event_start, 'hour') else 0
        event_minute = event_start.minute if hasattr(event_start, 'minute') else 0

        hours_before = event_hour - hours
        minutes_before = event_minute - minutes

        if hours_before < 0 or (hours_before == 0 and minutes_before <= 0):
            # Time is same or after event, use 1 hour before as default
            trigger_line = "TRIGGER:-PT1H"
        else:
            total_minutes = hours_before * 60 + minutes_before
            if total_minutes >= 60:
                trigger_line = f"TRIGGER:-PT{total_minutes // 60}H"
            else:
                trigger_line = f"TRIGGER:-PT{total_minutes}M"
    else:
        # Days before event
        trigger_line = f"TRIGGER:-P{days}D"

    description = alarm_config.get("description", "Event Reminder")
    return (
        "BEGIN:VALARM\n"
        "ACTION:DISPLAY\n"
        f"DESCRIPTION:{description}\n"
        f"{trigger_line}\n"
        "END:VALARM\n"
    )


def _format_ical_datetime(dt: datetime.datetime) -> str:
    """Format datetime for iCal (UTC with Z suffix)."""
    if dt.tzinfo:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def make_ics_event(event: Dict[str, Any]) -> str:
    """Return an iCalendar VEVENT string for a Tesla Owners UK event.

    Populates all applicable RFC 5545 VEVENT properties from scraped data.
    """
    title = event.get("title", "Untitled Event")
    start_at = event.get("start_at")
    end_at = event.get("end_at")
    location = event.get("location", "")
    description_raw = event.get("description", "")
    slug = event.get("slug", "")
    ti_to_url = event.get("url", "")
    event_id = event.get("id")
    created_at = event.get("created_at")
    updated_at = event.get("updated_at")
    timezone_name = event.get("timezone", "")
    date_or_range = event.get("date_or_range", "")
    banner_url = event.get("banner_url", "")
    email_address = event.get("email_address", "")
    reply_to_email = event.get("reply_to_email", "")
    ticket_success_message = event.get("ticket_success_message", "")
    registrations_count = event.get("registrations_count")
    tickets_count = event.get("tickets_count")
    releases = event.get("releases") or []

    event_url = f"{BASE_URL}/events/{slug}" if slug else ti_to_url or ""

    start_dt = parse_iso_datetime(start_at)
    end_dt = parse_iso_datetime(end_at)

    if not start_dt:
        logger.warning("Skipping event %s: no valid start time", title)
        return ""

    # Use end_dt if valid, otherwise default to start + 1 hour
    if end_dt and end_dt > start_dt:
        end_dt_use = end_dt
    else:
        end_dt_use = start_dt + datetime.timedelta(hours=1)

    # Format for iCal: convert to UTC for Z-suffix format
    if start_dt.tzinfo:
        start_utc = start_dt.astimezone(datetime.timezone.utc)
        end_utc = end_dt_use.astimezone(datetime.timezone.utc)
        start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
        end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")
    else:
        start_str = start_dt.strftime("%Y%m%dT%H%M%S")
        end_str = end_dt_use.strftime("%Y%m%dT%H%M%S")

    # Build description with all available info
    description = strip_html(description_raw)
    desc_parts = [description] if description else []
    if date_or_range:
        desc_parts.append(f"\nWhen: {date_or_range}")
    if timezone_name:
        desc_parts.append(f"Timezone: {timezone_name}")
    additional = event.get("additional_info")
    if additional:
        desc_parts.append(f"\n{strip_html(str(additional))}")
    if ticket_success_message:
        desc_parts.append(f"\n{ticket_success_message}")
    if event_url:
        desc_parts.append(f"\nEvent details: {event_url}")
    if ti_to_url and ti_to_url != event_url:
        desc_parts.append(f"\nBook tickets: {ti_to_url}")
    homepage_url = event.get("homepage_url")
    if homepage_url and homepage_url not in (event_url, ti_to_url):
        desc_parts.append(f"\nEvent website: {homepage_url}")
    # Map/directions: use map_link or build from lat/long
    map_link = event.get("map_link")
    if not map_link:
        lat, lon = event.get("map_latitude"), event.get("map_longitude")
        if lat is not None and lon is not None:
            try:
                map_link = f"https://www.google.com/maps?q={float(lat)},{float(lon)}"
            except (TypeError, ValueError):
                pass
    if map_link:
        desc_parts.append(f"\nMap/directions: {map_link}")
    # Ticket info from releases
    if releases:
        ticket_lines = []
        for r in releases[:5]:
            name = (r.get("title") or r.get("state_name") or "").strip()
            price = r.get("display_price") if r.get("display_price") is not None else r.get("price")
            if name or price is not None:
                if price is not None and price != "":
                    try:
                        p = float(price)
                        ticket_lines.append(f"  • {name}: £{p:.0f}" if name else f"  • £{p:.0f}")
                    except (TypeError, ValueError):
                        ticket_lines.append(f"  • {name}: {price}" if name else f"  • {price}")
                elif name:
                    ticket_lines.append(f"  • {name}")
        if ticket_lines:
            desc_parts.append("\nTickets:\n" + "\n".join(ticket_lines))
    if registrations_count is not None or tickets_count is not None:
        parts = []
        if registrations_count is not None:
            parts.append(f"{registrations_count} registration(s)")
        if tickets_count is not None:
            parts.append(f"{tickets_count} ticket(s)")
        if parts:
            desc_parts.append(f"\n{' | '.join(parts)}")
    desc_parts.append("\nTesla Owners UK event – https://teslaowners.org.uk/events")
    description_text = "\n".join(desc_parts)

    # UID (required): stable unique identifier
    uid = f"{event_id or slug}@teslaowners.org.uk" if (event_id or slug) else None
    if not uid:
        uid = f"{start_str}-{slug or title[:20]}@teslaowners.org.uk"

    # DTSTAMP (required): when the event was created/updated in our system
    dtstamp_str = _format_ical_datetime(datetime.datetime.now(datetime.timezone.utc))
    if updated_at:
        upd_dt = parse_iso_datetime(updated_at)
        if upd_dt:
            dtstamp_str = _format_ical_datetime(upd_dt)

    # Build VEVENT
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp_str}",
        f"DTSTART:{start_str}",
        f"DTEND:{end_str}",
        f"SUMMARY:{title}",
        escape_and_fold_ical_text(description_text, "DESCRIPTION:"),
    ]

    # CREATED
    if created_at:
        created_dt = parse_iso_datetime(created_at)
        if created_dt:
            lines.append(f"CREATED:{_format_ical_datetime(created_dt)}")

    # LAST-MODIFIED
    if updated_at:
        mod_dt = parse_iso_datetime(updated_at)
        if mod_dt:
            lines.append(f"LAST-MODIFIED:{_format_ical_datetime(mod_dt)}")

    # LOCATION
    if location:
        lines.append(f"LOCATION:{location}")

    # GEO
    lat, lon = event.get("map_latitude"), event.get("map_longitude")
    if lat is not None and lon is not None:
        try:
            lines.append(f"GEO:{float(lat)};{float(lon)}")
        except (TypeError, ValueError):
            pass

    # URL
    if event_url:
        lines.append(f"URL:{event_url}")

    # ORGANIZER (CN=Common Name)
    organizer_email = reply_to_email or email_address
    if organizer_email:
        org_name = "Tesla Owners UK"
        lines.append(f'ORGANIZER;CN={org_name}:mailto:{organizer_email}')

    # CONTACT (repeatable) - add all unique contact emails
    seen_emails = set()
    for email in (reply_to_email, email_address):
        if email and email not in seen_emails:
            seen_emails.add(email)
            lines.append(f"CONTACT:mailto:{email}")

    # COMMENT (repeatable) - extra metadata
    if timezone_name:
        lines.append(escape_and_fold_ical_text(f"Timezone: {timezone_name}", "COMMENT:"))
    if date_or_range:
        lines.append(escape_and_fold_ical_text(f"Date: {date_or_range}", "COMMENT:"))

    # CATEGORIES
    lines.append("CATEGORIES:Tesla Owners UK,Event")

    # CLASS (PUBLIC for public events)
    lines.append("CLASS:PUBLIC")

    # STATUS
    lines.append("STATUS:CONFIRMED")

    # ATTACH - banner image
    if banner_url and banner_url.startswith("http"):
        lines.append(f"ATTACH:{banner_url}")

    # RESOURCES - venue/location as resource
    if location:
        lines.append(f"RESOURCES:{location}")

    # X- properties for extra metadata
    if timezone_name:
        lines.append(f"X-TIMEZONE:{timezone_name}")
    if event_id:
        lines.append(f"X-EVENT-ID:{event_id}")
    if tickets_count is not None:
        lines.append(f"X-TICKETS-COUNT:{tickets_count}")
    if registrations_count is not None:
        lines.append(f"X-REGISTRATIONS-COUNT:{registrations_count}")

    # VALARMs (nested components)
    if NOTIFICATIONS.get("enabled", False):
        for alarm in NOTIFICATIONS.get("alarms", []):
            lines.extend(line for line in generate_alarm(alarm, start_dt).split("\n") if line.strip())

    lines.append("END:VEVENT")
    return "\n".join(lines) + "\n"


def build_index_html(events: List[Dict[str, Any]], upcoming_count: Optional[int] = None) -> str:
    """Build the index HTML page with calendar link and featured events."""
    ics_url = "tocuk.ics"
    count = upcoming_count if upcoming_count is not None else len(events)

    # Featured: show upcoming events first, then past (up to 8)
    featured_html = ""
    if events:
        today = datetime.datetime.now(datetime.timezone.utc)
        upcoming_first = []
        past = []
        for e in events:
            start = parse_iso_datetime(e.get("start_at"))
            if start:
                s = start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start
                (upcoming_first if s >= today else past).append(e)
            else:
                upcoming_first.append(e)
        ordered = upcoming_first + past
        featured_items = []
        for e in ordered[:8]:
            start = parse_iso_datetime(e.get("start_at"))
            date_str = start.strftime("%d %b %Y") if start else ""
            title = e.get("title", "Event")
            featured_items.append(
                f'<div class="featured-event">'
                f'<span class="date">{date_str}</span>{title}'
                f"</div>"
            )
        featured_html = "\n        ".join(featured_items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tesla Owners UK – Events Calendar</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0a0a0f;
      --surface: #12121a;
      --surface-2: #1a1a24;
      --border: rgba(0,212,255,0.25);
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --cyan: #00d4ff;
      --red: #e53935;
      --accent: #00d4ff;
      --accent-dim: rgba(0,212,255,0.15);
      --radius: 16px;
      --radius-sm: 10px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Space Grotesk', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{ max-width: 700px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    .hero {{
      text-align: center;
      padding: 3rem 0 2rem;
    }}
    .hero .badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--accent);
      background: var(--accent-dim);
      padding: 0.4rem 1rem;
      border-radius: 100px;
      margin-bottom: 1rem;
    }}
    .hero h1 {{
      font-size: clamp(1.75rem, 5vw, 2.5rem);
      font-weight: 800;
      margin-bottom: 0.75rem;
    }}
    .hero .tagline {{ color: var(--text-muted); font-size: 1rem; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      margin: 2rem 0;
      text-align: center;
    }}
    .card h2 {{ font-size: 1.25rem; margin-bottom: 1rem; }}
    .card .meta {{ color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.95rem; }}
    .card .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.75rem 1.5rem;
      background: linear-gradient(135deg, var(--cyan), #a855f7);
      color: var(--bg);
      font-weight: 600;
      font-size: 1rem;
      border-radius: var(--radius-sm);
      text-decoration: none;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card .btn:hover {{
      transform: scale(1.02);
      box-shadow: 0 4px 20px rgba(0,212,255,0.3);
    }}
    .featured-section {{ margin: 3rem 0; }}
    .featured-section h2 {{ font-size: 1.25rem; margin-bottom: 1rem; }}
    .featured-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 1rem;
    }}
    .featured-event {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1rem;
      font-size: 0.9rem;
    }}
    .featured-event .date {{
      font-size: 0.75rem;
      color: var(--accent);
      font-weight: 600;
      margin-bottom: 0.5rem;
      display: block;
    }}
    .howto-section {{ margin: 3rem 0; }}
    .howto-section h2 {{ font-size: 1.25rem; margin-bottom: 1rem; }}
    .howto-section p {{ color: var(--text-muted); margin-bottom: 0.5rem; font-size: 0.95rem; }}
    footer {{
      text-align: center;
      padding: 2rem;
      color: var(--text-muted);
      font-size: 0.85rem;
      border-top: 1px solid rgba(255,255,255,0.06);
      margin-top: 3rem;
    }}
    footer a {{ color: var(--cyan); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="badge">Tesla Owners UK · Live</div>
      <h1>Events Calendar</h1>
      <p class="tagline">Subscribe to Tesla Owners UK events. Track days, meetups, AGMs and more.</p>
    </header>

    <section class="card">
      <h2>Subscribe to the calendar</h2>
      <p class="meta">{count} upcoming event{'' if count == 1 else 's'}</p>
      <a href="{ics_url}" class="btn">Subscribe to calendar</a>
    </section>

    <section class="featured-section">
      <h2>Upcoming events</h2>
      <div class="featured-grid">
        {featured_html}
      </div>
    </section>

    <section class="howto-section">
      <h2>How to use</h2>
      <p><strong>Google Calendar:</strong> Add other calendars → From URL → paste the calendar link.</p>
      <p><strong>Apple Calendar:</strong> File → New Calendar Subscription → paste the URL.</p>
      <p><strong>Outlook:</strong> Add calendar → Subscribe from web → paste the URL.</p>
    </section>

    <footer>
      <p>Fan-made project. Not affiliated with Tesla Owners UK Limited.</p>
      <p style="margin-top: 0.5rem;">
        <a href="https://teslaowners.org.uk/events">Tesla Owners UK Events</a>
        <span aria-hidden="true"> · </span>
        <a href="https://github.com/evenwebb/tesla-owners-club-uk-events-calendar">Source</a>
      </p>
    </footer>
  </div>
  <script>
  (function() {{
    var ua = navigator.userAgent || '';
    var isIOS = /iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    var isMac = /Macintosh|Mac OS X/.test(ua) && !isIOS;
    document.querySelectorAll('a[href$=".ics"]').forEach(function(link) {{
      var href = link.getAttribute('href');
      if (!href) return;
      var abs = new URL(href, window.location.href).href;
      if (isIOS || isMac) link.href = abs.replace(/^https?:\\/\\//, 'webcal://');
    }});
  }})();
  </script>
</body>
</html>"""


def main() -> None:
    """Main function to scrape events and generate iCal file."""
    logger.info("Fetching events from %s", EVENTS_URL)
    response = fetch_with_retries(EVENTS_URL)
    events = extract_events_from_page(response.text)

    if not events:
        logger.warning("No events found")
        print("No events found.")
        return

    # Split into future and past (keep all events for output)
    today = datetime.datetime.now(datetime.timezone.utc)
    future_events = []
    past_events = []
    for e in events:
        start = parse_iso_datetime(e.get("start_at"))
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=datetime.timezone.utc)
            if start >= today:
                future_events.append(e)
            else:
                past_events.append(e)
        else:
            future_events.append(e)  # No start = treat as future

    # Skip full scrape if no NEW upcoming events (reduces GitHub Actions usage)
    # Only update when an event is added; ignore events moving to past
    current_upcoming_slugs = {e.get("slug") for e in future_events if e.get("slug")}
    previous_slugs = load_last_upcoming_slugs()
    if not has_new_events(current_upcoming_slugs, previous_slugs):
        logger.info("No new events (current=%d, previous=%d), skipping full scrape", len(current_upcoming_slugs), len(previous_slugs))
        print("No new events. Skipping update to reduce GitHub usage.")
        # Update state file to reflect current upcoming events (important for when events move to past)
        save_last_upcoming_slugs(list(current_upcoming_slugs))
        return

    # All events to output: past first, then future (sorted by start date)
    all_events = past_events + future_events
    all_events.sort(key=lambda x: parse_iso_datetime(x.get("start_at", "")) or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))

    # Fetch detail page for each event and merge (richer description, map coords, etc.)
    cache = load_cache()
    enriched_events = []
    for event in all_events:
        slug = event.get("slug")
        detail = fetch_event_detail(slug, cache) if slug else None
        enriched_events.append(merge_event_detail(event, detail))
    save_cache(cache)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Generate iCal
    event_lines = []
    for event in enriched_events:
        ical = make_ics_event(event)
        if ical:
            event_lines.append(ical)

    ical_content = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//Tesla Owners UK//Events Calendar//EN\n"
        "CALSCALE:GREGORIAN\n"
        "X-WR-CALNAME:Tesla Owners UK Events\n"
        "X-WR-CALDESC:Upcoming events from Tesla Owners UK\n"
        + "".join(event_lines)
        + "END:VCALENDAR\n"
    )

    ics_path = Path(OUTPUT_DIR) / "tocuk.ics"
    ics_path.write_text(ical_content, encoding="utf-8")
    logger.info("Wrote %s (%d events)", ics_path, len(event_lines))

    # Save current upcoming slugs for next run comparison (reuse from earlier)
    save_last_upcoming_slugs(list(current_upcoming_slugs))

    # Generate index
    index_path = Path(OUTPUT_DIR) / "index.html"
    index_path.write_text(build_index_html(enriched_events, upcoming_count=len(current_upcoming_slugs)), encoding="utf-8")
    logger.info("Wrote %s", index_path)

    print(f"\n✓ Created {OUTPUT_DIR}/ with tocuk.ics ({len(event_lines)} events) and index.html\n")
    for event in enriched_events:
        start = parse_iso_datetime(event.get("start_at"))
        date_str = start.strftime("%d %B %Y %H:%M") if start else "?"
        print(f"  • {event.get('title')} – {date_str} @ {event.get('location', 'TBC')}")


if __name__ == "__main__":
    main()
