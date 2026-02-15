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

# Health status file for monitoring
HEALTH_FILE = "docs/.health_status.json"

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


def validate_event_data(event: Dict[str, Any]) -> bool:
    """Validate that event dict has minimum required fields.

    Args:
        event: Event dictionary to validate

    Returns:
        True if event has required fields, False otherwise
    """
    # Must have at least a title or slug
    if not event.get("title") and not event.get("slug"):
        logger.warning("Event missing both title and slug: %s", event)
        return False

    # Must have a start time
    if not event.get("start_at"):
        logger.warning("Event %s missing start_at", event.get("title") or event.get("slug"))
        return False

    return True


def extract_events_from_page(html: str) -> List[Dict[str, Any]]:
    """Extract events from __NEXT_DATA__ JSON embedded in the page with validation.

    Returns:
        List of valid event dicts with title, start_at, end_at, location, description, url, slug.
    """
    # Try to find __NEXT_DATA__
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        logger.error("Could not find __NEXT_DATA__ in page")
        # Attempt fallback: look for alternative script tags or data attributes
        # (This would be where you'd add alternative parsing strategies if the site changes)
        logger.info("Attempting fallback parsing strategies...")
        # For now, return empty - could add more sophisticated fallbacks here
        return []

    # Parse JSON with error handling
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error("Failed to parse __NEXT_DATA__ JSON: %s", e)
        logger.debug("JSON content: %s", match.group(1)[:500])  # Log first 500 chars for debugging
        return []

    # Validate schema structure
    if not isinstance(data, dict):
        logger.error("__NEXT_DATA__ is not a dict: %s", type(data))
        return []

    props = data.get("props")
    if not isinstance(props, dict):
        logger.error("props is missing or not a dict")
        return []

    page_props = props.get("pageProps")
    if not isinstance(page_props, dict):
        logger.error("pageProps is missing or not a dict")
        return []

    events_data = page_props.get("events")
    if not isinstance(events_data, dict):
        logger.warning("events is missing or not a dict, trying direct event list")
        # Fallback: check if pageProps itself contains event array
        if isinstance(page_props.get("events"), list):
            upcoming = page_props.get("events", [])
            past = []
        else:
            logger.error("Could not find events data in expected structure")
            return []
    else:
        upcoming = events_data.get("upcoming", [])
        past = events_data.get("past", [])

    # Validate event lists are actually lists
    if not isinstance(upcoming, list):
        logger.warning("upcoming is not a list, using empty list")
        upcoming = []
    if not isinstance(past, list):
        logger.warning("past is not a list, using empty list")
        past = []

    # Merge and deduplicate by slug (upcoming takes precedence)
    # Also validate each event has minimum required fields
    seen = set()
    merged = []
    invalid_count = 0

    for e in upcoming + past:
        if not isinstance(e, dict):
            logger.warning("Skipping non-dict event: %s", type(e))
            invalid_count += 1
            continue

        # Validate event has required fields
        if not validate_event_data(e):
            invalid_count += 1
            continue

        slug = e.get("slug")
        if slug and slug not in seen:
            seen.add(slug)
            merged.append(e)
        elif not slug:
            # Events without slugs are not deduplicated
            merged.append(e)

    if invalid_count > 0:
        logger.warning("Skipped %d invalid events", invalid_count)

    logger.info("Successfully extracted %d valid events", len(merged))
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


def save_health_status(
    status: str,
    event_count: int,
    message: str = "",
    error: Optional[str] = None
) -> None:
    """Save health status for monitoring and display.

    Args:
        status: "success", "partial", or "error"
        event_count: Number of events processed
        message: Human-readable status message
        error: Error message if status is "error"
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    health_data = {
        "status": status,
        "last_update": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_count": event_count,
        "message": message,
        "error": error,
    }
    try:
        with open(HEALTH_FILE, "w", encoding="utf-8") as f:
            json.dump(health_data, f, indent=2)
        logger.info("Saved health status: %s", status)
    except OSError as e:
        logger.warning("Health status save failed: %s", e)


def load_health_status() -> Optional[Dict[str, Any]]:
    """Load health status from disk."""
    if not os.path.exists(HEALTH_FILE):
        return None
    try:
        with open(HEALTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_event_from_detail_page(html: str) -> Optional[Dict[str, Any]]:
    """Extract full event data from __NEXT_DATA__ on an event detail page with validation.

    Returns:
        Event dict from pageProps.event, or None if not found or invalid.
    """
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        logger.debug("No __NEXT_DATA__ found in detail page")
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse detail page JSON: %s", e)
        return None

    # Validate structure
    if not isinstance(data, dict):
        logger.warning("Detail page data is not a dict")
        return None

    props = data.get("props")
    if not isinstance(props, dict):
        logger.warning("Detail page props missing or invalid")
        return None

    page_props = props.get("pageProps")
    if not isinstance(page_props, dict):
        logger.warning("Detail page pageProps missing or invalid")
        return None

    event = page_props.get("event")
    if not isinstance(event, dict):
        logger.warning("Detail page event missing or not a dict")
        return None

    # Validate minimum fields
    if not validate_event_data(event):
        logger.warning("Detail page event failed validation")
        return None

    return event


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

    # ATTACH - banner image with format type for better client support
    if banner_url and banner_url.startswith("http"):
        # Determine MIME type from URL extension
        fmttype = "image/jpeg"
        if banner_url.lower().endswith(".png"):
            fmttype = "image/png"
        elif banner_url.lower().endswith(".gif"):
            fmttype = "image/gif"
        elif banner_url.lower().endswith(".webp"):
            fmttype = "image/webp"
        lines.append(f"ATTACH;FMTTYPE={fmttype}:{banner_url}")

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


def get_event_status(event: Dict[str, Any]) -> tuple[str, str]:
    """Determine event status and badge emoji.

    Returns:
        (status_text, emoji) tuple
    """
    tickets_count = event.get("tickets_count")
    registrations_count = event.get("registrations_count")
    releases = event.get("releases") or []

    # Check if sold out (no tickets available in any release)
    if releases:
        available = sum(r.get("quantity_available", 0) for r in releases if isinstance(r.get("quantity_available"), int))
        if available == 0:
            return ("Sold out", "❌")
        elif available <= 10:
            return (f"{available} tickets left", "⚠️")

    # Check registration/ticket counts
    if tickets_count is not None and tickets_count > 0:
        if registrations_count is not None:
            capacity_pct = (registrations_count / tickets_count) * 100 if tickets_count > 0 else 0
            if capacity_pct >= 100:
                return ("Sold out", "❌")
            elif capacity_pct >= 80:
                return ("Limited tickets", "⚠️")
            else:
                return ("Open for registration", "✅")
        else:
            return ("Open for registration", "✅")

    return ("", "")


def build_index_html(
    events: List[Dict[str, Any]],
    upcoming_count: Optional[int] = None,
    health_status: Optional[Dict[str, Any]] = None
) -> str:
    """Build the index HTML page with calendar link and featured events."""
    ics_url = "tocuk.ics"
    count = upcoming_count if upcoming_count is not None else len(events)

    # Build health status display
    health_html = ""
    if health_status:
        last_update = health_status.get("last_update", "")
        status = health_status.get("status", "unknown")
        message = health_status.get("message", "")
        error = health_status.get("error")

        if last_update:
            try:
                update_dt = datetime.datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                now = datetime.datetime.now(datetime.timezone.utc)
                delta = now - update_dt
                if delta.total_seconds() < 3600:
                    time_ago = f"{int(delta.total_seconds() / 60)} minutes ago"
                elif delta.total_seconds() < 86400:
                    time_ago = f"{int(delta.total_seconds() / 3600)} hours ago"
                else:
                    time_ago = f"{int(delta.total_seconds() / 86400)} days ago"
            except (ValueError, TypeError):
                time_ago = "recently"
        else:
            time_ago = "unknown"

        status_emoji = "✅" if status == "success" else ("⚠️" if status == "partial" else "❌")
        status_class = "success" if status == "success" else ("warning" if status == "partial" else "error")

        health_html = f'''
    <div class="health-status {status_class}">
      <div class="health-icon">{status_emoji}</div>
      <div class="health-info">
        <div class="health-main">Last updated: {time_ago}</div>
        <div class="health-message">{message}</div>
        {f'<div class="health-error">Error: {error}</div>' if error else ''}
      </div>
    </div>'''

    # Featured: show upcoming events first, then past - render ALL for filtering
    featured_html = ""
    events_json_data = []

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

        # Build event cards with full data for search/filter
        featured_items = []
        for idx, e in enumerate(ordered):
            start = parse_iso_datetime(e.get("start_at"))
            date_str = start.strftime("%d %b %Y") if start else ""
            time_str = start.strftime("%H:%M") if start else ""
            title = e.get("title", "Event")
            location = e.get("location", "")
            description = strip_html(e.get("description", ""))[:200]
            slug = e.get("slug", "")
            url = f"https://teslaowners.org.uk/events/{slug}" if slug else ""
            status_text, status_emoji = get_event_status(e)

            # Determine if upcoming
            is_upcoming = start and (start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start) >= today
            status_badge = ""
            if is_upcoming and status_text:
                status_badge = f'<span class="status-badge">{status_emoji} {status_text}</span>'

            # Store event data for JavaScript
            events_json_data.append({
                "title": title,
                "date": date_str,
                "time": time_str,
                "location": location,
                "description": description,
                "url": url,
                "upcoming": is_upcoming
            })

            # Event card with click handler
            featured_items.append(
                f'<div class="featured-event" data-event-idx="{idx}" onclick="showEventModal({idx})">'
                f'<span class="date">{date_str}</span>'
                f'<span class="title">{title}</span>'
                f'<span class="location-small">{location[:30]}{"..." if len(location) > 30 else ""}</span>'
                f'{status_badge}'
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
    [data-theme="light"] {{
      --bg: #ffffff;
      --surface: #f8f9fa;
      --surface-2: #e9ecef;
      --border: rgba(0,0,0,0.1);
      --text: #1a1a1a;
      --text-muted: #6c757d;
      --accent-dim: rgba(0,212,255,0.1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Space Grotesk', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      transition: background 0.3s, color 0.3s;
    }}
    .page {{ max-width: 700px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    .controls {{
      display: flex;
      gap: 1rem;
      margin: 1.5rem 0;
      flex-wrap: wrap;
    }}
    .search-box {{
      flex: 1;
      min-width: 200px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.75rem 1rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
    }}
    .search-box::placeholder {{ color: var(--text-muted); }}
    .theme-toggle {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.75rem 1rem;
      color: var(--text);
      cursor: pointer;
      font-size: 1.2rem;
      transition: transform 0.2s;
    }}
    .theme-toggle:hover {{ transform: scale(1.05); }}
    .modal {{
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.8);
      z-index: 1000;
      padding: 2rem;
      overflow-y: auto;
    }}
    .modal.active {{ display: flex; align-items: center; justify-content: center; }}
    .modal-content {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      max-width: 600px;
      width: 100%;
      position: relative;
    }}
    .modal-close {{
      position: absolute;
      top: 1rem;
      right: 1rem;
      background: none;
      border: none;
      font-size: 1.5rem;
      color: var(--text-muted);
      cursor: pointer;
      padding: 0.5rem;
      line-height: 1;
    }}
    .modal-close:hover {{ color: var(--text); }}
    .modal-title {{ font-size: 1.5rem; margin-bottom: 1rem; padding-right: 2rem; }}
    .modal-meta {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .modal-meta strong {{ color: var(--accent); }}
    .modal-description {{ line-height: 1.7; margin-bottom: 1.5rem; }}
    .modal-link {{
      display: inline-block;
      padding: 0.75rem 1.5rem;
      background: linear-gradient(135deg, var(--cyan), #a855f7);
      color: var(--bg);
      text-decoration: none;
      border-radius: var(--radius-sm);
      font-weight: 600;
    }}
    .modal-link:hover {{ opacity: 0.9; }}
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
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      cursor: pointer;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .featured-event:hover {{
      transform: translateY(-2px);
      border-color: var(--accent);
    }}
    .featured-event .date {{
      font-size: 0.75rem;
      color: var(--accent);
      font-weight: 600;
      display: block;
    }}
    .featured-event .title {{
      font-weight: 500;
      line-height: 1.3;
    }}
    .featured-event .location-small {{
      font-size: 0.75rem;
      color: var(--text-muted);
      display: block;
    }}
    .featured-event .status-badge {{
      font-size: 0.7rem;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      background: var(--surface-2);
      border: 1px solid var(--border);
      display: inline-block;
      margin-top: 0.25rem;
      width: fit-content;
    }}
    .featured-event.hidden {{ display: none; }}
    .health-status {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1rem;
      margin: 1.5rem 0;
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.9rem;
    }}
    .health-status.success {{ border-color: rgba(34, 197, 94, 0.4); }}
    .health-status.warning {{ border-color: rgba(251, 191, 36, 0.4); }}
    .health-status.error {{ border-color: rgba(239, 68, 68, 0.4); }}
    .health-icon {{ font-size: 1.5rem; }}
    .health-info {{ flex: 1; }}
    .health-main {{ font-weight: 500; }}
    .health-message {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }}
    .health-error {{ font-size: 0.85rem; color: #ef4444; margin-top: 0.25rem; }}
    .howto-section {{ margin: 3rem 0; }}
    .howto-section h2 {{ font-size: 1.25rem; margin-bottom: 0.5rem; }}
    .howto-intro {{ color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.95rem; }}
    .howto-footer {{
      margin-top: 1.5rem;
      padding: 1rem;
      background: var(--surface-2);
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
      text-align: center;
      color: var(--text-muted);
    }}
    .accordion-group {{ display: flex; flex-direction: column; gap: 0.5rem; }}
    .accordion-button {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      width: 100%;
      padding: 1rem 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text);
      font-family: inherit;
      font-size: 1rem;
      font-weight: 500;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .accordion-button:hover {{
      background: var(--surface-2);
      border-color: var(--accent);
    }}
    .accordion-button[aria-expanded="true"] {{
      border-bottom-left-radius: 0;
      border-bottom-right-radius: 0;
      border-bottom-color: transparent;
    }}
    .accordion-icon {{ font-size: 1.25rem; }}
    .accordion-title {{ flex: 1; }}
    .accordion-chevron {{
      font-size: 0.75rem;
      transition: transform 0.2s;
    }}
    .accordion-button[aria-expanded="true"] .accordion-chevron {{
      transform: rotate(180deg);
    }}
    .accordion-content {{
      display: none;
      padding: 1.5rem 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-top: none;
      border-bottom-left-radius: var(--radius-sm);
      border-bottom-right-radius: var(--radius-sm);
      margin-bottom: 0.5rem;
    }}
    .accordion-content.active {{ display: block; }}
    .instructions-list {{
      margin: 0.75rem 0;
      padding-left: 1.5rem;
      line-height: 1.8;
    }}
    .instructions-list li {{ margin-bottom: 0.5rem; }}
    .instructions-list strong {{ color: var(--accent); }}
    .instructions-list a {{ color: var(--cyan); text-decoration: none; }}
    .instructions-list a:hover {{ text-decoration: underline; }}
    .note {{
      margin-top: 1rem;
      padding: 0.75rem;
      background: var(--surface-2);
      border-radius: 6px;
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
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
      <div class="badge">Tesla Owners UK · Live Calendar</div>
      <h1>Tesla Owners UK Events</h1>
      <p class="tagline">Never miss a Tesla Owners UK event — track days, meetups, AGMs and exclusive gatherings. Subscribe once, stay updated forever.</p>
    </header>
{health_html}
    <section class="card">
      <h2>Subscribe to the calendar</h2>
      <p class="meta">{count} upcoming event{'' if count == 1 else 's'}</p>
      <a href="{ics_url}" class="btn">Subscribe to calendar</a>
    </section>

    <div class="controls">
      <input type="text" class="search-box" id="searchBox" placeholder="Search events by title or location..." onkeyup="filterEvents()">
      <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark/light mode">🌓</button>
    </div>

    <section class="featured-section">
      <h2>Upcoming events <span id="eventCount"></span></h2>
      <div class="featured-grid" id="eventGrid">
        {featured_html}
      </div>
    </section>

    <section class="howto-section">
      <h2>How to subscribe</h2>
      <p class="howto-intro">Choose your calendar app to see instructions:</p>

      <div class="accordion-group">
        <button class="accordion-button" onclick="toggleAccordion('google')" aria-expanded="false">
          <span class="accordion-icon">📅</span>
          <span class="accordion-title">Google Calendar</span>
          <span class="accordion-chevron">▼</span>
        </button>
        <div class="accordion-content" id="accordion-google">
          <ol class="instructions-list">
            <li>Copy the calendar URL above (click the "Subscribe to calendar" button)</li>
            <li>Open <a href="https://calendar.google.com" target="_blank" rel="noopener">Google Calendar</a></li>
            <li>On the left sidebar, click the <strong>+</strong> next to "Other calendars"</li>
            <li>Select <strong>"From URL"</strong></li>
            <li>Paste the calendar URL and click <strong>Add calendar</strong></li>
          </ol>
          <p class="note">💡 Tip: On Android, the subscribe button will open Google Calendar automatically</p>
        </div>

        <button class="accordion-button" onclick="toggleAccordion('apple')" aria-expanded="false">
          <span class="accordion-icon">🍎</span>
          <span class="accordion-title">Apple Calendar (iPhone/iPad/Mac)</span>
          <span class="accordion-chevron">▼</span>
        </button>
        <div class="accordion-content" id="accordion-apple">
          <ol class="instructions-list">
            <li>On iPhone/iPad, tap the "Subscribe to calendar" button above</li>
            <li>Your device will automatically open Calendar and prompt to subscribe</li>
            <li>Tap <strong>Subscribe</strong> to add the calendar</li>
          </ol>
          <p class="note"><strong>On Mac:</strong></p>
          <ol class="instructions-list">
            <li>Click the "Subscribe to calendar" button (it will use webcal:// protocol)</li>
            <li>Or manually: File → New Calendar Subscription → paste the URL</li>
          </ol>
        </div>

        <button class="accordion-button" onclick="toggleAccordion('outlook')" aria-expanded="false">
          <span class="accordion-icon">📧</span>
          <span class="accordion-title">Outlook</span>
          <span class="accordion-chevron">▼</span>
        </button>
        <div class="accordion-content" id="accordion-outlook">
          <ol class="instructions-list">
            <li>Copy the calendar URL above</li>
            <li>Open <a href="https://outlook.live.com/calendar" target="_blank" rel="noopener">Outlook Calendar</a></li>
            <li>Click <strong>Add calendar</strong></li>
            <li>Select <strong>"Subscribe from web"</strong></li>
            <li>Paste the calendar URL and click <strong>Import</strong></li>
          </ol>
        </div>

        <button class="accordion-button" onclick="toggleAccordion('other')" aria-expanded="false">
          <span class="accordion-icon">🗓️</span>
          <span class="accordion-title">Other Calendar Apps</span>
          <span class="accordion-chevron">▼</span>
        </button>
        <div class="accordion-content" id="accordion-other">
          <p>Most calendar applications support iCalendar (.ics) subscriptions:</p>
          <ol class="instructions-list">
            <li>Copy the calendar URL from the "Subscribe to calendar" button</li>
            <li>Look for "Add calendar," "Subscribe," or "Import" in your calendar app</li>
            <li>Choose "From URL" or "Web calendar" option</li>
            <li>Paste the calendar URL</li>
          </ol>
          <p class="note"><strong>Compatible apps:</strong> Mozilla Thunderbird, Yahoo Calendar, CalDAV-compatible apps, and more</p>
        </div>
      </div>

      <p class="howto-footer">ℹ️ The calendar updates automatically when new events are added — no need to re-subscribe!</p>
    </section>

    <footer>
      <p>An open source fan-made project. Not affiliated with Tesla Owners UK Limited.</p>
      <p style="margin-top: 0.5rem;">
        <a href="archive.html">📁 Event Archive</a>
        <span aria-hidden="true"> · </span>
        <a href="https://teslaowners.org.uk/events" target="_blank" rel="noopener">🔗 Official Events Page</a>
        <span aria-hidden="true"> · </span>
        <a href="https://github.com/evenwebb/tesla-owners-club-uk-events-calendar" target="_blank" rel="noopener">⭐ GitHub</a>
      </p>
    </footer>
  </div>

  <!-- Event Detail Modal -->
  <div id="eventModal" class="modal" onclick="if(event.target===this) closeModal()">
    <div class="modal-content">
      <button class="modal-close" onclick="closeModal()">&times;</button>
      <h2 class="modal-title" id="modalTitle"></h2>
      <div class="modal-meta">
        <div><strong>📅 Date:</strong> <span id="modalDate"></span></div>
        <div><strong>🕒 Time:</strong> <span id="modalTime"></span></div>
        <div><strong>📍 Location:</strong> <span id="modalLocation"></span></div>
      </div>
      <div class="modal-description" id="modalDescription"></div>
      <a id="modalLink" href="#" target="_blank" rel="noopener" class="modal-link">View full details →</a>
    </div>
  </div>

  <script>
  // Event data
  const eventsData = {json.dumps(events_json_data)};

  // Dark mode toggle
  function toggleTheme() {{
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  }}

  // Load saved theme
  (function() {{
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }})();

  // Filter events
  function filterEvents() {{
    const searchTerm = document.getElementById('searchBox').value.toLowerCase();
    const eventCards = document.querySelectorAll('.featured-event');
    let visibleCount = 0;

    eventCards.forEach((card, idx) => {{
      const event = eventsData[idx];
      const matchesSearch = !searchTerm ||
        event.title.toLowerCase().includes(searchTerm) ||
        event.location.toLowerCase().includes(searchTerm) ||
        event.description.toLowerCase().includes(searchTerm);

      if (matchesSearch) {{
        card.classList.remove('hidden');
        visibleCount++;
      }} else {{
        card.classList.add('hidden');
      }}
    }});

    // Update count
    const countSpan = document.getElementById('eventCount');
    if (searchTerm) {{
      countSpan.textContent = `({{visibleCount}} shown)`;
    }} else {{
      countSpan.textContent = '';
    }}
  }}

  // Show event modal
  function showEventModal(idx) {{
    const event = eventsData[idx];
    document.getElementById('modalTitle').textContent = event.title;
    document.getElementById('modalDate').textContent = event.date;
    document.getElementById('modalTime').textContent = event.time || 'TBA';
    document.getElementById('modalLocation').textContent = event.location || 'TBA';
    document.getElementById('modalDescription').textContent = event.description || 'No description available.';
    document.getElementById('modalLink').href = event.url || '#';

    if (!event.url) {{
      document.getElementById('modalLink').style.display = 'none';
    }} else {{
      document.getElementById('modalLink').style.display = 'inline-block';
    }}

    document.getElementById('eventModal').classList.add('active');
    document.body.style.overflow = 'hidden';
  }}

  // Close modal
  function closeModal() {{
    document.getElementById('eventModal').classList.remove('active');
    document.body.style.overflow = '';
  }}

  // Escape key closes modal
  document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeModal();
  }});

  // Accordion toggle
  function toggleAccordion(id) {{
    const button = event.currentTarget;
    const content = document.getElementById('accordion-' + id);
    const isExpanded = button.getAttribute('aria-expanded') === 'true';

    // Close all accordions
    document.querySelectorAll('.accordion-button').forEach(btn => {{
      btn.setAttribute('aria-expanded', 'false');
    }});
    document.querySelectorAll('.accordion-content').forEach(content => {{
      content.classList.remove('active');
    }});

    // Toggle current
    if (!isExpanded) {{
      button.setAttribute('aria-expanded', 'true');
      content.classList.add('active');
    }}
  }}

  // Device-specific calendar link handling
  (function() {{
    var ua = navigator.userAgent || '';
    var isIOS = /iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    var isMac = /Macintosh|Mac OS X/.test(ua) && !isIOS;
    var isAndroid = /Android/.test(ua);

    document.querySelectorAll('a[href$=".ics"]').forEach(function(link) {{
      var href = link.getAttribute('href');
      if (!href) return;
      var abs = new URL(href, window.location.href).href;

      if (isIOS || isMac) {{
        // Use webcal:// protocol for Apple devices
        link.href = abs.replace(/^https?:\\/\\//, 'webcal://');
      }} else if (isAndroid) {{
        // Use Google Calendar render URL for Android
        link.href = 'https://calendar.google.com/calendar/render?cid=' + encodeURIComponent(abs);
      }}
    }});
  }})();
  </script>
</body>
</html>"""


def build_archive_html(past_events: List[Dict[str, Any]]) -> str:
    """Build the archive HTML page with past events organized by year/month."""
    if not past_events:
        events_html = '<p class="no-events">No past events in archive yet.</p>'
        return build_archive_template(events_html, 0)

    # Group by year and month
    events_by_year_month = {}
    for e in past_events:
        start = parse_iso_datetime(e.get("start_at"))
        if not start:
            continue
        year = start.year
        month = start.strftime("%B")
        key = (year, month, start.month)  # Include numeric month for sorting
        if key not in events_by_year_month:
            events_by_year_month[key] = []
        events_by_year_month[key].append((e, start))

    # Sort by year/month descending (most recent first)
    sorted_groups = sorted(events_by_year_month.items(), key=lambda x: (x[0][0], x[0][2]), reverse=True)

    # Build HTML
    events_html = ""
    for (year, month_name, _), events_list in sorted_groups:
        events_html += f'<div class="archive-group"><h3>{month_name} {year}</h3><div class="archive-events">'
        # Sort events within month by date
        events_list.sort(key=lambda x: x[1], reverse=True)
        for event, start in events_list:
            title = event.get("title", "Untitled Event")
            location = event.get("location", "")
            date_str = start.strftime("%d %b %Y")
            slug = event.get("slug", "")
            url = f"https://teslaowners.org.uk/events/{slug}" if slug else ""

            location_html = f'<span class="location">{location}</span>' if location else ''
            link_html = f'<a href="{url}" target="_blank" rel="noopener">View details →</a>' if url else ''

            events_html += f'''
        <div class="archive-event">
          <div class="archive-date">{date_str}</div>
          <div class="archive-details">
            <div class="archive-title">{title}</div>
            {location_html}
          </div>
          {link_html}
        </div>'''
        events_html += '</div></div>'

    return build_archive_template(events_html, len(past_events))


def build_archive_template(events_html: str, event_count: int) -> str:
    """Build the archive page template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Archive – Tesla Owners UK Events Calendar</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0a0a0f;
      --surface: #12121a;
      --surface-2: #1a1a24;
      --border: rgba(0,212,255,0.25);
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --cyan: #00d4ff;
      --accent: #00d4ff;
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
    .page {{ max-width: 800px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    .header {{
      text-align: center;
      padding: 2rem 0;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }}
    .header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
    .header .subtitle {{ color: var(--text-muted); }}
    .header .back-link {{
      display: inline-block;
      margin-top: 1rem;
      color: var(--cyan);
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .header .back-link:hover {{ text-decoration: underline; }}
    .stats {{
      text-align: center;
      padding: 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      margin-bottom: 2rem;
    }}
    .archive-group {{ margin-bottom: 2rem; }}
    .archive-group h3 {{
      font-size: 1.25rem;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .archive-events {{ display: flex; flex-direction: column; gap: 0.75rem; }}
    .archive-event {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1rem;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 1rem;
      align-items: center;
    }}
    .archive-date {{
      font-size: 0.85rem;
      color: var(--accent);
      font-weight: 600;
      white-space: nowrap;
    }}
    .archive-details {{ flex: 1; }}
    .archive-title {{ font-weight: 600; margin-bottom: 0.25rem; }}
    .location {{
      font-size: 0.85rem;
      color: var(--text-muted);
      display: block;
    }}
    .archive-event a {{
      color: var(--cyan);
      text-decoration: none;
      font-size: 0.85rem;
      white-space: nowrap;
    }}
    .archive-event a:hover {{ text-decoration: underline; }}
    .no-events {{
      text-align: center;
      padding: 3rem;
      color: var(--text-muted);
    }}
    @media (max-width: 640px) {{
      .archive-event {{
        grid-template-columns: 1fr;
        gap: 0.5rem;
      }}
      .archive-event a {{ margin-top: 0.5rem; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="header">
      <h1>Event Archive</h1>
      <p class="subtitle">Past Tesla Owners UK events</p>
      <a href="index.html" class="back-link">← Back to calendar</a>
    </header>

    <div class="stats">
      <strong>{event_count}</strong> past event{'' if event_count == 1 else 's'} in archive
    </div>

    {events_html}
  </div>
</body>
</html>"""


def main() -> None:
    """Main function to scrape events and generate iCal file."""
    try:
        logger.info("Fetching events from %s", EVENTS_URL)
        response = fetch_with_retries(EVENTS_URL)
        events = extract_events_from_page(response.text)

        if not events:
            logger.warning("No events found")
            print("No events found.")
            save_health_status(
                "error",
                0,
                "No events found on website",
                "Event extraction returned empty list"
            )
            return
    except Exception as e:
        logger.error("Failed to fetch or parse events: %s", e)
        save_health_status(
            "error",
            0,
            "Failed to fetch events from website",
            str(e)
        )
        raise

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
        # Update health status (no error, just no new events)
        save_health_status(
            "success",
            len(current_upcoming_slugs),
            f"No new events detected ({len(current_upcoming_slugs)} upcoming events)"
        )
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

    # Save health status
    save_health_status(
        "success",
        len(event_lines),
        f"Successfully processed {len(event_lines)} events ({len(current_upcoming_slugs)} upcoming)"
    )

    # Generate index with health status
    health_status = load_health_status()
    index_path = Path(OUTPUT_DIR) / "index.html"
    index_path.write_text(
        build_index_html(enriched_events, upcoming_count=len(current_upcoming_slugs), health_status=health_status),
        encoding="utf-8"
    )
    logger.info("Wrote %s", index_path)

    # Generate archive page with past events
    # Filter enriched events to get only past ones
    today = datetime.datetime.now(datetime.timezone.utc)
    past_enriched = []
    for e in enriched_events:
        start = parse_iso_datetime(e.get("start_at"))
        if start:
            s = start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start
            if s < today:
                past_enriched.append(e)

    archive_path = Path(OUTPUT_DIR) / "archive.html"
    archive_path.write_text(build_archive_html(past_enriched), encoding="utf-8")
    logger.info("Wrote %s with %d past events", archive_path, len(past_enriched))

    print(f"\n✓ Created {OUTPUT_DIR}/ with tocuk.ics ({len(event_lines)} events), index.html, and archive.html\n")
    for event in enriched_events:
        start = parse_iso_datetime(event.get("start_at"))
        date_str = start.strftime("%d %B %Y %H:%M") if start else "?"
        print(f"  • {event.get('title')} – {date_str} @ {event.get('location', 'TBC')}")


if __name__ == "__main__":
    main()
