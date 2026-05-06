"""Tesla Owners Club UK Events Calendar Scraper.

Loads events from the Tesla Owners UK Ti.to account (checkout JSON + per-event
ICS) and generates an iCalendar file plus static HTML. ICS times from Ti.to are
interpreted in Europe/London when no TZID is present (UK club events), then
emitted in UTC for broad mobile client compatibility.
"""
import datetime
import hashlib
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

import requests

# ============================================================================
# CONSTANTS
# ============================================================================
TITO_ACCOUNT = "teslaownersuk"
TITO_CHECKOUT_JSON = f"https://checkout.tito.io/{TITO_ACCOUNT}.json"
TITO_PUBLIC_BASE = f"https://ti.to/{TITO_ACCOUNT}"
# Tesla Owners UK public site (footer / JSON-LD organizer)
CLUB_PUBLIC_URL = "https://teslaowners.uk"
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

# State file: skip full scrape when upcoming metadata fingerprints match last run
STATE_FILE = "docs/.last_upcoming.json"

# Normalised phrases meaning “venue not final yet” (case-insensitive, stripped)
TBC_LOCATION_PHRASES = frozenset(
    {
        "tbc",
        "tba",
        "tbd",
        "n/a",
        "na",
        "none",
        "-",
        "venue tbc",
        "venue tba",
        "location tbc",
        "location tba",
        "to be confirmed",
        "to be announced",
        "to be decided",
        "to be advised",
        "details tbc",
    }
)

# Health status file for monitoring
HEALTH_FILE = "docs/.health_status.json"

# Public site URL for OG tags, sitemap, and robots (override when forking Pages)
_cal_site = (os.environ.get("CALENDAR_SITE_URL") or "").strip()
SITE_PUBLIC_URL = (
    _cal_site
    if _cal_site
    else "https://evenwebb.github.io/tesla-owners-club-uk-events-calendar"
).rstrip("/")

# Per-slug iCal SEQUENCE / DTSTAMP persistence (subscribers see updates reliably)
ICAL_SEQUENCE_FILE = os.path.join(OUTPUT_DIR, ".ical_sequence.json")

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


def validate_tito_list_row(row: Dict[str, Any]) -> bool:
    """Ti.to checkout JSON row: needs slug, title, and display time string."""
    if not row.get("title") and not row.get("slug"):
        logger.warning("Ti.to row missing title and slug: %s", row)
        return False
    if not row.get("time"):
        logger.warning(
            "Ti.to row %s missing time string",
            row.get("slug") or row.get("title"),
        )
        return False
    return True


def validate_event_data(event: Dict[str, Any]) -> bool:
    """Validate enriched event (after ICS merge) for HTML / iCal."""
    if not event.get("title") and not event.get("slug"):
        logger.warning("Event missing both title and slug: %s", event)
        return False
    if not event.get("start_at"):
        logger.warning("Event %s missing start_at", event.get("title") or event.get("slug"))
        return False
    return True


def slug_from_tito_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"ti\.to/" + re.escape(TITO_ACCOUNT) + r"/([^/?#]+)/?", url, re.I)
    return unquote(m.group(1)) if m else None


def tito_event_public_url(slug: Optional[str]) -> str:
    if not slug:
        return ""
    return f"{TITO_PUBLIC_BASE}/{quote(str(slug), safe='-_.~')}"


def unfold_ical_lines(raw: str) -> List[str]:
    """RFC 5545 line unfolding (continuation lines start with space or tab)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if not line:
            continue
        if line[0] in " \t" and out:
            out[-1] += line[1:]
        else:
            out.append(line)
    return out


def unescape_ical_text_value(val: str) -> str:
    """Unescape RFC 5545 TEXT in property values (LOCATION, SUMMARY, etc.)."""
    if not val:
        return ""
    return (
        val.replace("\\N", "\n")
        .replace("\\n", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def split_ical_content_line(line: str) -> Tuple[str, Dict[str, str], str]:
    """Split 'NAME;PARAM=VAL:VALUE' into name (upper), param dict, value."""
    if ":" not in line:
        return line.upper(), {}, ""
    key_part, value = line.split(":", 1)
    sem = key_part.split(";", 1)
    name = sem[0].upper()
    params: Dict[str, str] = {}
    if len(sem) > 1:
        for chunk in sem[1].split(";"):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                params[k.upper()] = v
    return name, params, value


def _trim_ical_datetime_value(value: str) -> str:
    v = value.strip()
    if "." in v and v.endswith("Z"):
        return v.split(".")[0] + "Z"
    if "." in v and "T" in v:
        return v.split(".")[0]
    return v


def ical_value_to_datetime(
    value: str,
    params: Dict[str, str],
    *,
    floating_wall_clock_tz: str = "Europe/London",
    assume_utc_if_naive: bool = False,
) -> Optional[datetime.datetime]:
    """Parse DTSTART/DTEND/CREATED-style values from Ti.to ICS."""
    raw = _trim_ical_datetime_value(value)
    if not raw:
        return None
    tzid = params.get("TZID")
    try:
        if len(raw) == 8 and "T" not in raw:
            d = datetime.datetime.strptime(raw, "%Y%m%d").date()
            tz = ZoneInfo(tzid or floating_wall_clock_tz)
            return datetime.datetime.combine(d, datetime.time.min, tzinfo=tz)
        if raw.endswith("Z"):
            core = raw[:-1]
            for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
                try:
                    return datetime.datetime.strptime(core, fmt).replace(
                        tzinfo=datetime.timezone.utc
                    )
                except ValueError:
                    continue
            return None
        if tzid:
            try:
                tz = ZoneInfo(tzid)
            except Exception:
                tz = ZoneInfo(floating_wall_clock_tz)
            for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
                try:
                    return datetime.datetime.strptime(raw, fmt).replace(tzinfo=tz)
                except ValueError:
                    continue
            return None
        dt_naive = None
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                dt_naive = datetime.datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt_naive is None:
            return None
        if assume_utc_if_naive:
            return dt_naive.replace(tzinfo=datetime.timezone.utc)
        return dt_naive.replace(tzinfo=ZoneInfo(floating_wall_clock_tz))
    except (ValueError, TypeError, OSError) as e:
        logger.debug("ical datetime parse failed %r: %s", value, e)
        return None


def parse_tito_event_ics(ics_text: str) -> Optional[Dict[str, Any]]:
    """Parse first VEVENT from Ti.to checkout ICS into our event-detail shape."""
    if "BEGIN:VEVENT" not in ics_text:
        return None
    start = ics_text.find("BEGIN:VEVENT")
    end = ics_text.find("END:VEVENT", start)
    if start < 0 or end < 0:
        return None
    block = ics_text[start : end + len("END:VEVENT")]
    lines = unfold_ical_lines(block)
    props: Dict[str, List[Tuple[Dict[str, str], str]]] = {}
    for line in lines:
        if not line or line.startswith("BEGIN:") or line.startswith("END:"):
            continue
        name, params, value = split_ical_content_line(line)
        props.setdefault(name, []).append((params, value))

    def first_val(key: str) -> Optional[str]:
        items = props.get(key)
        if not items:
            return None
        return items[0][1]

    def first_params(key: str) -> Dict[str, str]:
        items = props.get(key)
        if not items:
            return {}
        return items[0][0]

    dtstart = first_val("DTSTART")
    dtend = first_val("DTEND")
    if not dtstart:
        return None

    start_dt = ical_value_to_datetime(dtstart, first_params("DTSTART"))
    end_dt = None
    if dtend:
        end_dt = ical_value_to_datetime(dtend, first_params("DTEND"))

    created_raw = first_val("CREATED")
    modified_raw = first_val("LAST-MODIFIED")
    created_dt = None
    mod_dt = None
    if created_raw:
        created_dt = ical_value_to_datetime(
            created_raw, first_params("CREATED"), assume_utc_if_naive=True
        )
    if modified_raw:
        mod_dt = ical_value_to_datetime(
            modified_raw, first_params("LAST-MODIFIED"), assume_utc_if_naive=True
        )

    status_line = (first_val("STATUS") or "").strip().upper()
    cancelled = status_line == "CANCELLED"

    desc_parts = []
    for params, val in props.get("DESCRIPTION", []):
        desc_parts.append(unescape_ical_text_value(val))
    description = "\n".join(desc_parts).strip()

    loc_parts = [unescape_ical_text_value(v) for _, v in props.get("LOCATION", [])]
    location = "\n".join(loc_parts).strip() if loc_parts else ""

    summary = unescape_ical_text_value((first_val("SUMMARY") or "").strip())

    urls = []
    for key in ("URL",):
        for params_u, val in props.get(key, []):
            v = val.strip()
            if v and v not in urls:
                urls.append(v)

    uid_raw = (first_val("UID") or "").strip()
    geo_line = first_val("GEO")
    lat: Optional[float] = None
    lon: Optional[float] = None
    if geo_line and ";" in geo_line:
        try:
            a, b = geo_line.split(";", 1)
            lat, lon = float(a), float(b)
        except (ValueError, TypeError):
            pass

    org_line = first_val("ORGANIZER") or ""
    organizer_email = ""
    if org_line.lower().startswith("mailto:"):
        organizer_email = org_line[7:].strip()

    detail: Dict[str, Any] = {
        "start_at": start_dt.isoformat() if start_dt else None,
        "end_at": end_dt.isoformat() if end_dt else None,
        "description": description,
        "location": location,
        "cancelled": cancelled,
    }
    if summary:
        detail["title"] = summary
    if uid_raw:
        detail["ical_uid"] = uid_raw
    if created_dt:
        detail["created_at"] = created_dt.isoformat()
    if mod_dt:
        detail["updated_at"] = mod_dt.isoformat()
    if urls:
        detail["url"] = urls[0]
        if len(urls) > 1:
            detail["homepage_url"] = urls[1]
    if organizer_email:
        detail["email_address"] = organizer_email
        detail["reply_to_email"] = organizer_email
    if lat is not None and lon is not None:
        detail["map_latitude"] = lat
        detail["map_longitude"] = lon
        detail["map_link"] = f"https://www.google.com/maps?q={lat},{lon}"

    stamp_raw = first_val("DTSTAMP")
    if stamp_raw:
        stamp_dt = ical_value_to_datetime(
            stamp_raw, first_params("DTSTAMP"), assume_utc_if_naive=True
        )
        if stamp_dt:
            detail["source_dtstamp"] = stamp_dt.isoformat()

    return detail


def row_from_tito_json(obj: Dict[str, Any], bucket: str) -> Optional[Dict[str, Any]]:
    """Normalise one Ti.to checkout JSON event object."""
    if not isinstance(obj, dict):
        return None
    url = (obj.get("url") or "").strip()
    slug = slug_from_tito_url(url)
    title = (obj.get("title") or "").strip()
    if not slug:
        logger.warning("Skipping Ti.to row without slug in URL: %s", url or obj)
        return None
    time_str = (obj.get("time") or "").strip()
    if not time_str and bucket == "unscheduled":
        time_str = "Unscheduled"

    row: Dict[str, Any] = {
        "slug": slug,
        "title": title,
        "location": (obj.get("location") or "").strip(),
        "time": time_str,
        "date_or_range": time_str,
        "banner_url": fix_banner_url((obj.get("banner_url") or "").strip()),
        "url": url or tito_event_public_url(slug),
        "tito_bucket": bucket,
    }
    if not validate_tito_list_row(row):
        return None
    return row


def fetch_tito_event_rows() -> List[Dict[str, Any]]:
    """Download Tesla Owners UK event list from checkout.tito.io JSON."""
    logger.info("Fetching Ti.to event list %s", TITO_CHECKOUT_JSON)
    response = fetch_with_retries(TITO_CHECKOUT_JSON)
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error("Ti.to JSON invalid: %s", e)
        return []

    events_block = data.get("events")
    if not isinstance(events_block, dict):
        logger.error("Ti.to JSON missing events object")
        return []

    upcoming = events_block.get("upcoming") or []
    past = events_block.get("past") or []
    unscheduled = events_block.get("unscheduled") or []

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for bucket, lst in (
        ("upcoming", upcoming),
        ("unscheduled", unscheduled),
        ("past", past),
    ):
        if not isinstance(lst, list):
            continue
        for obj in lst:
            row = row_from_tito_json(obj, bucket)
            if not row:
                continue
            s = row["slug"]
            if s in seen:
                continue
            seen.add(s)
            merged.append(row)

    logger.info("Loaded %d event row(s) from Ti.to", len(merged))
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


def is_tbc_location_value(location: Optional[str]) -> bool:
    """True if location is missing or is a common TBC/TBA placeholder."""
    if location is None:
        return True
    s = str(location).strip()
    if not s:
        return True
    low = s.lower().rstrip(".").replace("\u2014", "-")
    if low in TBC_LOCATION_PHRASES:
        return True
    return False


def display_event_location(location: Optional[str]) -> str:
    """User-facing location line; shows Venue TBC when unknown."""
    if is_tbc_location_value(location):
        return "Venue TBC"
    return str(location).strip()


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


def fetch_event_detail(
    slug: str,
    cache: Dict[str, dict],
    list_event: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch per-event ICS from Ti.to checkout. Uses cache unless list metadata changed."""
    if not slug:
        return None

    if slug in cache:
        cached_body = {k: v for k, v in cache[slug].items() if k != "cached_at"}
        if list_event is not None:
            if event_metadata_fingerprint(list_event) != event_metadata_fingerprint(
                cached_body
            ):
                logger.info("Detail cache stale for %s (listing time/venue changed)", slug)
                del cache[slug]
            else:
                logger.info("Using cached detail for: %s", slug)
                return cached_body
        else:
            logger.info("Using cached detail for: %s", slug)
            return cached_body

    url = f"https://checkout.tito.io/{TITO_ACCOUNT}/{quote(str(slug), safe='')}?format=ics"
    try:
        time.sleep(FETCH_DELAY_SEC)
        response = fetch_with_retries(url)
        event = parse_tito_event_ics(response.text)
        if event and event.get("start_at"):
            cache[slug] = {**event, "cached_at": datetime.datetime.now().isoformat()}
            logger.info("Fetched Ti.to ICS for: %s", slug)
            return event
        logger.warning("Ti.to ICS missing start for: %s", slug)
    except requests.RequestException as e:
        logger.warning("Failed to fetch Ti.to ICS %s: %s", slug, e)
    return None


def fix_banner_url(url: str) -> str:
    """Fix malformed banner URLs with duplicate paths.

    Tesla Owners UK API sometimes returns URLs like:
    https://cdn.com/path/https://cdn.com/path/image.png

    This function extracts the correct URL.
    """
    if not url or not url.startswith("http"):
        return url

    # Check if URL contains duplicate https://
    if url.count("https://") > 1 or url.count("http://") > 1:
        # Extract the last complete URL
        parts = url.split("https://")
        if len(parts) > 2:
            return "https://" + parts[-1]
        parts = url.split("http://")
        if len(parts) > 2:
            return "http://" + parts[-1]

    return url


def merge_event_detail(
    list_event: Dict[str, Any], detail_event: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge Ti.to ICS detail into checkout JSON row. Detail wins for schedule, venue, copy."""
    if not detail_event:
        return list_event
    merged = dict(list_event)
    if detail_event.get("title"):
        merged["title"] = detail_event["title"]
    # Prefer detail page for description (often fuller)
    if detail_event.get("description"):
        merged["description"] = detail_event["description"]
    # Detail page is authoritative for start/end (handles date/time changes, same slug/title)
    if detail_event.get("start_at"):
        merged["start_at"] = detail_event["start_at"]
    if detail_event.get("end_at"):
        merged["end_at"] = detail_event["end_at"]
    # Venue: fill TBC from detail, or replace when the club updates location
    dloc = detail_event.get("location")
    if dloc is not None and str(dloc).strip():
        merged["location"] = str(dloc).strip()
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
        merged["banner_url"] = fix_banner_url(detail_event["banner_url"])
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
    if detail_event.get("ical_uid"):
        merged["ical_uid"] = detail_event["ical_uid"]
    if "cancelled" in detail_event:
        merged["cancelled"] = bool(detail_event["cancelled"])
    if detail_event.get("source_dtstamp"):
        merged["source_dtstamp"] = detail_event["source_dtstamp"]
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
        # fromisoformat handles fractional seconds and offsets (Python 3.11+)
        return datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse datetime %r: %s", iso_str, e)
        return None


def normalize_timestamp_for_fingerprint(iso_str: Optional[str]) -> str:
    """Normalise start/end for stable fingerprinting across list vs detail JSON."""
    if not iso_str:
        return ""
    dt = parse_iso_datetime(iso_str)
    if not dt:
        return str(iso_str).strip()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.isoformat(timespec="seconds")


def event_metadata_fingerprint(event: Dict[str, Any]) -> str:
    """Fingerprint schedule + venue (slug is identity; title excluded).

    Uses Ti.to list `time` when ISO start/end not yet merged from ICS.
    """
    loc = (event.get("location") or "").strip()
    start_fp = normalize_timestamp_for_fingerprint(event.get("start_at"))
    if not start_fp and event.get("time"):
        start_fp = str(event.get("time")).strip()
    end_fp = normalize_timestamp_for_fingerprint(event.get("end_at"))
    canc = "1" if event.get("cancelled") else "0"
    return "\x1f".join([start_fp, end_fp, loc, canc])


def ical_revision_fingerprint(event: Dict[str, Any]) -> str:
    """Vary SEQUENCE when enriched start/end/location/title/cancelled changes."""
    return "\x1f".join(
        [
            normalize_timestamp_for_fingerprint(event.get("start_at")),
            normalize_timestamp_for_fingerprint(event.get("end_at")),
            (event.get("location") or "").strip(),
            (event.get("title") or "").strip(),
            "1" if event.get("cancelled") else "0",
        ]
    )


def build_upcoming_state(future_events: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map stable keys to metadata fingerprints for skip logic."""
    out: Dict[str, str] = {}
    for e in future_events:
        slug = e.get("slug")
        fp = event_metadata_fingerprint(e)
        if slug:
            out[str(slug)] = fp
        else:
            basis = f"{fp}\x1f{e.get('title', '')!s}"
            h = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:26]
            out[f"__noshlug_{h}"] = fp
    return dict(sorted(out.items()))


def load_last_upcoming_state() -> Optional[Dict[str, str]]:
    """Load v2 upcoming fingerprints; None if missing or legacy-only file (forces full run)."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    upcoming = data.get("upcoming")
    if isinstance(upcoming, dict):
        return {str(k): str(v) for k, v in upcoming.items()}
    return None


def save_last_upcoming_state(upcoming: Dict[str, str]) -> None:
    """Persist upcoming metadata fingerprints (v2)."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 2,
                    "upcoming": upcoming,
                    "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Saved state for %d upcoming row(s)", len(upcoming))
    except OSError as e:
        logger.warning("State save failed: %s", e)


def escape_ical_text(text: str) -> str:
    """RFC 5545 TEXT value escaping (backslash, newline, semicolon, comma)."""
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def escape_and_fold_ical_text(text: str, prefix: str = "") -> str:
    """Escape and fold text for iCalendar format per RFC 5545."""
    escaped = escape_ical_text(text)
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
    desc_esc = escape_ical_text(str(description))
    return (
        "BEGIN:VALARM\n"
        "ACTION:DISPLAY\n"
        f"DESCRIPTION:{desc_esc}\n"
        f"{trigger_line}\n"
        "END:VALARM\n"
    )


def _format_ical_datetime(dt: datetime.datetime) -> str:
    """Format datetime for iCal (UTC with Z suffix)."""
    if dt.tzinfo:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_ical_z_datetime(s: str) -> Optional[datetime.datetime]:
    """Parse DTSTAMP-style UTC literal."""
    if not s or len(s) < 15:
        return None
    try:
        return datetime.datetime.strptime(s[:15], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        return None


def load_ical_sequence_state() -> Dict[str, Dict[str, Any]]:
    """slug -> {seq, fp, dtstamp} for RFC 5545 SEQUENCE / stable DTSTAMP."""
    if not os.path.exists(ICAL_SEQUENCE_FILE):
        return {}
    try:
        with open(ICAL_SEQUENCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, dict) and "seq" in v and "fp" in v:
            out[k] = v
    return out


def save_ical_sequence_state(state: Dict[str, Dict[str, Any]]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with open(ICAL_SEQUENCE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info("Saved iCal SEQUENCE state (%d slug(s))", len(state))
    except OSError as e:
        logger.warning("iCal sequence save failed: %s", e)


def write_seo_files(site_base_url: str) -> None:
    """Write sitemap.xml and robots.txt for GitHub Pages."""
    base = site_base_url.rstrip("/")
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{base}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>
  <url><loc>{base}/index.html</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>
  <url><loc>{base}/archive.html</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>
  <url><loc>{base}/tocuk.ics</loc><changefreq>daily</changefreq><priority>0.9</priority></url>
</urlset>
"""
    robots = f"""User-agent: *
Allow: /

Sitemap: {base}/sitemap.xml
"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    try:
        (Path(OUTPUT_DIR) / "sitemap.xml").write_text(sitemap, encoding="utf-8")
        (Path(OUTPUT_DIR) / "robots.txt").write_text(robots, encoding="utf-8")
        logger.info("Wrote sitemap.xml and robots.txt")
    except OSError as e:
        logger.warning("SEO files write failed: %s", e)


def make_ics_event(
    event: Dict[str, Any],
    sequence: int = 0,
    dtstamp_utc: Optional[datetime.datetime] = None,
) -> str:
    """Return an iCalendar VEVENT string for a Tesla Owners UK event.

    Uses UTC DTSTART/DTEND (Z) for reliable Android/iOS subscription sync.
    No VALARM components (alarms off). Cancelled events use STATUS:CANCELLED
    and a CANCELLED: SUMMARY prefix when not already present.
    """
    raw_title = event.get("title", "Untitled Event")
    cancelled = bool(event.get("cancelled"))
    if cancelled:
        t = str(raw_title).strip()
        up = t.upper()
        if up.startswith("CANCELLED:") or up.startswith("CANCELLED :"):
            title = t
        else:
            title = f"CANCELLED: {raw_title}"
    else:
        title = raw_title

    start_at = event.get("start_at")
    end_at = event.get("end_at")
    location = display_event_location(event.get("location"))
    description_raw = event.get("description", "")
    slug = str(event.get("slug", "") or "")
    ti_to_url = (event.get("url") or "").strip() or tito_event_public_url(slug)
    event_url = ti_to_url
    ical_uid_raw = (event.get("ical_uid") or "").strip()
    created_at = event.get("created_at")
    updated_at = event.get("updated_at")
    timezone_name = (event.get("timezone") or "").strip() or "Europe/London"
    date_or_range = event.get("date_or_range", "")
    list_time = (event.get("time") or "").strip()
    tito_bucket = (event.get("tito_bucket") or "").strip()
    source_dtstamp = event.get("source_dtstamp")
    banner_url = event.get("banner_url", "")
    email_address = event.get("email_address", "")
    reply_to_email = event.get("reply_to_email", "")
    ticket_success_message = event.get("ticket_success_message", "")
    registrations_count = event.get("registrations_count")
    tickets_count = event.get("tickets_count")
    releases = event.get("releases") or []

    start_dt = parse_iso_datetime(start_at)
    end_dt = parse_iso_datetime(end_at)

    if not start_dt:
        logger.warning("Skipping event %s: no valid start time", raw_title)
        return ""

    # Use end_dt if valid, otherwise default to start + 1 hour
    if end_dt and end_dt > start_dt:
        end_dt_use = end_dt
    else:
        end_dt_use = start_dt + datetime.timedelta(hours=1)

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
    if end_dt_use.tzinfo is None:
        end_dt_use = end_dt_use.replace(tzinfo=datetime.timezone.utc)
    start_utc = start_dt.astimezone(datetime.timezone.utc)
    end_utc = end_dt_use.astimezone(datetime.timezone.utc)
    start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
    end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")

    # Build description with all available info
    description = strip_html(description_raw)
    desc_parts = [description] if description else []
    if list_time:
        desc_parts.append(f"\nListing: {list_time}")
    if date_or_range and date_or_range != list_time:
        desc_parts.append(f"\nWhen: {date_or_range}")
    if timezone_name:
        desc_parts.append(f"\nWall clock / source TZ: {timezone_name}")
    additional = event.get("additional_info")
    if additional:
        desc_parts.append(f"\n{strip_html(str(additional))}")
    if ticket_success_message:
        desc_parts.append(f"\n{ticket_success_message}")
    if event_url:
        desc_parts.append(f"\nEvent & tickets (Ti.to): {event_url}")
    homepage_url = event.get("homepage_url")
    if homepage_url and homepage_url != event_url:
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
    if tito_bucket:
        desc_parts.append(f"\nSource list: {tito_bucket}")
    if source_dtstamp:
        desc_parts.append(f"\nTi.to DTSTAMP: {source_dtstamp}")
    desc_parts.append(f"\nTesla Owners UK: {TITO_PUBLIC_BASE}")
    desc_parts.append(f"\nClub site: {CLUB_PUBLIC_URL}/")
    description_text = "\n".join(desc_parts)

    if ical_uid_raw and "@" in ical_uid_raw:
        uid = ical_uid_raw
    elif ical_uid_raw:
        uid = f"{ical_uid_raw}@checkout.tito.io"
    elif slug:
        uid = f"{TITO_ACCOUNT}-{slug}@ti.to"
    else:
        safe_t = "".join(c for c in title[:24] if c.isalnum() or c in "-_")
        uid = f"{start_str}-{safe_t}@ti.to"

    if dtstamp_utc is None:
        dtstamp_utc = datetime.datetime.now(datetime.timezone.utc)
    elif dtstamp_utc.tzinfo is None:
        dtstamp_utc = dtstamp_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        dtstamp_utc = dtstamp_utc.astimezone(datetime.timezone.utc)
    dtstamp_str = _format_ical_datetime(dtstamp_utc)

    # Build VEVENT (SUMMARY/LOCATION use TEXT escaping per RFC 5545)
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp_str}",
        f"SEQUENCE:{max(0, int(sequence))}",
        f"DTSTART:{start_str}",
        f"DTEND:{end_str}",
        escape_and_fold_ical_text(title, "SUMMARY:"),
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
        lines.append(escape_and_fold_ical_text(location, "LOCATION:"))

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
    if list_time:
        lines.append(escape_and_fold_ical_text(f"Ti.to listing: {list_time}", "COMMENT:"))
    if tito_bucket:
        lines.append(escape_and_fold_ical_text(f"Ti.to bucket: {tito_bucket}", "COMMENT:"))

    # CATEGORIES
    lines.append("CATEGORIES:Tesla Owners UK,Event")

    # CLASS (PUBLIC for public events)
    lines.append("CLASS:PUBLIC")

    lines.append("TRANSP:OPAQUE")

    # STATUS
    lines.append("STATUS:CANCELLED" if cancelled else "STATUS:CONFIRMED")

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
        lines.append(escape_and_fold_ical_text(location, "RESOURCES:"))

    # X- properties for extra metadata
    if timezone_name:
        lines.append(f"X-TIMEZONE:{timezone_name}")
    if slug:
        lines.append(f"X-TITO-SLUG:{slug}")
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


def categorize_event(event: Dict[str, Any]) -> str:
    """Categorize event by type based on title and description.

    Returns:
        Event category: 'track-day', 'meetup', 'agm', 'exhibition', 'online', 'international'
    """
    title = event.get("title", "").lower()
    description = event.get("description", "").lower()
    location = event.get("location", "").lower()
    combined = f"{title} {description} {location}"

    # Check for specific event types
    if "agm" in title or "annual general meeting" in title:
        return "agm"
    elif "online" in location or "virtual" in combined or "zoom" in combined:
        return "online"
    elif any(word in combined for word in ["track day", "circuit", "racing"]):
        return "track-day"
    elif any(word in combined for word in ["everything electric", "fully charged", "exhibition", "show", "supercharged"]):
        return "exhibition"
    elif any(country in location for country in ["texas", "europe", "austria", "germany", "france", "spain", "italy"]):
        return "international"
    else:
        return "meetup"  # Default category


def get_event_region(event: Dict[str, Any]) -> str:
    """Determine event region based on location.

    Returns:
        Region: 'north', 'south', 'midlands', 'online', 'international', 'tbc', 'other'
    """
    raw = event.get("location", "")
    if is_tbc_location_value(raw):
        return "tbc"

    location = raw.strip().lower()

    if "online" in location or "virtual" in location:
        return "online"

    # International locations
    if any(country in location for country in ["texas", "usa", "europe", "austria", "germany", "france", "spain", "italy", "flachau", "salzburg", "austin"]):
        return "international"

    # Northern England/Scotland
    if any(place in location for place in ["yorkshire", "north", "newcastle", "leeds", "manchester", "liverpool", "scotland", "durham", "sedgefield", "hardwick"]):
        return "north"

    # Southern England
    if any(place in location for place in ["london", "south", "brighton", "southampton", "kent", "surrey", "farnborough"]):
        return "south"

    # Midlands
    if any(place in location for place in ["midlands", "birmingham", "nottingham", "leicester", "derby", "cheltenham", "gloucester"]):
        return "midlands"

    # West
    if any(place in location for place in ["cheltenham", "bristol", "bath", "gloucester"]):
        return "west"

    return "other"


def get_event_status(event: Dict[str, Any]) -> tuple[str, str]:
    """Determine event status and badge emoji.

    Returns:
        (status_text, emoji) tuple
    """
    if event.get("cancelled"):
        return ("Cancelled", "🚫")

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


def generate_json_ld(events: List[Dict[str, Any]]) -> str:
    """Generate JSON-LD structured data for events (Schema.org Event type).

    Returns:
        JSON-LD script tag content for SEO
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    upcoming_events = []

    for e in events:
        start = parse_iso_datetime(e.get("start_at"))
        if not start:
            continue

        start_aware = start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start
        if start_aware < today:
            continue  # Skip past events

        end = parse_iso_datetime(e.get("end_at"))
        location_name = display_event_location(e.get("location"))

        event_data = {
            "@context": "https://schema.org",
            "@type": "Event",
            "name": e.get("title", ""),
            "startDate": start.isoformat(),
            "description": strip_html(e.get("description", ""))[:200],
            "eventStatus": (
                "https://schema.org/EventCancelled"
                if e.get("cancelled")
                else "https://schema.org/EventScheduled"
            ),
            "eventAttendanceMode": "https://schema.org/OfflineEventAttendanceMode",
            "organizer": {
                "@type": "Organization",
                "name": "Tesla Owners UK",
                "url": CLUB_PUBLIC_URL,
            },
        }

        if end:
            event_data["endDate"] = end.isoformat()

        if location_name:
            if "online" in location_name.lower():
                event_data["eventAttendanceMode"] = "https://schema.org/OnlineEventAttendanceMode"
                event_data["location"] = {
                    "@type": "VirtualLocation",
                    "url": CLUB_PUBLIC_URL,
                }
            elif location_name == "Venue TBC":
                event_data["location"] = {
                    "@type": "Place",
                    "name": "Venue TBC",
                }
            else:
                event_data["location"] = {
                    "@type": "Place",
                    "name": location_name,
                    "address": {
                        "@type": "PostalAddress",
                        "addressLocality": location_name.split(",")[0].strip() if "," in location_name else location_name
                    }
                }

        if e.get("banner_url"):
            event_data["image"] = e.get("banner_url")

        slug = e.get("slug")
        pub = (e.get("url") or "").strip() or tito_event_public_url(slug)
        if pub:
            event_data["url"] = pub

        upcoming_events.append(event_data)

    if not upcoming_events:
        return ""

    # Return as JSON-LD script tag
    json_content = json.dumps(upcoming_events, indent=2, ensure_ascii=False)
    return f'<script type="application/ld+json">\n{json_content}\n</script>'


def build_index_html(
    events: List[Dict[str, Any]],
    upcoming_count: Optional[int] = None,
    health_status: Optional[Dict[str, Any]] = None,
    site_public_url: Optional[str] = None,
) -> str:
    """Build the index HTML page with calendar link and featured events."""
    site_public_url = (site_public_url or SITE_PUBLIC_URL).rstrip("/")
    og_image_url = f"{site_public_url}/og-image.png"
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
        last_update_attr = html.escape(str(last_update), quote=True)
        message_esc = html.escape(str(message), quote=False)
        error_esc = html.escape(str(error), quote=False) if error else ""

        health_html = f'''
    <div class="health-status {status_class}">
      <div class="health-icon">{status_emoji}</div>
      <div class="health-info">
        <div class="health-main">Last updated: <span id="healthTimestamp" data-timestamp="{last_update_attr}">{time_ago}</span></div>
        <div class="health-message">{message_esc}</div>
        {f'<div class="health-error">Error: {error_esc}</div>' if error_esc else ''}
      </div>
    </div>'''

    # Main grid: upcoming only (past events live on archive.html)
    featured_html = ""
    events_json_data = []
    stats = {
        "total": 0,
        "next_event": None,
        "days_until": None,
        "past_events": 0,
        "next_event_date": "",
    }

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

        # Calculate stats for dashboard (past count includes full scraped list)
        stats = {
            "total": len(upcoming_first),
            "next_event": None,
            "days_until": None,
            "past_events": len(past),
            "next_event_date": "",
        }

        if upcoming_first:
            # Find next event
            next_event = min(upcoming_first, key=lambda e: parse_iso_datetime(e.get("start_at")) or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc))
            stats["next_event"] = next_event.get("title")
            next_date = parse_iso_datetime(next_event.get("start_at"))
            if next_date:
                days_until = (next_date.replace(tzinfo=datetime.timezone.utc) if next_date.tzinfo is None else next_date) - today
                stats["days_until"] = max(0, days_until.days)
                stats["next_event_date"] = next_date.isoformat()

        # Build event cards with full data for search/filter (upcoming rows only)
        featured_items = []
        for idx, e in enumerate(upcoming_first):
            start = parse_iso_datetime(e.get("start_at"))
            date_str = start.strftime("%d %b %Y") if start else ""
            time_str = start.strftime("%H:%M") if start else ""
            title = e.get("title", "Event")
            location = display_event_location(e.get("location"))
            description = strip_html(e.get("description", ""))[:200]
            slug = e.get("slug", "")
            url = (
                (e.get("url") or "").strip() or tito_event_public_url(str(slug) if slug else None)
            )
            banner_url = e.get("banner_url", "")
            status_text, status_emoji = get_event_status(e)

            # Categorize event
            category = categorize_event(e)
            region = get_event_region(e)

            # All cards in this grid are upcoming (TBC dates still listed as upcoming)
            is_upcoming = not start or (start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start) >= today
            status_badge = ""
            if is_upcoming and status_text:
                status_badge = (
                    f'<span class="status-badge">{status_emoji} '
                    f"{html.escape(status_text, quote=False)}</span>"
                )

            # Category badge
            category_labels = {
                "track-day": "🏁 Track Day",
                "meetup": "🤝 Meetup",
                "agm": "📋 AGM",
                "exhibition": "🎪 Exhibition",
                "online": "💻 Online",
                "international": "🌍 International"
            }
            category_badge = f'<span class="category-badge {category}">{category_labels.get(category, category)}</span>'

            # Store event data for JavaScript
            events_json_data.append({
                "title": title,
                "date": date_str,
                "time": time_str,
                "startIso": (e.get("start_at") or ""),
                "location": location,
                "description": description,
                "url": url,
                "upcoming": is_upcoming,
                "category": category,
                "region": region,
                "banner": banner_url
            })

            # Enhanced event card with optional banner image
            banner_html = ""
            if banner_url and banner_url.startswith("http"):
                safe_bg = str(banner_url).replace("'", "%27")
                banner_html = (
                    f'<div class="event-banner" style="background-image: url(\'{safe_bg}\')"></div>'
                )

            loc_short = location[:30] + ("..." if len(location) > 30 else "")
            featured_items.append(
                f'<div class="featured-event {category}" data-event-idx="{idx}" data-category="{category}" data-region="{region}" onclick="showEventModal({idx})">'
                f'{banner_html}'
                f'<div class="event-content">'
                f'<span class="date">{html.escape(date_str, quote=False)}</span>'
                f'{category_badge}'
                f'<span class="title">{html.escape(title, quote=False)}</span>'
                f'<span class="location-small">📍 {html.escape(loc_short, quote=False)}</span>'
                f'{status_badge}'
                f'</div>'
                f"</div>"
            )
        featured_html = "\n        ".join(featured_items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tesla Owners UK: Events Calendar</title>

  <!-- SEO Meta Tags -->
  <meta name="description" content="Subscribe to Tesla Owners UK events calendar - track days, meetups, AGMs and exhibitions. Never miss an event with automatic updates.">
  <meta name="keywords" content="Tesla, Tesla Owners UK, TOCUK, events, calendar, meetups, track days, AGM, Supercharged, Everything Electric">
  <meta name="author" content="evenwebb">
  <link rel="canonical" href="{site_public_url}/">

  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="{site_public_url}/">
  <meta property="og:title" content="Tesla Owners UK Events Calendar">
  <meta property="og:description" content="Never miss a Tesla Owners UK event - subscribe once, stay updated forever. Track days, meetups, AGMs and exclusive gatherings.">
  <meta property="og:site_name" content="Tesla Owners UK Events">
  <meta property="og:image" content="{og_image_url}">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="Tesla Owners UK Events Calendar">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:url" content="{site_public_url}/">
  <meta name="twitter:title" content="Tesla Owners UK Events Calendar">
  <meta name="twitter:description" content="Never miss a Tesla Owners UK event - subscribe once, stay updated forever">
  <meta name="twitter:image" content="{og_image_url}">
  <meta name="twitter:image:alt" content="Tesla Owners UK Events Calendar">

  <!-- Mobile Theme Color -->
  <meta name="theme-color" content="#00d4ff" media="(prefers-color-scheme: dark)">
  <meta name="theme-color" content="#00d4ff" media="(prefers-color-scheme: light)">

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
      overflow: hidden;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }}
    .featured-event:hover {{
      transform: translateY(-2px);
      border-color: var(--accent);
      box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15);
    }}
    .event-banner {{
      width: 100%;
      height: 120px;
      background-size: cover;
      background-position: center;
      background-color: var(--surface-2);
    }}
    .event-content {{
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      flex: 1;
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
      margin: 0.25rem 0;
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
    .category-badge {{
      font-size: 0.65rem;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 600;
      width: fit-content;
      display: inline-block;
    }}
    .category-badge.track-day {{ background: rgba(255, 59, 48, 0.2); color: #ff3b30; border: 1px solid rgba(255, 59, 48, 0.4); }}
    .category-badge.meetup {{ background: rgba(52, 199, 89, 0.2); color: #34c759; border: 1px solid rgba(52, 199, 89, 0.4); }}
    .category-badge.agm {{ background: rgba(10, 132, 255, 0.2); color: #0a84ff; border: 1px solid rgba(10, 132, 255, 0.4); }}
    .category-badge.exhibition {{ background: rgba(255, 149, 0, 0.2); color: #ff9500; border: 1px solid rgba(255, 149, 0, 0.4); }}
    .category-badge.online {{ background: rgba(94, 92, 230, 0.2); color: #5e5ce6; border: 1px solid rgba(94, 92, 230, 0.4); }}
    .category-badge.international {{ background: rgba(191, 90, 242, 0.2); color: #bf5af2; border: 1px solid rgba(191, 90, 242, 0.4); }}
    .featured-event.hidden {{ display: none; }}
    .stats-dashboard {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin: 2rem 0;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1.25rem;
      text-align: center;
    }}
    .stat-number {{
      font-size: 2.5rem;
      font-weight: 700;
      color: var(--accent);
      line-height: 1;
      margin-bottom: 0.5rem;
    }}
    .stat-label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .quick-add-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      justify-content: center;
      margin: 1.5rem 0;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.75rem 1.5rem;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 0.95rem;
      text-decoration: none;
      border: none;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      font-family: inherit;
    }}
    .btn-primary {{
      background: linear-gradient(135deg, var(--cyan), #a855f7);
      color: var(--bg);
    }}
    .btn-secondary {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
    }}
    .btn:hover {{
      transform: scale(1.02);
      box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
    }}
    .subscribe-hint {{
      text-align: center;
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 1rem;
    }}
    .filter-section {{
      margin: 2rem 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
    }}
    .filter-group {{
      margin-bottom: 1.5rem;
    }}
    .filter-group:last-child {{ margin-bottom: 0; }}
    .filter-label {{
      display: block;
      font-weight: 600;
      font-size: 0.9rem;
      margin-bottom: 0.75rem;
      color: var(--text);
    }}
    .filter-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .filter-chip {{
      padding: 0.5rem 1rem;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 100px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      color: var(--text);
      font-family: inherit;
    }}
    .filter-chip:hover {{
      background: var(--surface);
      border-color: var(--accent);
    }}
    .filter-chip.active {{
      background: var(--accent);
      color: var(--bg);
      border-color: var(--accent);
    }}
    @media (max-width: 640px) {{
      .stats-dashboard {{ grid-template-columns: 1fr 1fr; }}
      .quick-add-buttons {{ flex-direction: column; }}
      .btn {{ width: 100%; }}
    }}
    .health-status {{
      background: transparent;
      border: none;
      border-radius: var(--radius-sm);
      padding: 0.5rem 0;
      margin: 0.75rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      opacity: 0.6;
    }}
    .health-status.success {{ opacity: 0.6; }}
    .health-status.warning {{ opacity: 0.7; }}
    .health-status.error {{ opacity: 0.8; }}
    .health-icon {{ font-size: 0.9rem; }}
    .health-info {{ flex: 1; }}
    .health-main {{ font-weight: 400; }}
    .health-message {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0; }}
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

    .referral-cta {{
      background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(229, 62, 62, 0.1));
      border: 1px solid rgba(0, 212, 255, 0.3);
      border-radius: var(--radius-sm);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .referral-title {{
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--text);
    }}
    .referral-text {{
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
    }}
    .referral-note {{
      color: var(--text-muted);
      margin-bottom: 1rem;
      font-size: 0.75rem;
      opacity: 0.7;
    }}
    .referral-link {{
      display: inline-block;
      background: var(--cyan);
      color: var(--bg);
      padding: 0.75rem 1.5rem;
      border-radius: var(--radius-sm);
      font-weight: 600;
      text-decoration: none !important;
      transition: all 0.2s ease;
    }}
    .referral-link:hover {{
      background: #00b8e6;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }}
  </style>

  <!-- Structured Data (JSON-LD) for SEO -->
  {generate_json_ld(events)}
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="badge">Tesla Owners UK · Live Calendar</div>
      <h1>Tesla Owners UK Events</h1>
      <p class="tagline">Never miss a Tesla Owners UK event: track days, meetups, AGMs, and exclusive gatherings. Subscribe once, stay updated forever.</p>
    </header>
{health_html}
    <!-- Stats Dashboard -->
    <section class="stats-dashboard">
      <div class="stat-card">
        <div class="stat-number">{stats['total']}</div>
        <div class="stat-label">Upcoming Events</div>
      </div>
      <div class="stat-card">
        <div class="stat-number" id="daysUntilNext" data-next-event-date="{stats.get('next_event_date', '')}">{stats['days_until'] if stats['days_until'] is not None else '-'}</div>
        <div class="stat-label">Days Until Next Event</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{stats['past_events']}</div>
        <div class="stat-label">Past Events</div>
      </div>
    </section>

    <section class="card">
      <h2>Subscribe to the calendar</h2>
      <p class="meta">{count} upcoming event{'' if count == 1 else 's'}</p>

      <!-- Quick Add Buttons -->
      <div class="quick-add-buttons">
        <a href="{ics_url}" class="btn btn-primary" id="subscribeBtn">📅 Subscribe</a>
        <a href="{ics_url}" class="btn btn-secondary" download>💾 Download .ics</a>
        <button class="btn btn-secondary" onclick="copyCalendarURL()" title="Copy calendar URL">📋 Copy URL</button>
      </div>

      <p class="subscribe-hint">Choose "Subscribe" for automatic updates, or "Download" for a one-time import</p>
    </section>

    <div class="controls">
      <input type="text" class="search-box" id="searchBox" placeholder="Search events by title or location..." onkeyup="filterEvents()">
      <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark/light mode">🌓</button>
    </div>

    <!-- Filter Chips -->
    <div class="filter-section">
      <div class="filter-group">
        <label class="filter-label">Event Type:</label>
        <div class="filter-chips" id="categoryFilters">
          <button class="filter-chip active" data-category="all" onclick="filterByCategory('all')">All Events</button>
          <button class="filter-chip" data-category="track-day" onclick="filterByCategory('track-day')">🏁 Track Days</button>
          <button class="filter-chip" data-category="meetup" onclick="filterByCategory('meetup')">🤝 Meetups</button>
          <button class="filter-chip" data-category="agm" onclick="filterByCategory('agm')">📋 AGMs</button>
          <button class="filter-chip" data-category="exhibition" onclick="filterByCategory('exhibition')">🎪 Exhibitions</button>
          <button class="filter-chip" data-category="online" onclick="filterByCategory('online')">💻 Online</button>
          <button class="filter-chip" data-category="international" onclick="filterByCategory('international')">🌍 International</button>
        </div>
      </div>

      <div class="filter-group">
        <label class="filter-label">Region:</label>
        <div class="filter-chips" id="regionFilters">
          <button class="filter-chip active" data-region="all" onclick="filterByRegion('all')">All Regions</button>
          <button class="filter-chip" data-region="north" onclick="filterByRegion('north')">⬆️ North</button>
          <button class="filter-chip" data-region="south" onclick="filterByRegion('south')">⬇️ South</button>
          <button class="filter-chip" data-region="midlands" onclick="filterByRegion('midlands')">🏴󠁧󠁢󠁥󠁮󠁧󠁿 Midlands</button>
          <button class="filter-chip" data-region="west" onclick="filterByRegion('west')">⬅️ West</button>
          <button class="filter-chip" data-region="online" onclick="filterByRegion('online')">💻 Online</button>
          <button class="filter-chip" data-region="international" onclick="filterByRegion('international')">🌍 International</button>
          <button class="filter-chip" data-region="tbc" onclick="filterByRegion('tbc')">📌 Venue TBC</button>
          <button class="filter-chip" data-region="other" onclick="filterByRegion('other')">📍 Other UK</button>
        </div>
      </div>
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

      <p class="howto-footer">ℹ️ The calendar updates automatically when new events are added, so no need to re-subscribe!</p>
    </section>

    <footer>
      <div class="referral-cta">
        <p class="referral-title">🚗 Ordering a Tesla?</p>
        <p class="referral-text">Use my referral link and get <strong>650 free Supercharging miles</strong> or <strong>£500 off</strong> your new Tesla!</p>
        <a href="https://ts.la/steven201536" target="_blank" rel="noopener" class="referral-link">Claim Your Tesla Benefits →</a>
      </div>

      <p style="margin-top: 1.5rem;">Created by <a href="https://github.com/evenwebb" target="_blank" rel="noopener">evenwebb</a>, Tesla owner & fan 🔋</p>
      <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.7;">An open source fan-made project. Not affiliated with Tesla Owners UK Limited.</p>
      <p style="margin-top: 0.5rem;">
        <a href="archive.html">📁 Event Archive</a>
        <span aria-hidden="true"> · </span>
        <a href="https://ti.to/teslaownersuk" target="_blank" rel="noopener">🔗 Tesla Owners UK on Ti.to</a>
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

  // Next upcoming start: from embedded eventsData so the counter rolls forward without redeploy.
  const MS_PER_DAY = 1000 * 60 * 60 * 24;

  function nextUpcomingStartIso() {{
    const now = Date.now();
    let best = null;
    let bestT = Infinity;
    for (const ev of eventsData) {{
      if (!ev.upcoming || !ev.startIso) continue;
      const t = new Date(ev.startIso).getTime();
      if (Number.isNaN(t) || t < now) continue;
      if (t < bestT) {{ bestT = t; best = ev.startIso; }}
    }}
    return best;
  }}

  function updateDaysUntil() {{
    const element = document.getElementById('daysUntilNext');
    if (!element) return;

    const nextIso = nextUpcomingStartIso();
    if (!nextIso) {{
      element.textContent = '-';
      element.removeAttribute('data-next-event-date');
      return;
    }}
    element.setAttribute('data-next-event-date', nextIso);

    const eventTime = new Date(nextIso).getTime();
    if (Number.isNaN(eventTime)) {{
      element.textContent = '-';
      return;
    }}

    const diffMs = eventTime - Date.now();
    const diffDays = Math.floor(diffMs / MS_PER_DAY);

    if (diffDays >= 0) {{
      element.textContent = String(diffDays);
    }} else {{
      element.textContent = '-';
    }}
  }}

  updateDaysUntil();
  setInterval(updateDaysUntil, 60 * 1000);
  document.addEventListener('visibilitychange', function() {{
    if (document.visibilityState === 'visible') updateDaysUntil();
  }});

  // Update "Last updated" timestamp dynamically
  function updateHealthTimestamp() {{
    const element = document.getElementById('healthTimestamp');
    if (!element) return;

    const timestamp = element.getAttribute('data-timestamp');
    if (!timestamp) return;

    const updateTime = new Date(timestamp);
    const now = new Date();
    const diffSeconds = Math.floor((now - updateTime) / 1000);

    let timeAgo;
    if (diffSeconds < 60) {{
      timeAgo = 'just now';
    }} else if (diffSeconds < 3600) {{
      const mins = Math.floor(diffSeconds / 60);
      timeAgo = `${{mins}} minute${{mins > 1 ? 's' : ''}} ago`;
    }} else if (diffSeconds < 86400) {{
      const hours = Math.floor(diffSeconds / 3600);
      timeAgo = `${{hours}} hour${{hours > 1 ? 's' : ''}} ago`;
    }} else {{
      const days = Math.floor(diffSeconds / 86400);
      timeAgo = `${{days}} day${{days > 1 ? 's' : ''}} ago`;
    }}

    element.textContent = timeAgo;
  }}

  // Update health timestamp on page load and every minute
  updateHealthTimestamp();
  setInterval(updateHealthTimestamp, 1000 * 60); // Update every minute

  // Filter state
  let activeCategory = 'all';
  let activeRegion = 'all';

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

      const matchesCategory = activeCategory === 'all' || card.dataset.category === activeCategory;
      const matchesRegion = activeRegion === 'all' || card.dataset.region === activeRegion;

      if (matchesSearch && matchesCategory && matchesRegion) {{
        card.classList.remove('hidden');
        visibleCount++;
      }} else {{
        card.classList.add('hidden');
      }}
    }});

    // Update count
    const countSpan = document.getElementById('eventCount');
    if (searchTerm || activeCategory !== 'all' || activeRegion !== 'all') {{
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

  // Copy calendar URL to clipboard
  function copyCalendarURL() {{
    const calendarURL = new URL('{ics_url}', window.location.href).href;
    navigator.clipboard.writeText(calendarURL).then(() => {{
      const btn = event.currentTarget;
      const originalText = btn.textContent;
      btn.textContent = '✓ Copied!';
      btn.style.background = 'rgba(34, 197, 94, 0.2)';
      btn.style.borderColor = 'rgba(34, 197, 94, 0.4)';
      setTimeout(() => {{
        btn.textContent = originalText;
        btn.style.background = '';
        btn.style.borderColor = '';
      }}, 2000);
    }}).catch(err => {{
      console.error('Failed to copy:', err);
      alert('Failed to copy URL. Please copy manually: ' + calendarURL);
    }});
  }}

  // Filter by category
  function filterByCategory(category) {{
    activeCategory = category;

    // Update chip styles
    document.querySelectorAll('.filter-chip[data-category]').forEach(chip => {{
      if (chip.dataset.category === category) {{
        chip.classList.add('active');
      }} else {{
        chip.classList.remove('active');
      }}
    }});

    filterEvents();
  }}

  // Filter by region
  function filterByRegion(region) {{
    activeRegion = region;

    // Update chip styles
    document.querySelectorAll('.filter-chip[data-region]').forEach(chip => {{
      if (chip.dataset.region === region) {{
        chip.classList.add('active');
      }} else {{
        chip.classList.remove('active');
      }}
    }});

    filterEvents();
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


def build_archive_html(
    past_events: List[Dict[str, Any]],
    site_public_url: Optional[str] = None,
) -> str:
    """Build the archive HTML page with past events organized by year/month."""
    site_base = (site_public_url or SITE_PUBLIC_URL).rstrip("/")
    og_image_url = f"{site_base}/og-image.png"
    if not past_events:
        events_html = '<p class="no-events">No past events in archive yet.</p>'
        return build_archive_template(events_html, 0, site_base, og_image_url)

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
            location = display_event_location(event.get("location"))
            date_str = start.strftime("%d %b %Y")
            slug = event.get("slug", "")
            url = (
                (event.get("url") or "").strip()
                or tito_event_public_url(str(slug) if slug else None)
            )

            location_html = (
                f'<span class="location">{html.escape(location, quote=False)}</span>'
                if location
                else ""
            )
            url_esc = html.escape(url, quote=True)
            link_html = (
                f'<a href="{url_esc}" target="_blank" rel="noopener">View details →</a>'
                if url
                else ""
            )

            events_html += f'''
        <div class="archive-event">
          <div class="archive-date">{html.escape(date_str, quote=False)}</div>
          <div class="archive-details">
            <div class="archive-title">{html.escape(title, quote=False)}</div>
            {location_html}
          </div>
          {link_html}
        </div>'''
        events_html += '</div></div>'

    return build_archive_template(events_html, len(past_events), site_base, og_image_url)


def build_archive_template(
    events_html: str,
    event_count: int,
    site_base: str,
    og_image_url: str,
) -> str:
    """Build the archive page template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Archive: Tesla Owners UK Events Calendar</title>

  <!-- SEO Meta Tags -->
  <meta name="description" content="Browse past Tesla Owners UK events - track days, meetups, AGMs and exhibitions. Archive of completed community gatherings.">
  <meta name="keywords" content="Tesla, Tesla Owners UK, TOCUK, events archive, past events, history">
  <meta name="author" content="evenwebb">

  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="{site_base}/archive.html">
  <meta property="og:title" content="Event Archive: Tesla Owners UK">
  <meta property="og:description" content="Browse past Tesla Owners UK events and community gatherings">
  <meta property="og:site_name" content="Tesla Owners UK Events">
  <meta property="og:image" content="{og_image_url}">
  <meta property="og:image:alt" content="Tesla Owners UK Events Calendar">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:url" content="{site_base}/archive.html">
  <meta name="twitter:title" content="Event Archive: Tesla Owners UK">
  <meta name="twitter:description" content="Browse past Tesla Owners UK events and community gatherings">
  <meta name="twitter:image" content="{og_image_url}">
  <meta name="twitter:image:alt" content="Tesla Owners UK Events Calendar">

  <!-- Mobile Theme Color -->
  <meta name="theme-color" content="#00d4ff" media="(prefers-color-scheme: dark)">
  <meta name="theme-color" content="#00d4ff" media="(prefers-color-scheme: light)">

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


def _enriched_upcoming_rows(enriched: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows treated as upcoming for skip-state + dashboard counts (by wall time)."""
    today = datetime.datetime.now(datetime.timezone.utc)
    out: List[Dict[str, Any]] = []
    for e in enriched:
        st = parse_iso_datetime(e.get("start_at"))
        if not st:
            if e.get("tito_bucket") in ("upcoming", "unscheduled"):
                out.append(e)
            continue
        s = st.replace(tzinfo=datetime.timezone.utc) if st.tzinfo is None else st
        if s >= today:
            out.append(e)
    return out


def main() -> None:
    """Main function to scrape events and generate iCal file."""
    try:
        list_rows = fetch_tito_event_rows()
        if not list_rows:
            logger.warning("No events found")
            print("No events found.")
            save_health_status(
                "error",
                0,
                "No events found on Ti.to",
                "Ti.to checkout JSON returned no events",
            )
            return
    except Exception as e:
        logger.error("Failed to fetch or parse events: %s", e)
        save_health_status(
            "error",
            0,
            "Failed to fetch events from Ti.to",
            str(e),
        )
        raise

    past_rows = [r for r in list_rows if r.get("tito_bucket") == "past"]
    future_rows = [r for r in list_rows if r.get("tito_bucket") != "past"]
    all_events = past_rows + future_rows

    # Fetch per-event ICS and merge into list rows (times, description, cancellation, etc.)
    cache = load_cache()
    enriched_events = []
    for event in all_events:
        slug = event.get("slug")
        detail = (
            fetch_event_detail(slug, cache, list_event=event) if slug else None
        )
        enriched_events.append(merge_event_detail(event, detail))
    save_cache(cache)

    enriched_events.sort(
        key=lambda x: parse_iso_datetime(x.get("start_at", ""))
        or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )
    upcoming_enriched = _enriched_upcoming_rows(enriched_events)
    today = datetime.datetime.now(datetime.timezone.utc)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Generate iCal (upcoming only; SEQUENCE/DTSTAMP help clients apply updates to same UID)
    seq_state = load_ical_sequence_state()
    new_seq_state: Dict[str, Dict[str, Any]] = {}
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    event_lines = []
    for event in enriched_events:
        start = parse_iso_datetime(event.get("start_at"))
        if start:
            s = start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start
            if s < today:
                continue

        slug = str(event.get("slug") or "")
        fp = ical_revision_fingerprint(event)
        seq_n = 0
        dts_utc = now_utc
        if slug:
            prev = seq_state.get(slug)
            if not prev:
                seq_n = 0
                dts_utc = now_utc
            elif str(prev.get("fp")) != fp:
                seq_n = int(prev.get("seq", 0)) + 1
                dts_utc = now_utc
            else:
                seq_n = int(prev.get("seq", 0))
                dts_utc = _parse_ical_z_datetime(str(prev.get("dtstamp", ""))) or now_utc

        ical = make_ics_event(event, sequence=seq_n, dtstamp_utc=dts_utc)
        if ical:
            event_lines.append(ical)
            if slug:
                new_seq_state[slug] = {
                    "seq": seq_n,
                    "fp": fp,
                    "dtstamp": _format_ical_datetime(dts_utc),
                }

    save_ical_sequence_state(new_seq_state)

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

    save_last_upcoming_state(build_upcoming_state(upcoming_enriched))

    # Save health status
    save_health_status(
        "success",
        len(event_lines),
        f"Successfully processed {len(event_lines)} iCal event(s) ({len(upcoming_enriched)} upcoming on site)",
    )

    # Generate index with health status
    health_status = load_health_status()
    index_path = Path(OUTPUT_DIR) / "index.html"
    index_path.write_text(
        build_index_html(
            enriched_events,
            upcoming_count=len(upcoming_enriched),
            health_status=health_status,
            site_public_url=SITE_PUBLIC_URL,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", index_path)

    write_seo_files(SITE_PUBLIC_URL)

    # Generate archive page with past events
    past_enriched = []
    for e in enriched_events:
        start = parse_iso_datetime(e.get("start_at"))
        if start:
            s = start.replace(tzinfo=datetime.timezone.utc) if start.tzinfo is None else start
            if s < today:
                past_enriched.append(e)

    archive_path = Path(OUTPUT_DIR) / "archive.html"
    archive_path.write_text(
        build_archive_html(past_enriched, site_public_url=SITE_PUBLIC_URL),
        encoding="utf-8",
    )
    logger.info("Wrote %s with %d past events", archive_path, len(past_enriched))

    print(f"\n✓ Created {OUTPUT_DIR}/ with tocuk.ics ({len(event_lines)} events), index.html, and archive.html\n")
    for event in enriched_events:
        start = parse_iso_datetime(event.get("start_at"))
        date_str = start.strftime("%d %B %Y %H:%M") if start else "?"
        print(
            f"  • {event.get('title')}, {date_str} @ {display_event_location(event.get('location'))}"
        )


if __name__ == "__main__":
    main()
