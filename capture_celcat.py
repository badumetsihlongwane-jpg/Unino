#!/usr/bin/env python3
"""
capture_celcat.py

Opens a persistent Google Chrome browser session via Playwright to allow normal sign-in
and navigation on the NWU Celcat timetable system. Automatically intercepts, records,
and saves calendar and resource JSON responses to `celcat_capture/`, and compiles
the captured events into `nwu_mahikeng_timetable.csv` when you close the browser.
"""

import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent / "celcat_capture"
CELCAT_URL = "https://celcat-tst.nwu.ac.za:8446"


def clean_html(text: str) -> str:
    """Removes HTML tags and cleans up whitespace and excess blank lines."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def parse_event_details(event: dict):
    """
    Extracts module_code, class_group, venue_code, venue_name, and clean description.
    Handles NWU Celcat data structures:
      - modules: list of strings (e.g. ['ALDE111_NAS_002'])
      - description: raw text / HTML containing module code, class group, and room
      - sites: room / campus identifiers if available
    """
    raw_desc = event.get("description", "") or ""
    clean_desc = clean_html(raw_desc)
    desc_lines = clean_desc.splitlines() if clean_desc else []

    # 1. Determine candidate module string
    modules_field = event.get("modules")
    candidate_module = ""
    if isinstance(modules_field, list) and modules_field:
        candidate_module = str(modules_field[0]).strip()
    elif isinstance(modules_field, str) and modules_field:
        candidate_module = modules_field.strip()
    elif desc_lines:
        candidate_module = desc_lines[0]

    module_code = ""
    class_group = ""
    if candidate_module:
        # Match standard module code: e.g. ALDE111, CMPG111, ACCC112 (accounting for trailing _, spaces, or end)
        m_code = re.search(r"(?<![A-Za-z0-9])([A-Z]{4}\s*\d{3,4}[A-Z]?)(?:_|\s|$|[^A-Za-z0-9])", candidate_module)
        if m_code:
            module_code = m_code.group(1).replace(" ", "")

        # Extract class group
        if "_" in candidate_module:
            parts = candidate_module.split("_", 1)
            class_group = parts[1]
        else:
            class_group = candidate_module

    # 2. Extract venue code and venue name
    venue = ""
    # Check description lines for venue format like 'MC-A5_105 [180]' or campus prefixes 'MC-', 'PC-', 'VC-'
    for line in desc_lines:
        # Venue with capacity in brackets: e.g. MC-A5_105 [180]
        m_cap = re.search(r"([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\s*\[\d+\]", line)
        if m_cap:
            venue = m_cap.group(1).strip()
            break
        # Or campus venue prefix
        m_prefix = re.search(r"\b((?:MC|PC|VC)-[A-Za-z0-9_-]+)", line)
        if m_prefix:
            venue = m_prefix.group(1).strip()
            break

    # Fallback to 'sites' field if not found in description
    if not venue:
        sites_field = event.get("sites")
        if isinstance(sites_field, list) and sites_field:
            venue = str(sites_field[0]).strip()
        elif isinstance(sites_field, str) and sites_field:
            venue = sites_field.strip()

    venue_code = venue
    venue_name = venue

    return module_code, class_group, venue_code, venue_name, clean_desc


def normalise_event(event: dict, verified_at: str) -> dict:
    """Normalises raw Celcat event JSON into a clean, flat timetable record."""
    module_code, class_group, venue_code, venue_name, description = parse_event_details(event)
    start = event.get("start", "")
    end = event.get("end", "")

    return {
        "celcat_event_id": str(event.get("id", "")),
        "module_code": module_code,
        "class_group": class_group,
        "start_datetime": start,
        "end_datetime": end,
        "timezone": "Africa/Johannesburg",
        "venue_code": venue_code,
        "venue_name": venue_name,
        "description": description,
        "last_verified": verified_at,
    }


def find_events(data) -> list:
    """Extracts a list of event objects from raw response payload."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "data", "items", "calendarData", "rows"):
            val = data.get(key)
            if isinstance(val, list):
                return val
        if "id" in data:
            return [data]
    return []


def export_csv():
    """Compiles all captured GetCalendarData JSON files into nwu_mahikeng_timetable.csv."""
    rows_by_id = {}
    total_files = 0
    calendar_files = 0

    for file in sorted(OUT.glob("*.json")):
        total_files += 1
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # Only event responses are expected to contain schedule entries.
        if "getcalendardata" not in record.get("url", "").lower():
            continue

        calendar_files += 1
        data = record.get("response", {})
        if not isinstance(data, (dict, list)):
            continue

        for event in find_events(data):
            row = normalise_event(event, record.get("captured_at", ""))
            event_id = row["celcat_event_id"]
            if not event_id:
                continue

            # Repeated appearances of the same source event are deduplicated.
            previous = rows_by_id.get(event_id)
            if previous is None or row["last_verified"] > previous["last_verified"]:
                rows_by_id[event_id] = row

    columns = [
        "celcat_event_id",
        "module_code",
        "class_group",
        "start_datetime",
        "end_datetime",
        "timezone",
        "venue_code",
        "venue_name",
        "description",
        "last_verified",
    ]
    output = OUT / "nwu_mahikeng_timetable.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_by_id.values())

    print("\n" + "=" * 60)
    print("CSV EXPORT SUMMARY")
    print(f"- Total JSON files scanned: {total_files}")
    print(f"- Calendar response files:  {calendar_files}")
    print(f"- Unique events exported:   {len(rows_by_id)}")
    print(f"- Output file:              {output}")
    print("=" * 60)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = set()

    async with async_playwright() as p:
        # Separate browser profile; you log in yourself.
        # Launch persistent context using installed Google Chrome channel (handles macOS 12 compatibility).
        launch_kwargs = {
            "user_data_dir": str(OUT / "browser_profile"),
            "headless": False,
            "viewport": {"width": 1366, "height": 820},
        }
        try:
            context = await p.chromium.launch_persistent_context(channel="chrome", **launch_kwargs)
        except Exception as e:
            print(f"[!] Launching via Chrome channel failed ({e}), falling back to default chromium.")
            context = await p.chromium.launch_persistent_context(**launch_kwargs)

        page = context.pages[0] if context.pages else await context.new_page()

        async def handle_response(response):
            url = response.url.lower()
            target_keywords = [
                "getcalendardata",
                "getsidebaresource",
                "getsidebarevent",
                "readresourcelistitems",
            ]
            if not any(k in url for k in target_keywords):
                return

            try:
                text = await response.text()
                data = json.loads(text)
            except Exception:
                return

            captured_at = datetime.now(timezone.utc).isoformat()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            record = {
                "captured_at": captured_at,
                "url": response.url,
                "status": response.status,
                "request_method": response.request.method,
                "request_post_data": response.request.post_data,
                "response": data,
            }

            out_file = OUT / f"{timestamp}.json"
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

            endpoint_name = response.url.split("?")[0].split("/")[-1]
            events_count = len(find_events(data)) if "getcalendardata" in url else 0
            count_str = f" ({events_count} events)" if events_count else ""
            print(f"[✓] Captured: {endpoint_name}{count_str} -> {out_file.name}")

        def on_response(response):
            task = asyncio.create_task(handle_response(response))
            pending.add(task)
            task.add_done_callback(pending.discard)

        context.on("response", on_response)

        print("\n" + "=" * 60)
        print("NWU CELCAT RECORDER ACTIVE")
        print(f"Opening: {CELCAT_URL}")
        print("Navigate Celcat, select modules/groups, or change views.")
        print("All schedule & resource responses are captured automatically.")
        print("To finish, CLOSE THE BROWSER WINDOW.")
        print("=" * 60 + "\n")

        try:
            await page.goto(CELCAT_URL, wait_until="domcontentloaded")
        except Exception as e:
            print(f"Page navigation notice: {e}")

        # Keep running until the user closes all pages or browser window
        try:
            while context.pages and not all(p.is_closed() for p in context.pages):
                await asyncio.sleep(0.5)
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\nSession interrupted by user.")
        finally:
            if pending:
                print("Flushing pending captures...")
                await asyncio.gather(*pending, return_exceptions=True)
            try:
                await context.close()
            except Exception:
                pass

    export_csv()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        export_csv()
