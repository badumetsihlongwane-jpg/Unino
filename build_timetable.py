#!/usr/bin/env python3
"""
build_timetable.py

Converts captured Celcat event JSON files in `celcat_capture/` into a clean,
structured CSV: separating module codes, class groups, times, and venue codes.
Also extracts full resource catalogues (modules, groups, rooms) if present.

Usage:
    python build_timetable.py
    python build_timetable.py --dir celcat_capture --output celcat_capture/nwu_mahikeng_timetable.csv
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple, Any


def clean_html(text: str) -> str:
    """Removes HTML tags and cleans up whitespace and excess blank lines."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def parse_event_details(event: dict) -> Tuple[str, str, str, str, str]:
    """
    Extracts (module_code, class_group, venue_code, venue_name, description).
    Supports multiple NWU formats:
      - Modules array: ['ALDE111_NAS_002']
      - Description: 'ALDE111_NAS_002\\r\\n\\r\\n<br />\\r\\n\\r\\nMC-A5_105 [180]\\r\\n'
      - Sites / rooms array or string
    """
    raw_desc = event.get("description", "") or ""
    clean_desc = clean_html(raw_desc)
    desc_lines = clean_desc.splitlines() if clean_desc else []

    # 1. Module code & Class group extraction
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

    # 2. Venue code & Venue name extraction
    venue = ""
    for line in desc_lines:
        # Capacity format e.g. MC-A5_105 [180] or Room 101 [50]
        m_cap = re.search(r"([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\s*\[\d+\]", line)
        if m_cap:
            venue = m_cap.group(1).strip()
            break
        # Known campus venue prefix e.g. MC-, PC-, VC-
        m_prefix = re.search(r"\b((?:MC|PC|VC)-[A-Za-z0-9_-]+)", line)
        if m_prefix:
            venue = m_prefix.group(1).strip()
            break

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
    """Creates a standardised dictionary for CSV output."""
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


def find_events(data: Any) -> list:
    """Extracts event array from various response structures."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "data", "items", "calendarData", "rows"):
            val = data.get(key)
            if isinstance(val, list):
                return val
        if "id" in data and ("start" in data or "description" in data):
            return [data]
    return []


def process_captures(input_dir: Path, output_csv: Path, catalogue_json: Path = None):
    """Processes all JSON files in input_dir and generates clean CSV output."""
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON capture files found in {input_dir}.")
        print("Run `python capture_celcat.py` first to collect timetable responses.")
        return

    rows_by_id: Dict[str, dict] = {}
    sidebar_events_by_id: Dict[str, dict] = {}
    catalogue_items: Dict[str, dict] = {}
    calendar_files_count = 0
    total_raw_events = 0

    for file in json_files:
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        url = record.get("url", "").lower()
        captured_at = record.get("captured_at", "")
        data = record.get("response", {})

        # 1. Calendar events
        if "getcalendardata" in url:
            calendar_files_count += 1
            events = find_events(data)
            total_raw_events += len(events)
            for event in events:
                row = normalise_event(event, captured_at)
                event_id = row["celcat_event_id"]
                if not event_id:
                    continue

                prev = rows_by_id.get(event_id)
                if prev is None or row["last_verified"] > prev["last_verified"]:
                    rows_by_id[event_id] = row

        # 2. Sidebar event details
        elif "getsidebarevent" in url:
            if isinstance(data, dict):
                elements = data.get("elements", [])
                event_id = record.get("request_post_data", "")
                m = re.search(r"eventId=([^&]+)", event_id)
                eid = m.group(1) if m else ""
                if eid:
                    sidebar_events_by_id[eid] = {
                        el.get("label"): el.get("content") for el in elements if el.get("label")
                    }

        # 3. Discovered Catalogue (Groups / Modules / Sections)
        elif "readresourcelistitems" in url:
            if isinstance(data, dict):
                results = data.get("results", [])
                for item in results:
                    item_id = item.get("id")
                    if item_id:
                        catalogue_items[item_id] = item

    # Enrich events if sidebar data was captured
    for event_id, sb in sidebar_events_by_id.items():
        if event_id in rows_by_id:
            row = rows_by_id[event_id]
            if not row["venue_code"] and "Room" in sb:
                room_text = sb["Room"]
                m_cap = re.search(r"([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\s*\[\d+\]", room_text)
                row["venue_code"] = m_cap.group(1) if m_cap else room_text
                row["venue_name"] = room_text
            if not row["module_code"] and "Module" in sb:
                mod_text = sb["Module"]
                m_code = re.search(r"\b([A-Z]{4}\s*\d{3,4}[A-Z]?)\b", mod_text)
                if m_code:
                    row["module_code"] = m_code.group(1).replace(" ", "")

    # Export CSV
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows_by_id.values(),
        key=lambda r: (r["module_code"], r["start_datetime"], r["class_group"]),
    )

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted_rows)

    # Export catalogue if items found or requested
    if catalogue_json and catalogue_items:
        catalogue_json.parent.mkdir(parents=True, exist_ok=True)
        catalogue_json.write_text(
            json.dumps(list(catalogue_items.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Calculate statistics
    distinct_modules = sorted({r["module_code"] for r in sorted_rows if r["module_code"]})
    distinct_venues = sorted({r["venue_code"] for r in sorted_rows if r["venue_code"]})
    start_dates = [r["start_datetime"] for r in sorted_rows if r["start_datetime"]]

    print("\n" + "=" * 60)
    print("TIMETABLE BUILD SUMMARY")
    print(f"- Total JSON files analyzed:  {len(json_files)}")
    print(f"- Calendar response files:    {calendar_files_count}")
    print(f"- Total raw events observed:  {total_raw_events}")
    print(f"- Unique events deduplicated: {len(sorted_rows)}")
    print(f"- Unique module codes found:  {len(distinct_modules)}")
    if distinct_modules:
        sample_mods = distinct_modules[:10]
        suffix = f" ... (+{len(distinct_modules) - 10} more)" if len(distinct_modules) > 10 else ""
        print(f"  Sample modules: {', '.join(sample_mods)}{suffix}")
    print(f"- Unique venues found:        {len(distinct_venues)}")
    if start_dates:
        print(f"- Date span:                  {min(start_dates)} to {max(start_dates)}")
    print(f"- CSV output path:            {output_csv.resolve()}")
    if catalogue_items:
        print(f"- Catalogue items extracted:  {len(catalogue_items)}")
        if catalogue_json:
            print(f"- Catalogue JSON:             {catalogue_json.resolve()}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build clean NWU timetable CSV from captured Celcat JSON.")
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=Path("celcat_capture"),
        help="Directory with captured JSON responses (default: celcat_capture)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output CSV (default: <dir>/nwu_mahikeng_timetable.csv)",
    )
    parser.add_argument(
        "--catalogue",
        "-c",
        type=Path,
        default=None,
        help="Path to save catalogue JSON if resource items were captured",
    )
    args = parser.parse_args()

    input_dir = args.dir.resolve()
    output_csv = args.output.resolve() if args.output else input_dir / "nwu_mahikeng_timetable.csv"
    catalogue_json = args.catalogue.resolve() if args.catalogue else input_dir / "catalogue.json"

    process_captures(input_dir, output_csv, catalogue_json)


if __name__ == "__main__":
    main()
