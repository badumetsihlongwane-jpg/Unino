#!/usr/bin/env python3
"""
scrape_all_timetable.py

Automated NWU Celcat Timetable Harvester.
Automatically enumerates and iterates through all resources (rooms, sections, or groups)
across any NWU campus, fetches the full schedule, and generates clean deduplicated CSVs.

Supported Campuses:
  - Mahikeng (MC):        https://celcat-tst.nwu.ac.za:8446
  - Potchefstroom (PC):   https://celcat-tst.nwu.ac.za:8444
  - Vanderbijlpark (VC):  https://celcat-tst.nwu.ac.za:8445

Usage:
  # Fast automated harvest by rooms (all 91 rooms, full academic year):
  python3 scrape_all_timetable.py

  # Harvest by course/module sections (all 1373 sections):
  python3 scrape_all_timetable.py --by sections

  # Harvest both rooms and sections for 100% coverage:
  python3 scrape_all_timetable.py --by all

  # Harvest specific campus:
  python3 scrape_all_timetable.py --campus potchefstroom
  python3 scrape_all_timetable.py --campus vaal

  # Harvest specific semester or date range:
  python3 scrape_all_timetable.py --semester 1
  python3 scrape_all_timetable.py --start 2026-02-01 --end 2026-06-30
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Set, Tuple, Any

CAMPUS_CONFIGS = {
    "mahikeng": {
        "name": "Mahikeng Campus",
        "url": "https://celcat-tst.nwu.ac.za:8446",
        "code": "MC",
        "default_csv": "nwu_mahikeng_timetable.csv",
    },
    "potchefstroom": {
        "name": "Potchefstroom Campus",
        "url": "https://celcat-tst.nwu.ac.za:8444",
        "code": "PC",
        "default_csv": "nwu_potchefstroom_timetable.csv",
    },
    "vaal": {
        "name": "Vanderbijlpark Campus",
        "url": "https://celcat-tst.nwu.ac.za:8445",
        "code": "VC",
        "default_csv": "nwu_vaal_timetable.csv",
    },
}

RESOURCE_TYPES = {
    "sections": 100,  # Course / module sections (e.g. ACCC112_SME_001)
    "staff": 101,     # Lecturers / professors
    "rooms": 102,     # Lecture rooms & laboratories (e.g. MC-A5_105)
    "groups": 103,    # Curriculum / class groups (e.g. ALDE111, 2DJH06...)
}


def create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def parse_event_details(event: dict) -> Tuple[str, str, str, str, str]:
    raw_desc = event.get("description", "") or ""
    clean_desc = clean_html(raw_desc)
    desc_lines = clean_desc.splitlines() if clean_desc else []

    # 1. Module & Group extraction
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
        m_code = re.search(r"(?<![A-Za-z0-9])([A-Z]{4}\s*\d{3,4}[A-Z]?)(?:_|\s|$|[^A-Za-z0-9])", candidate_module)
        if m_code:
            module_code = m_code.group(1).replace(" ", "")

        if "_" in candidate_module:
            parts = candidate_module.split("_", 1)
            class_group = parts[1]
        else:
            class_group = candidate_module

    # 2. Venue extraction
    venue = ""
    for line in desc_lines:
        m_cap = re.search(r"([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\s*\[\d+\]", line)
        if m_cap:
            venue = m_cap.group(1).strip()
            break
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
    module_code, class_group, venue_code, venue_name, description = parse_event_details(event)
    return {
        "celcat_event_id": str(event.get("id", "")),
        "module_code": module_code,
        "class_group": class_group,
        "start_datetime": event.get("start", ""),
        "end_datetime": event.get("end", ""),
        "timezone": "Africa/Johannesburg",
        "venue_code": venue_code,
        "venue_name": venue_name,
        "description": description,
        "last_verified": verified_at,
    }


def fetch_resources(base_url: str, res_type_code: int, ctx: ssl.SSLContext) -> List[dict]:
    """Paginates through ReadResourceListItems to fetch 100% of available resources."""
    items = []
    page = 1
    page_size = 250
    while True:
        url = f"{base_url}/Home/ReadResourceListItems?myResources=false&searchTerm=%25&pageSize={page_size}&pageNumber={page}&resType={res_type_code}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
                results = data.get("results", [])
                items.extend(results)
                total = data.get("total", 0)
                if len(items) >= total or not results:
                    break
                page += 1
        except Exception as e:
            print(f"  [!] Error fetching resource page {page} for resType={res_type_code}: {e}")
            break
    return items


def query_calendar_batch(
    base_url: str,
    res_type_code: int,
    ids: List[str],
    start_date: str,
    end_date: str,
    ctx: ssl.SSLContext,
    retries: int = 3,
) -> List[dict]:
    """Queries GetCalendarData for a batch of federation IDs."""
    params = [
        ("start", start_date),
        ("end", end_date),
        ("resType", str(res_type_code)),
        ("calView", "month"),
        ("colourScheme", "3"),
    ]
    for fid in ids:
        params.append(("federationIds[]", fid))

    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/Home/GetCalendarData",
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=25) as r:
                raw = r.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            if attempt < retries:
                time.sleep(1.0 * attempt)
            else:
                print(f"\n  [!] Batch query failed after {retries} attempts: {e}")
                return []


def run_harvest(
    campus_key: str,
    strategy: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    output_csv: Path,
    batch_size_rooms: int = 10,
    batch_size_sections: int = 25,
    request_delay: float = 0.25,
):
    cfg = CAMPUS_CONFIGS[campus_key]
    base_url = cfg["url"]
    ctx = create_ssl_context()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"AUTOMATED NWU TIMETABLE HARVESTER")
    print(f"Campus:       {cfg['name']} ({base_url})")
    print(f"Strategy:     {strategy.upper()}")
    print(f"Date Range:   {start_date} -> {end_date}")
    print(f"Capture Dir:  {output_dir}")
    print(f"Output CSV:   {output_csv}")
    print("=" * 65 + "\n")

    modes = []
    if strategy in ("rooms", "all"):
        modes.append(("rooms", RESOURCE_TYPES["rooms"], batch_size_rooms))
    if strategy in ("sections", "all"):
        modes.append(("sections", RESOURCE_TYPES["sections"], batch_size_sections))
    if strategy == "groups":
        modes.append(("groups", RESOURCE_TYPES["groups"], batch_size_sections))

    rows_by_id: Dict[str, dict] = {}
    total_raw_events = 0
    start_time = time.time()

    for mode_name, res_code, bsize in modes:
        print(f"--- Discovering all {mode_name.upper()} on {cfg['name']} ---")
        resources = fetch_resources(base_url, res_code, ctx)
        resource_ids = [str(r.get("id")) for r in resources if r.get("id")]
        print(f"Found {len(resource_ids)} {mode_name}. Starting batched calendar harvesting...\n")

        # Save catalogue for this resource type
        cat_file = output_dir / f"catalogue_{campus_key}_{mode_name}.json"
        cat_file.write_text(json.dumps(resources, indent=2, ensure_ascii=False), encoding="utf-8")

        total_batches = (len(resource_ids) + bsize - 1) // bsize
        for idx in range(0, len(resource_ids), bsize):
            batch_num = (idx // bsize) + 1
            chunk = resource_ids[idx : idx + bsize]

            t0 = time.time()
            events = query_calendar_batch(base_url, res_code, chunk, start_date, end_date, ctx)
            elapsed = time.time() - t0
            captured_at = datetime.now(timezone.utc).isoformat()

            # Save raw batch response to maintain full provenance
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            record = {
                "captured_at": captured_at,
                "url": f"{base_url}/Home/GetCalendarData",
                "campus": campus_key,
                "mode": mode_name,
                "batch": batch_num,
                "federation_ids": chunk,
                "response": events,
            }
            batch_file = output_dir / f"{timestamp}_{campus_key}_{mode_name}_b{batch_num}.json"
            batch_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

            # Normalise and deduplicate
            batch_unique_added = 0
            total_raw_events += len(events)
            for ev in events:
                row = normalise_event(ev, captured_at)
                eid = row["celcat_event_id"]
                if not eid:
                    continue
                if eid not in rows_by_id:
                    batch_unique_added += 1
                rows_by_id[eid] = row

            progress = f"[{batch_num:02d}/{total_batches:02d}]"
            sample_ids = ", ".join(chunk[:3]) + ("..." if len(chunk) > 3 else "")
            print(
                f"{progress} {mode_name.capitalize()} {chunk[0]:<15s} "
                f"-> {len(events):4d} events ({batch_unique_added:3d} new) "
                f"| Total Unique: {len(rows_by_id):5d} | {elapsed:.2f}s"
            )

            if request_delay > 0:
                time.sleep(request_delay)

        print(f"\n[✓] Completed harvesting {mode_name.upper()}.\n")

    # Compile and export CSV
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

    distinct_modules = sorted({r["module_code"] for r in sorted_rows if r["module_code"]})
    distinct_venues = sorted({r["venue_code"] for r in sorted_rows if r["venue_code"]})
    total_time = time.time() - start_time

    print("=" * 65)
    print("HARVEST COMPLETED SUCCESSFULLY")
    print(f"- Time taken:                 {total_time:.1f} seconds")
    print(f"- Total raw events fetched:   {total_raw_events:,}")
    print(f"- Total unique deduplicated:  {len(sorted_rows):,}")
    print(f"- Distinct module codes:      {len(distinct_modules):,}")
    if distinct_modules:
        print(f"  Sample: {', '.join(distinct_modules[:8])}...")
    print(f"- Distinct venues:            {len(distinct_venues):,}")
    print(f"- Output CSV:                 {output_csv.resolve()}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated NWU Celcat Timetable Harvester (iterate through all rooms, modules, and schedules)."
    )
    parser.add_argument(
        "--campus",
        "-c",
        choices=["mahikeng", "potchefstroom", "vaal", "all"],
        default="mahikeng",
        help="Campus to harvest (default: mahikeng)",
    )
    parser.add_argument(
        "--by",
        "-b",
        choices=["rooms", "sections", "groups", "all"],
        default="rooms",
        help="Harvesting strategy: 'rooms' (fastest), 'sections' (by module), or 'all' (default: rooms)",
    )
    parser.add_argument(
        "--start",
        "-s",
        default="2026-02-01",
        help="Start date YYYY-MM-DD (default: 2026-02-01)",
    )
    parser.add_argument(
        "--end",
        "-e",
        default="2026-11-30",
        help="End date YYYY-MM-DD (default: 2026-11-30)",
    )
    parser.add_argument(
        "--semester",
        type=int,
        choices=[1, 2],
        default=None,
        help="Shorthand for semester 1 (Feb-Jun) or semester 2 (Jul-Nov)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=Path,
        default=Path("celcat_capture"),
        help="Directory to save raw JSON captures (default: celcat_capture)",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=None,
        help="Destination CSV file path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Polite delay between batch requests in seconds (default: 0.15)",
    )

    args = parser.parse_args()

    start_date = args.start
    end_date = args.end
    if args.semester == 1:
        start_date = "2026-02-01"
        end_date = "2026-06-30"
    elif args.semester == 2:
        start_date = "2026-07-01"
        end_date = "2026-11-30"

    campuses_to_run = (
        ["mahikeng", "potchefstroom", "vaal"] if args.campus == "all" else [args.campus]
    )

    for ckey in campuses_to_run:
        csv_path = (
            args.output_csv
            if (args.output_csv and len(campuses_to_run) == 1)
            else args.output_dir / CAMPUS_CONFIGS[ckey]["default_csv"]
        )
        run_harvest(
            campus_key=ckey,
            strategy=args.by,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output_dir,
            output_csv=csv_path,
            request_delay=args.delay,
        )


if __name__ == "__main__":
    main()
