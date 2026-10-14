#!/usr/bin/env python3
"""
search_timetable.py

Search and inspect captured NWU Celcat timetable CSV files.
Allows students to view their course schedule, check weekly recurring slots,
filter by class group or venue, and confirm their timetable.

Usage:
  # Quick search for a module code:
  python3 search_timetable.py ALDE111
  python3 search_timetable.py CMPG111

  # Search with specific class group:
  python3 search_timetable.py ALDE111 --group NAS_001

  # Search multiple modules at once:
  python3 search_timetable.py ALDE111 CMPG111 MATH111

  # Interactive search prompt:
  python3 search_timetable.py

  # Filter by venue:
  python3 search_timetable.py --venue MC-A5_105

  # Show all individual dated sessions:
  python3 search_timetable.py ALDE111 --dates

  # Export your personal schedule:
  python3 search_timetable.py ALDE111 CMPG111 --export my_schedule.csv
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

DEFAULT_CSV = Path("celcat_capture/nwu_mahikeng_timetable.csv")
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_timetable(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        # Try finding any csv in celcat_capture
        candidates = list(Path("celcat_capture").glob("*.csv")) if Path("celcat_capture").exists() else []
        if candidates:
            csv_path = candidates[0]
        else:
            print(f"[!] Timetable CSV not found at {csv_path}.")
            print("Run `python3 scrape_all_timetable.py` first to generate the timetable.")
            sys.exit(1)

    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def find_similar_modules(query: str, all_modules: Set[str], limit: int = 5) -> List[str]:
    query = query.upper()
    exact_contains = [m for m in all_modules if query in m]
    if exact_contains:
        return sorted(exact_contains)[:limit]
    # Prefix match
    prefix = [m for m in all_modules if m.startswith(query[:3])]
    return sorted(prefix)[:limit]


def display_results(rows: List[dict], show_dates: bool = False, group_filter: str = None):
    if group_filter:
        g_filt = group_filter.upper()
        rows = [r for r in rows if g_filt in r["class_group"].upper()]

    if not rows:
        print("  No schedule entries matched the filters.")
        return

    # Group into weekly recurring slots
    slots = defaultdict(list)
    for r in rows:
        try:
            st = datetime.fromisoformat(r["start_datetime"])
            et = datetime.fromisoformat(r["end_datetime"])
        except ValueError:
            continue
        day_name = st.strftime("%A")
        time_str = f"{st.strftime('%H:%M')} - {et.strftime('%H:%M')}"
        mod = r["module_code"] or "UNKNOWN"
        grp = r["class_group"] or "-"
        ven = r["venue_code"] or "-"
        key = (mod, day_name, time_str, ven, grp)
        slots[key].append(st.strftime("%Y-%m-%d"))

    # Sort slots: Module -> Day of week -> Time
    sorted_slots = sorted(
        slots.items(),
        key=lambda item: (
            item[0][0],  # Module
            DAYS_ORDER.index(item[0][1]) if item[0][1] in DAYS_ORDER else 99,  # Day
            item[0][2],  # Time
            item[0][4],  # Group
        ),
    )

    print("\n" + "=" * 80)
    print(f"{'WEEKLY RECURRING SCHEDULE':^80}")
    print("=" * 80)
    header = f"{'MODULE':<10s} {'DAY':<10s} {'TIME':<15s} {'GROUP':<14s} {'VENUE':<14s} {'WEEKS':<12s}"
    print(header)
    print("-" * 80)

    current_mod = None
    for (mod, day, time_str, venue, group), dates in sorted_slots:
        if current_mod and current_mod != mod:
            print("-" * 80)
        current_mod = mod

        weeks_str = f"{len(dates)} wks ({dates[0][5:]}..{dates[-1][5:]})"
        print(f"{mod:<10s} {day:<10s} {time_str:<15s} {group:<14s} {venue:<14s} {weeks_str:<12s}")

    print("=" * 80)
    print(f"Total slots: {len(sorted_slots)} | Total individual sessions: {len(rows)}\n")

    if show_dates:
        print("-" * 80)
        print("INDIVIDUAL SESSIONS (Chronological)")
        print("-" * 80)
        chron_rows = sorted(rows, key=lambda r: r["start_datetime"])
        for r in chron_rows:
            st = r["start_datetime"].replace("T", " ")
            et = r["end_datetime"].split("T")[-1]
            print(f"  {st} - {et} | {r['module_code']:<8s} | Group: {r['class_group']:<10s} | Venue: {r['venue_code']}")
        print("-" * 80 + "\n")


def search_courses(
    timetable: List[dict],
    queries: List[str],
    group_filter: str = None,
    venue_filter: str = None,
    show_dates: bool = False,
    export_path: Path = None,
):
    all_modules = {r["module_code"] for r in timetable if r["module_code"]}

    matched_rows = []
    for query in queries:
        q = query.strip().upper()
        if not q:
            continue

        # Check module code match
        mod_matches = [r for r in timetable if r["module_code"].upper() == q]
        if not mod_matches:
            # Check partial match on module or description
            mod_matches = [
                r for r in timetable
                if q in r["module_code"].upper()
                or q in r["class_group"].upper()
                or q in r["description"].upper()
            ]

        if mod_matches:
            matched_rows.extend(mod_matches)
        else:
            print(f"[!] No entries found for '{query}'.")
            suggestions = find_similar_modules(q, all_modules)
            if suggestions:
                print(f"    Did you mean: {', '.join(suggestions)}?")

    # Apply venue filter if provided
    if venue_filter:
        v_filt = venue_filter.upper()
        matched_rows = [r for r in matched_rows if v_filt in r["venue_code"].upper()]

    # Deduplicate matching rows
    unique_rows = {r["celcat_event_id"]: r for r in matched_rows if r.get("celcat_event_id")}.values()
    final_rows = list(unique_rows)

    display_results(final_rows, show_dates=show_dates, group_filter=group_filter)

    if export_path and final_rows:
        export_path.parent.mkdir(parents=True, exist_ok=True)
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
        with export_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(final_rows)
        print(f"[✓] Exported {len(final_rows)} entries to {export_path.resolve()}\n")


def interactive_loop(timetable: List[dict]):
    all_modules = sorted({r["module_code"] for r in timetable if r["module_code"]})
    print("\n" + "=" * 60)
    print("NWU TIMETABLE EXPLORER")
    print(f"Loaded {len(timetable):,} sessions across {len(all_modules):,} modules.")
    print("Type a module code (e.g. 'ALDE111', 'CMPG111'), or 'list', or 'q' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            line = input("Search course/module > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        if line.lower() in ("list", "all"):
            print("\nAvailable Modules (Sample of 30):")
            print("  " + ", ".join(all_modules[:30]) + f" ... (+{len(all_modules) - 30} more)\n")
            continue

        parts = line.split()
        search_courses(timetable, parts)


def main():
    parser = argparse.ArgumentParser(description="Search and view NWU Celcat Timetable schedules.")
    parser.add_argument("modules", nargs="*", help="Module code(s) to search for (e.g. ALDE111 CMPG111)")
    parser.add_argument("--group", "-g", help="Filter by class group (e.g. NAS_001, SME_002)")
    parser.add_argument("--venue", "-v", help="Filter by venue code (e.g. MC-A5_105)")
    parser.add_argument("--dates", "-d", action="store_true", help="Show all individual session dates")
    parser.add_argument("--file", "-f", type=Path, default=DEFAULT_CSV, help="Path to timetable CSV file")
    parser.add_argument("--export", "-e", type=Path, help="Export matched rows to a custom CSV file")
    parser.add_argument("--list", "-l", action="store_true", help="List all available module codes")

    args = parser.parse_args()
    timetable = load_timetable(args.file)

    if args.list:
        all_modules = sorted({r["module_code"] for r in timetable if r["module_code"]})
        print(f"\nTotal modules found: {len(all_modules)}")
        for i in range(0, len(all_modules), 10):
            print("  " + ", ".join(all_modules[i : i + 10]))
        print()
        return

    if not args.modules and not args.venue:
        interactive_loop(timetable)
    else:
        search_courses(
            timetable,
            args.modules or [""],
            group_filter=args.group,
            venue_filter=args.venue,
            show_dates=args.dates,
            export_path=args.export,
        )


if __name__ == "__main__":
    main()
