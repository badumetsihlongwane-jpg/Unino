#!/usr/bin/env python3
"""
build_courses_dataset.py

Harvests the complete NWU Course / Qualification Catalogue and automatically
maps each Degree and Year of Study to its official required and elective modules.

Generates:
  1. celcat_capture/nwu_courses_catalogue.json: Full structured course tree
  2. celcat_capture/nwu_courses_catalogue.csv:  Flat relational table for DB / Appwrite
  3. celcat_capture/nwu_courses_quick_suggest.json: Lightweight client-side lookup for Unino signup

Usage:
  python3 build_courses_dataset.py
  python3 build_courses_dataset.py --campus mahikeng
  python3 build_courses_dataset.py --campus potchefstroom
"""

import argparse
from collections import defaultdict
import concurrent.futures
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Set, Tuple, Any

CAMPUS_URLS = {
    "mahikeng": "https://celcat-tst.nwu.ac.za:8446",
    "potchefstroom": "https://celcat-tst.nwu.ac.za:8444",
    "vaal": "https://celcat-tst.nwu.ac.za:8445",
}

FACULTY_MAP = {
    "1D": "Faculty of Humanities",
    "1G": "Faculty of Humanities",
    "1F": "Faculty of Humanities",
    "1E": "Faculty of Humanities",
    "2D": "Faculty of Natural and Agricultural Sciences",
    "2H": "Faculty of Natural and Agricultural Sciences",
    "2X": "Faculty of Natural and Agricultural Sciences",
    "3D": "Faculty of Economic and Management Sciences",
    "5D": "Faculty of Law",
    "6D": "Faculty of Theology",
    "8D": "Faculty of Education",
}


def create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def infer_faculty_and_type(code: str, title: str) -> Tuple[str, str]:
    prefix2 = code[:2].upper()
    faculty = FACULTY_MAP.get(prefix2, "General / Inter-Faculty")

    title_lower = title.lower()
    if "bachelor of arts" in title_lower or title.startswith("BA "):
        deg_type = "Bachelor of Arts"
        if faculty == "General / Inter-Faculty":
            faculty = "Faculty of Humanities"
    elif "bcom" in title_lower or "commerce" in title_lower or "accounting" in title_lower or "badmin" in title_lower:
        deg_type = "BCom / BAdmin"
        if faculty == "General / Inter-Faculty":
            faculty = "Faculty of Economic and Management Sciences"
    elif "bsc" in title_lower or "science" in title_lower or "computing" in title_lower or "agriculture" in title_lower:
        deg_type = "Bachelor of Science (BSc)"
        if faculty == "General / Inter-Faculty":
            faculty = "Faculty of Natural and Agricultural Sciences"
    elif "llb" in title_lower or "law" in title_lower:
        deg_type = "Bachelor of Laws (LLB)"
        faculty = "Faculty of Law"
    elif "bed" in title_lower or "education" in title_lower or "teaching" in title_lower:
        deg_type = "Bachelor of Education (BEd)"
        faculty = "Faculty of Education"
    elif "hons" in title_lower or "honours" in title_lower:
        deg_type = "Honours Degree"
    elif "diploma" in title_lower:
        deg_type = "Diploma / Postgrad Diploma"
    elif "theology" in title_lower or "theol" in title_lower:
        deg_type = "Theology"
        faculty = "Faculty of Theology"
    else:
        deg_type = "Undergraduate Degree"

    return faculty, deg_type


def fetch_all_groups(base_url: str, ctx: ssl.SSLContext) -> List[dict]:
    """Paginates through ReadResourceListItems to fetch all groups (resType=103)."""
    items = []
    page = 1
    page_size = 250
    while True:
        url = f"{base_url}/Home/ReadResourceListItems?myResources=false&searchTerm=%25&pageSize={page_size}&pageNumber={page}&resType=103"
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
            print(f"  [!] Error on page {page}: {e}")
            break
    return items


def fetch_modules_for_cohort(base_url: str, cohort_id: str, date_range: Tuple[str, str], ctx: ssl.SSLContext) -> List[str]:
    """Queries GetCalendarData for a single cohort over a sample month to find active modules."""
    s_date, e_date = date_range
    params = urllib.parse.urlencode({
        "start": s_date,
        "end": e_date,
        "resType": "103",
        "calView": "month",
        "colourScheme": "3",
        "federationIds[]": cohort_id,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/Home/GetCalendarData",
        data=params,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as r:
            events = json.loads(r.read().decode("utf-8"))
            modules = set()
            for ev in events:
                for m in (ev.get("modules") or []):
                    # Extract 4-letter module code
                    m_match = re.search(r"(?<![A-Za-z0-9])([A-Z]{4}\s*\d{3,4}[A-Z]?)(?:_|\s|$|[^A-Za-z0-9])", m)
                    if m_match:
                        modules.add(m_match.group(1).replace(" ", ""))
            return sorted(list(modules))
    except Exception:
        return []


def build_catalogue(campus: str = "mahikeng", max_workers: int = 5, output_dir: Path = Path("celcat_capture")):
    base_url = CAMPUS_URLS.get(campus, CAMPUS_URLS["mahikeng"])
    ctx = create_ssl_context()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("NWU COURSE & MODULE SUGGESTION BUILDER")
    print(f"Campus:       {campus.capitalize()} ({base_url})")
    print(f"Output Dir:   {output_dir}")
    print("=" * 65 + "\n")

    print("[1/3] Enumerating all curriculum groups...")
    raw_groups = fetch_all_groups(base_url, ctx)
    print(f"  Fetched {len(raw_groups)} raw group records from Celcat.\n")

    # Filter out standalone module codes (e.g. ALDE111, CMPG111)
    # Group into courses: (base_title, base_curriculum_code)
    courses = {}
    cohorts_to_query = []

    for g in raw_groups:
        gid = g.get("id", "").strip()
        t = g.get("text", "").strip()

        # Skip module codes
        if re.match(r"^[A-Z]{4}\s*\d{3,4}", gid):
            continue

        m = re.match(r"^([A-Z0-9_-]+)\s*\((.+)\)$", t)
        if not m:
            continue

        code = m.group(1)
        full_title = m.group(2)

        # Extract Year of study (e.g. -Y1, -Y2, -Y3, -Y4)
        ym = re.search(r"-Y(\d)", code)
        year = int(ym.group(1)) if ym else 1

        # Extract Semester (e.g. _S1, _S2)
        sm = re.search(r"_S(\d)", code)
        sem = int(sm.group(1)) if sm else 1

        # Clean title
        clean_title = re.sub(r"\s*-\s*Y\d(?:_S\d(?:_\d)?)?$", "", full_title).strip()
        base_code = code.split("-Y")[0] if "-Y" in code else code

        if clean_title not in courses:
            faculty, deg_type = infer_faculty_and_type(base_code, clean_title)
            courses[clean_title] = {
                "course_id": base_code,
                "course_name": clean_title,
                "faculty": faculty,
                "qualification": deg_type,
                "years": {},
            }

        year_str = f"Year {year}"
        if year_str not in courses[clean_title]["years"]:
            courses[clean_title]["years"][year_str] = {
                "semester_1": set(),
                "semester_2": set(),
                "all_modules": set(),
            }

        # Select representative cohort ID (prefer _S1 or base -Y1 over sub-splits like _S1_2)
        if not code.endswith(("_2", "_3", "_4")):
            cohorts_to_query.append({
                "course_title": clean_title,
                "year_str": year_str,
                "sem": sem,
                "cohort_id": gid,
            })

    print(f"[2/3] Identified {len(courses)} distinct degree programmes across {len(cohorts_to_query)} cohorts.")
    print("  Mapping modules per degree & year of study (concurrent requests)...")

    # Dates to sample: Semester 1 (March 2026), Semester 2 (August 2026)
    s1_range = ("2026-03-01", "2026-03-28")
    s2_range = ("2026-08-01", "2026-08-28")

    def worker(item):
        date_range = s1_range if item["sem"] == 1 else s2_range
        mods = fetch_modules_for_cohort(base_url, item["cohort_id"], date_range, ctx)
        return item["course_title"], item["year_str"], item["sem"], mods

    t0 = time.time()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in cohorts_to_query}
        for future in concurrent.futures.as_completed(future_to_item):
            completed += 1
            if completed % 25 == 0 or completed == len(cohorts_to_query):
                print(f"  [{completed:03d}/{len(cohorts_to_query):03d}] Cohorts mapped ({time.time()-t0:.1f}s)...")
            try:
                title, year_str, sem, mods = future.result()
                if title in courses and year_str in courses[title]["years"]:
                    sem_key = f"semester_{sem}"
                    courses[title]["years"][year_str][sem_key].update(mods)
                    courses[title]["years"][year_str]["all_modules"].update(mods)
            except Exception:
                pass

    print(f"\n[3/3] Assembling exports...")

    # 1. Structured JSON
    structured_export = []
    quick_suggest = {}
    csv_rows = []

    for title, cdata in sorted(courses.items()):
        years_dict = {}
        quick_years = {}

        for y_str, y_data in sorted(cdata["years"].items()):
            s1_list = sorted(list(y_data["semester_1"]))
            s2_list = sorted(list(y_data["semester_2"]))
            all_list = sorted(list(y_data["all_modules"]))

            years_dict[y_str] = {
                "semester_1": s1_list,
                "semester_2": s2_list,
                "modules": all_list,
            }
            quick_years[y_str] = all_list

            # Flat CSV rows
            for m in s1_list:
                csv_rows.append({
                    "course_id": cdata["course_id"],
                    "course_name": title,
                    "qualification": cdata["qualification"],
                    "faculty": cdata["faculty"],
                    "year_of_study": y_str,
                    "semester": "Semester 1",
                    "module_code": m,
                })
            for m in s2_list:
                csv_rows.append({
                    "course_id": cdata["course_id"],
                    "course_name": title,
                    "qualification": cdata["qualification"],
                    "faculty": cdata["faculty"],
                    "year_of_study": y_str,
                    "semester": "Semester 2",
                    "module_code": m,
                })

        structured_export.append({
            "course_id": cdata["course_id"],
            "course_name": title,
            "faculty": cdata["faculty"],
            "qualification": cdata["qualification"],
            "years": years_dict,
        })

        quick_suggest[title] = quick_years

    # Output JSONs
    json_path = output_dir / "nwu_courses_catalogue.json"
    json_path.write_text(json.dumps(structured_export, indent=2, ensure_ascii=False), encoding="utf-8")

    quick_path = output_dir / "nwu_courses_quick_suggest.json"
    quick_path.write_text(json.dumps(quick_suggest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Output CSV
    csv_path = output_dir / "nwu_courses_catalogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["course_id", "course_name", "qualification", "faculty", "year_of_study", "semester", "module_code"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 65)
    print("DATASET GENERATION COMPLETE")
    print(f"- Total Degree Programmes:   {len(courses):,}")
    print(f"- Total Relational Mappings: {len(csv_rows):,}")
    print(f"- Structured JSON:           {json_path.resolve()}")
    print(f"- Quick Suggest JSON:        {quick_path.resolve()}")
    print(f"- Relational Table (CSV):    {csv_path.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build complete NWU course & module suggestion dataset.")
    parser.add_argument("--campus", "-c", default="mahikeng", choices=["mahikeng", "potchefstroom", "vaal"])
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of concurrent workers (default: 8)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("celcat_capture"))
    args = parser.parse_args()

    build_catalogue(campus=args.campus, max_workers=args.workers, output_dir=args.output_dir)
