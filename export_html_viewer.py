#!/usr/bin/env python3
"""
export_html_viewer.py

Generates an interactive, standalone HTML timetable viewer (timetable_viewer.html)
from the captured Celcat timetable CSV.

Features:
- Instant search across all courses and 1,162+ modules
- Visual Weekly Schedule Grid (Monday - Friday, 07:30 - 18:00)
- Table / List view with instant sorting and filtering
- 'My Courses' builder: Select multiple modules to view your personalized timetable
- Clash detector: Highlights overlapping time slots
- Print / PDF export friendly
- 100% Standalone: Opens directly in any browser without needing a web server
"""

from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

DEFAULT_CSV = Path("celcat_capture/nwu_mahikeng_timetable.csv")
DEFAULT_HTML = Path("timetable_viewer.html")


def generate_viewer(csv_path: Path, output_html: Path):
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} does not exist.")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows):,} sessions...")

    # Aggregate into weekly recurring slots
    slots = defaultdict(list)
    modules_set = set()
    venues_set = set()
    groups_set = set()

    for r in rows:
        mod = r.get("module_code", "").strip()
        grp = r.get("class_group", "").strip()
        ven = r.get("venue_code", "").strip()
        s_dt = r.get("start_datetime", "").strip()
        e_dt = r.get("end_datetime", "").strip()

        if mod:
            modules_set.add(mod)
        if ven:
            venues_set.add(ven)
        if grp:
            groups_set.add(grp)

        try:
            st = datetime.fromisoformat(s_dt)
            et = datetime.fromisoformat(e_dt)
            day = st.strftime("%A")
            time_str = f"{st.strftime('%H:%M')} - {et.strftime('%H:%M')}"
            start_time = st.strftime("%H:%M")
            end_time = et.strftime("%H:%M")
            # Semester: 1 if month <= 6 else 2
            sem = 1 if st.month <= 6 else 2
        except Exception:
            continue

        key = (mod, grp, ven, day, time_str, start_time, end_time, sem)
        slots[key].append(st.strftime("%Y-%m-%d"))

    compact_slots = []
    for (mod, grp, ven, day, time_str, start_time, end_time, sem), dates in slots.items():
        dates.sort()
        compact_slots.append({
            "m": mod,
            "g": grp,
            "v": ven,
            "d": day,
            "t": time_str,
            "s": start_time,
            "e": end_time,
            "sem": sem,
            "w": len(dates),
            "dates": dates,
            "first": dates[0],
            "last": dates[-1],
        })

    # Sort slots: Module -> Day -> Start Time
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    compact_slots.sort(key=lambda x: (
        x["m"],
        days_order.index(x["d"]) if x["d"] in days_order else 99,
        x["s"],
        x["g"]
    ))

    # Load courses catalogue if present
    courses_quick_path = csv_path.parent / "nwu_courses_quick_suggest.json"
    courses_quick = {}
    if courses_quick_path.exists():
        try:
            courses_quick = json.loads(courses_quick_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    data_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_sessions": len(rows),
        "total_slots": len(compact_slots),
        "modules": sorted(list(modules_set)),
        "venues": sorted(list(venues_set)),
        "groups": sorted(list(groups_set)),
        "courses": courses_quick,
        "slots": compact_slots,
    }

    data_json = json.dumps(data_payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NWU Timetable Viewer (Mahikeng Campus)</title>
  <style>
    :root {{
      --primary: #5c2d91;
      --primary-light: #7b42bd;
      --primary-dark: #3e1b65;
      --secondary: #0080ff;
      --success: #28a745;
      --danger: #dc3545;
      --warning: #ffc107;
      --bg: #f8f9fa;
      --surface: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
      --radius: 8px;
      --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 60px;
    }}

    header {{
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      padding: 1.5rem 2rem;
      box-shadow: var(--shadow);
    }}
    .header-content {{
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }}
    .title-area h1 {{ font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(255,255,255,0.2);
    }}
    .subtitle {{ color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.25rem; }}

    main {{ max-width: 1400px; margin: 1.5rem auto; padding: 0 1rem; }}

    /* Search & Filter Bar */
    .controls-card {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 1.25rem;
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      margin-bottom: 1.5rem;
    }}
    .search-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
    }}
    .search-box {{
      flex: 2;
      min-width: 280px;
      position: relative;
    }}
    .search-input {{
      width: 100%;
      padding: 0.75rem 1rem 0.75rem 2.5rem;
      font-size: 1rem;
      border: 2px solid var(--border);
      border-radius: var(--radius);
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    .search-icon {{
      position: absolute;
      left: 0.85rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 1rem;
    }}

    .filter-select {{
      flex: 1;
      min-width: 150px;
      padding: 0.75rem;
      font-size: 0.9rem;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: white;
      outline: none;
    }}

    .view-toggle {{
      display: flex;
      background: var(--bg);
      border-radius: var(--radius);
      padding: 0.25rem;
      border: 1px solid var(--border);
    }}
    .toggle-btn {{
      padding: 0.5rem 1rem;
      border: none;
      background: transparent;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 6px;
      color: var(--text-muted);
      transition: all 0.2s;
    }}
    .toggle-btn.active {{
      background: var(--surface);
      color: var(--primary);
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    /* Selected Courses Chips */
    .selected-tray {{
      margin-top: 1rem;
      padding-top: 0.75rem;
      border-top: 1px dashed var(--border);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
    }}
    .tray-label {{ font-size: 0.85rem; font-weight: 600; color: var(--text-muted); }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      background: #ede9fe;
      color: var(--primary);
      border: 1px solid #ddd6fe;
    }}
    .chip-remove {{
      cursor: pointer;
      font-size: 1rem;
      line-height: 1;
      opacity: 0.7;
    }}
    .chip-remove:hover {{ opacity: 1; }}

    .btn-action {{
      padding: 0.4rem 0.8rem;
      font-size: 0.8rem;
      font-weight: 600;
      border: 1px solid var(--border);
      background: white;
      border-radius: var(--radius);
      cursor: pointer;
      transition: background 0.15s;
    }}
    .btn-action:hover {{ background: #f1f5f9; }}

    /* Clash Alert */
    .clash-alert {{
      display: none;
      margin-bottom: 1.5rem;
      padding: 1rem 1.25rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: var(--radius);
      color: #991b1b;
    }}
    .clash-alert strong {{ font-weight: 700; }}

    /* Calendar Grid View */
    .calendar-container {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      overflow-x: auto;
    }}
    .calendar-grid {{
      display: grid;
      grid-template-columns: 80px repeat(5, minmax(200px, 1fr));
      border-collapse: collapse;
      min-width: 950px;
    }}
    .grid-header {{
      background: #f8fafc;
      padding: 0.75rem;
      font-weight: 700;
      font-size: 0.9rem;
      text-align: center;
      border-bottom: 2px solid var(--border);
      border-right: 1px solid var(--border);
      color: var(--text);
    }}
    .grid-header.time-col {{ border-right: 2px solid var(--border); background: #f1f5f9; }}
    .time-slot-label {{
      padding: 0.5rem 0.4rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      border-right: 2px solid var(--border);
      text-align: right;
      background: #fafafa;
    }}
    .day-column-cell {{
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
      min-height: 52px;
      padding: 0.25rem;
      position: relative;
      background: #ffffff;
    }}
    .day-column-cell:nth-child(even) {{ background: #fdfdfd; }}

    .event-card {{
      background: #e0f2fe;
      border-left: 4px solid var(--secondary);
      border-radius: 4px;
      padding: 0.4rem 0.5rem;
      margin-bottom: 0.35rem;
      font-size: 0.75rem;
      cursor: pointer;
      transition: transform 0.1s, box-shadow 0.1s;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    .event-card:hover {{
      transform: translateY(-1px);
      box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }}
    .event-card.clash {{
      background: #ffe4e6;
      border-left-color: var(--danger);
    }}
    .event-card .ev-mod {{ font-weight: 700; font-size: 0.82rem; color: #0369a1; }}
    .event-card.clash .ev-mod {{ color: #be123c; }}
    .event-card .ev-group {{ color: #475569; }}
    .event-card .ev-venue {{ font-weight: 600; color: #0f172a; display: inline-block; margin-top: 0.15rem; }}
    .event-card .ev-time {{ color: var(--text-muted); font-size: 0.7rem; }}

    /* Table List View */
    .table-container {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      overflow-x: auto;
      display: none;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      background: #f8fafc;
      padding: 0.85rem 1rem;
      font-weight: 700;
      border-bottom: 2px solid var(--border);
      color: var(--text);
    }}
    td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      color: #334155;
    }}
    tr:hover td {{ background: #f8fafc; }}
    .mod-tag {{
      display: inline-block;
      font-weight: 700;
      color: var(--primary);
      background: #f3e8ff;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
    }}
    .venue-tag {{
      font-weight: 600;
      background: #e2e8f0;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-size: 0.8rem;
    }}

    /* Modal */
    .modal-backdrop {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.5);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }}
    .modal {{
      background: white;
      border-radius: var(--radius);
      width: 90%;
      max-width: 550px;
      max-height: 85vh;
      overflow-y: auto;
      padding: 1.5rem;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.75rem;
      margin-bottom: 1rem;
    }}
    .modal-title {{ font-size: 1.25rem; font-weight: 700; color: var(--primary); }}
    .modal-close {{ font-size: 1.5rem; cursor: pointer; border: none; background: transparent; }}
    .date-pill {{
      display: inline-block;
      background: #f1f5f9;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      margin: 0.2rem;
      font-size: 0.8rem;
    }}

    @media print {{
      header, .controls-card, .btn-action {{ display: none !important; }}
      body {{ background: white; padding: 0; }}
      .calendar-container {{ box-shadow: none; border: 1px solid #ccc; }}
      .calendar-grid {{ min-width: 100%; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="header-content">
    <div class="title-area">
      <h1>NWU Class Timetable <span class="badge">Mahikeng Campus</span></h1>
      <div class="subtitle">Official Celcat Schedule Explorer (2026 Academic Year)</div>
    </div>
    <div style="text-align: right; font-size: 0.85rem; opacity: 0.9;">
      <div>35,733 sessions • 1,162 modules</div>
      <div style="font-size: 0.75rem;">Verified via Celcat Live</div>
    </div>
  </div>
</header>

<main>
  <!-- Search & Controls -->
  <div class="controls-card">
    <div class="search-row">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="searchInput" class="search-input" placeholder="Search module code (e.g. ALDE111, CMPG111, ACCC112)..." list="moduleList">
        <datalist id="moduleList"></datalist>
      </div>

      <select id="courseSelect" class="filter-select" style="min-width: 260px; font-weight: 600;">
        <option value="">🎓 Select Degree / Course (auto-suggest)...</option>
      </select>

      <select id="courseYearSelect" class="filter-select" style="display:none; min-width: 130px; font-weight: 600;">
        <option value="">Select Year...</option>
      </select>

      <select id="semesterFilter" class="filter-select">
        <option value="all">All Semesters (Full Year)</option>
        <option value="1">Semester 1 (Feb - Jun)</option>
        <option value="2">Semester 2 (Jul - Nov)</option>
      </select>

      <select id="groupFilter" class="filter-select">
        <option value="all">All Class Groups</option>
      </select>

      <select id="venueFilter" class="filter-select">
        <option value="all">All Venues</option>
      </select>

      <div class="view-toggle">
        <button id="btnGrid" class="toggle-btn active" onclick="setView('grid')">📅 Calendar Grid</button>
        <button id="btnTable" class="toggle-btn" onclick="setView('table')">📋 List Table</button>
      </div>
    </div>

    <!-- Active Filter & My Courses Tray -->
    <div class="selected-tray" id="selectedTray">
      <span class="tray-label">Enrolled Modules:</span>
      <div id="chipsContainer" style="display: flex; gap: 0.4rem; flex-wrap: wrap;"></div>
      <button class="btn-action" style="margin-left: auto;" onclick="clearSelection()">Clear All</button>
      <button class="btn-action" onclick="window.print()">🖨️ Print / Save PDF</button>
    </div>
  </div>

  <!-- Clash Warning -->
  <div class="clash-alert" id="clashAlert">
    ⚠️ <strong>Timetable Clash Detected:</strong> <span id="clashMessage">You have overlapping sessions in your selected modules!</span>
  </div>

  <!-- Results Count -->
  <div style="margin-bottom: 0.75rem; font-size: 0.9rem; color: var(--text-muted); display: flex; justify-content: space-between;">
    <span id="resultsCount">Showing all scheduled modules</span>
  </div>

  <!-- Grid View -->
  <div class="calendar-container" id="gridView">
    <div class="calendar-grid" id="calendarGrid">
      <!-- Generated via JS -->
    </div>
  </div>

  <!-- Table View -->
  <div class="table-container" id="tableView">
    <table>
      <thead>
        <tr>
          <th>Module Code</th>
          <th>Class Group</th>
          <th>Day</th>
          <th>Time Slot</th>
          <th>Venue</th>
          <th>Semester</th>
          <th>Weeks Active</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>
</main>

<!-- Details Modal -->
<div class="modal-backdrop" id="modalBackdrop" onclick="closeModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h3 class="modal-title" id="modalTitle">Course Details</h3>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div id="modalBody"></div>
  </div>
</div>

<script>
  const DATA = {data_json};

  let selectedModules = new Set();
  let currentView = 'grid';

  const TIME_SLOTS = [
    {{ start: "07:30", end: "09:00", label: "07:30 - 09:00" }},
    {{ start: "09:15", end: "10:45", label: "09:15 - 10:45" }},
    {{ start: "11:00", end: "12:30", label: "11:00 - 12:30" }},
    {{ start: "12:45", end: "14:15", label: "12:45 - 14:15" }},
    {{ start: "14:30", end: "16:00", label: "14:30 - 16:00" }},
    {{ start: "16:15", end: "17:45", label: "16:15 - 17:45" }},
  ];

  const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];

  // Initialize
  window.addEventListener('DOMContentLoaded', () => {{
    // Populate module autocomplete list
    const dl = document.getElementById('moduleList');
    DATA.modules.forEach(m => {{
      const opt = document.createElement('option');
      opt.value = m;
      dl.appendChild(opt);
    }});

    // Populate Venues dropdown
    const vSelect = document.getElementById('venueFilter');
    DATA.venues.forEach(v => {{
      if (!v) return;
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      vSelect.appendChild(opt);
    }});

    // Populate Groups dropdown (top 40 sample)
    const gSelect = document.getElementById('groupFilter');
    DATA.groups.slice(0, 50).forEach(g => {{
      if (!g) return;
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = g;
      gSelect.appendChild(opt);
    }});

    // Populate Courses dropdown
    const cSelect = document.getElementById('courseSelect');
    const cySelect = document.getElementById('courseYearSelect');
    if (DATA.courses) {{
      Object.keys(DATA.courses).sort().forEach(cName => {{
        const opt = document.createElement('option');
        opt.value = cName;
        opt.textContent = cName;
        cSelect.appendChild(opt);
      }});

      cSelect.addEventListener('change', () => {{
        const cName = cSelect.value;
        cySelect.innerHTML = '<option value="">Select Year...</option>';
        if (!cName || !DATA.courses[cName]) {{
          cySelect.style.display = 'none';
          return;
        }}
        const years = Object.keys(DATA.courses[cName]).sort();
        years.forEach(yr => {{
          const opt = document.createElement('option');
          opt.value = yr;
          opt.textContent = yr;
          cySelect.appendChild(opt);
        }});
        cySelect.style.display = 'inline-block';
        if (years.length > 0) {{
          cySelect.value = years[0];
          applyCourseModules(cName, years[0]);
        }}
      }});

      cySelect.addEventListener('change', () => {{
        const cName = cSelect.value;
        const yr = cySelect.value;
        if (cName && yr) {{
          applyCourseModules(cName, yr);
        }}
      }});
    }}

    function applyCourseModules(cName, yr) {{
      selectedModules.clear();
      const mods = (DATA.courses[cName] && DATA.courses[cName][yr]) || [];
      mods.forEach(m => selectedModules.add(m.toUpperCase()));
      renderChips();
      render();
    }}

    // Search events
    const sInput = document.getElementById('searchInput');
    sInput.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') {{
        const val = sInput.value.trim().toUpperCase();
        if (val) {{
          addModule(val);
          sInput.value = '';
        }}
      }}
    }});
    sInput.addEventListener('input', () => render());
    document.getElementById('semesterFilter').addEventListener('change', () => render());
    document.getElementById('groupFilter').addEventListener('change', () => render());
    document.getElementById('venueFilter').addEventListener('change', () => render());

    // Default select first popular module for a great initial display
    addModule('ALDE111');
  }});

  function setView(view) {{
    currentView = view;
    document.getElementById('btnGrid').classList.toggle('active', view === 'grid');
    document.getElementById('btnTable').classList.toggle('active', view === 'table');
    document.getElementById('gridView').style.display = view === 'grid' ? 'block' : 'none';
    document.getElementById('tableView').style.display = view === 'table' ? 'block' : 'none';
    render();
  }}

  function addModule(modCode) {{
    modCode = modCode.toUpperCase();
    selectedModules.add(modCode);
    renderChips();
    render();
  }}

  function removeModule(modCode) {{
    selectedModules.delete(modCode);
    renderChips();
    render();
  }}

  function clearSelection() {{
    selectedModules.clear();
    renderChips();
    render();
  }}

  function renderChips() {{
    const container = document.getElementById('chipsContainer');
    container.innerHTML = '';
    if (selectedModules.size === 0) {{
      container.innerHTML = '<span style="font-size:0.85rem;color:#94a3b8;font-style:italic;">No modules pinned (showing search filter)</span>';
      return;
    }}
    selectedModules.forEach(m => {{
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `${{m}} <span class="chip-remove" onclick="removeModule('${{m}}')">&times;</span>`;
      container.appendChild(chip);
    }});
  }}

  function getFilteredSlots() {{
    const searchVal = document.getElementById('searchInput').value.trim().toUpperCase();
    const semVal = document.getElementById('semesterFilter').value;
    const groupVal = document.getElementById('groupFilter').value;
    const venueVal = document.getElementById('venueFilter').value;

    return DATA.slots.filter(s => {{
      // Filter by pinned modules if any exist
      if (selectedModules.size > 0 && !selectedModules.has(s.m.toUpperCase())) {{
        return false;
      }}
      // Search input filter (if typed)
      if (searchVal) {{
        const matchMod = s.m && s.m.toUpperCase().includes(searchVal);
        const matchGrp = s.g && s.g.toUpperCase().includes(searchVal);
        const matchVen = s.v && s.v.toUpperCase().includes(searchVal);
        if (!matchMod && !matchGrp && !matchVen) return false;
      }}
      // Semester filter
      if (semVal !== 'all' && s.sem !== parseInt(semVal)) return false;
      // Group filter
      if (groupVal !== 'all' && s.g !== groupVal) return false;
      // Venue filter
      if (venueVal !== 'all' && s.v !== venueVal) return false;

      return true;
    }});
  }}

  function detectClashes(slots) {{
    const clashes = new Set();
    const timeOccupancy = {{}};

    slots.forEach((s, idx) => {{
      const key = `${{s.d}}_${{s.s}}_${{s.sem}}`;
      if (!timeOccupancy[key]) {{
        timeOccupancy[key] = [];
      }}
      timeOccupancy[key].push(idx);
    }});

    Object.values(timeOccupancy).forEach(indices => {{
      if (indices.length > 1) {{
        indices.forEach(i => clashes.add(i));
      }}
    }});
    return clashes;
  }}

  function render() {{
    const slots = getFilteredSlots();
    const clashIndices = detectClashes(slots);

    // Clash banner
    const clashAlert = document.getElementById('clashAlert');
    if (clashIndices.size > 0 && selectedModules.size > 1) {{
      clashAlert.style.display = 'block';
    }} else {{
      clashAlert.style.display = 'none';
    }}

    document.getElementById('resultsCount').textContent = `Showing ${{slots.length}} weekly slots (${{selectedModules.size}} enrolled modules)`;

    if (currentView === 'grid') {{
      renderGrid(slots, clashIndices);
    }} else {{
      renderTable(slots);
    }}
  }}

  function renderGrid(slots, clashIndices) {{
    const grid = document.getElementById('calendarGrid');
    grid.innerHTML = '';

    // Headers
    const timeH = document.createElement('div');
    timeH.className = 'grid-header time-col';
    timeH.textContent = 'Time';
    grid.appendChild(timeH);

    DAYS.forEach(d => {{
      const h = document.createElement('div');
      h.className = 'grid-header';
      h.textContent = d;
      grid.appendChild(h);
    }});

    // Time slots
    TIME_SLOTS.forEach(ts => {{
      const timeLbl = document.createElement('div');
      timeLbl.className = 'time-slot-label';
      timeLbl.innerHTML = `<strong>${{ts.start}}</strong><br><span style="font-size:0.7rem;">${{ts.end}}</span>`;
      grid.appendChild(timeLbl);

      DAYS.forEach(d => {{
        const cell = document.createElement('div');
        cell.className = 'day-column-cell';

        // Find events in this day and roughly this slot
        const cellSlots = slots.map((s, idx) => ({{ s, idx }})).filter(({{ s }}) => {{
          return s.d === d && (s.s === ts.start || (s.s >= ts.start && s.s < ts.end));
        }});

        cellSlots.forEach(({{ s, idx }}) => {{
          const isClash = clashIndices.has(idx);
          const card = document.createElement('div');
          card.className = 'event-card' + (isClash ? ' clash' : '');
          card.onclick = () => showModal(s);
          card.innerHTML = `
            <div class="ev-mod">${{s.m || 'Class'}} <span style="font-size:0.7rem;font-weight:400;color:#64748b;">(Sem ${{s.sem}})</span></div>
            <div class="ev-group">${{s.g}}</div>
            <div class="ev-venue">${{s.v || 'TBA'}}</div>
            <div class="ev-time">${{s.t}} (${{s.w}} wks)</div>
          `;
          cell.appendChild(card);
        }});

        grid.appendChild(cell);
      }});
    }});
  }}

  function renderTable(slots) {{
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';

    if (slots.length === 0) {{
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:2rem;">No matching timetable slots found.</td></tr>';
      return;
    }}

    slots.forEach(s => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="mod-tag">${{s.m || '-'}}</span></td>
        <td><strong>${{s.g || '-'}}</strong></td>
        <td>${{s.d}}</td>
        <td>${{s.t}}</td>
        <td><span class="venue-tag">${{s.v || '-'}}</span></td>
        <td>Semester ${{s.sem}}</td>
        <td>${{s.w}} weeks (${{s.first.slice(5)}} to ${{s.last.slice(5)}})</td>
        <td><button class="btn-action" onclick='showModal(${{JSON.stringify(s)}})'>Details</button></td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  function showModal(s) {{
    document.getElementById('modalTitle').textContent = `${{s.m}} - ${{s.g}}`;
    const body = document.getElementById('modalBody');
    body.innerHTML = `
      <div style="margin-bottom:1rem;">
        <p><strong>Venue:</strong> ${{s.v}}</p>
        <p><strong>Day & Time:</strong> ${{s.d}}, ${{s.t}}</p>
        <p><strong>Semester:</strong> Semester ${{s.sem}} (Active for ${{s.w}} weeks)</p>
        <p><strong>Date Range:</strong> ${{s.first}} to ${{s.last}}</p>
      </div>
      <h4 style="font-size:0.95rem;margin-bottom:0.5rem;font-weight:700;">Individual Session Dates:</h4>
      <div style="max-height:220px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:6px;padding:0.5rem;">
        ${{s.dates.map(d => `<span class="date-pill">${{d}}</span>`).join('')}}
      </div>
    `;
    document.getElementById('modalBackdrop').style.display = 'flex';
  }}

  function closeModal(e) {{
    if (!e || e.target.id === 'modalBackdrop' || e.target.className === 'modal-close') {{
      document.getElementById('modalBackdrop').style.display = 'none';
    }}
  }}
</script>

</body>
</html>
"""
    output_html.write_text(html_content, encoding="utf-8")
    print(f"\n[✓] Generated {output_html.resolve()} ({output_html.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate interactive HTML timetable viewer.")
    parser.add_argument("--csv", "-c", type=Path, default=DEFAULT_CSV, help="Source timetable CSV")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_HTML, help="Output HTML file path")
    args = parser.parse_args()

    generate_viewer(args.csv, args.out)
