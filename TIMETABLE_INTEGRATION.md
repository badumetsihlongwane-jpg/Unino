# NWU Celcat Timetable & Course Data Integration Guide

This guide is written for developers and AI coding assistants (GitHub Copilot, Cursor, etc.) integrating the captured NWU Celcat timetable and course catalog into the **Unino** web & mobile app.

---

## 1. Directory & File Overview

All generated datasets, harvesters, and tools are located in the repository root:

```text
├── celcat_capture/
│   ├── nwu_mahikeng_timetable.csv          # Complete 2026 timetable (35,733 records)
│   ├── nwu_courses_quick_suggest.json      # Course -> Year -> Modules map for onboarding
│   ├── nwu_courses_catalogue.csv           # Relational degree-to-module table (3,456 rows)
│   ├── nwu_courses_catalogue.json          # Full structured curriculum tree (200 degrees)
│   └── *.json                              # Raw intercepted Celcat response dumps
├── scrape_all_timetable.py                 # Automated fast harvester (queries Celcat API directly)
├── build_courses_dataset.py                # Harvester that maps degrees to active modules
├── build_timetable.py                      # Standalone compiler: converts raw JSON into clean CSV
├── search_timetable.py                     # Fast CLI search tool for testing course lookups
├── export_html_viewer.py                   # Generator for the standalone web viewer
├── timetable_viewer.html                   # Standalone interactive dashboard (Grid & List)
└── capture_celcat.py                       # Playwright browser recorder (manual fallback)
```

---

## 2. Data Schemas & Formats

### A. Course Quick-Suggest: `celcat_capture/nwu_courses_quick_suggest.json`
Designed for instant client-side lookup during user sign-up in `index.html` and `js/app.js`:

```json
{
  "BAdmin in Development and Mng with Local Government": {
    "Year 1": ["ACCS111", "ALDE111", "ECON112", "INFS111", "LGAM115", "PMAN111"],
    "Year 2": ["LGAM211", "LGAM212", "LGAM216", "PMAN214"],
    "Year 3": ["LGAM312", "LGAM318", "PMAN313", "WVES312"]
  },
  "BSocSc with Geography": {
    "Year 1": ["ALDE111", "DEVS111", "ECON112", "ENLL111", "GEOG111", "HIST112", "HPOP111", "IKSM111", "PHIL112", "POLI112", "PSYC111", "SETM111", "SOCY111"],
    "Year 2": ["DEVS211", "ECON211", "ENLL211", "GEOG211", "HIST213", "HPOP212", "IKSM211", "PHIL213", "POLI213", "PSYC211", "PSYC214", "SETM211", "SOCY211"],
    "Year 3": ["DEVS311", "GEOG311", "HIST312", "HPOP311", "HPOP312", "HPOP322", "POLI313", "SOCY314", "SOCY317", "WVLS317"]
  }
}
```

### B. Relational Course Table: `celcat_capture/nwu_courses_catalogue.csv`
Ready to import into Appwrite Database, Firestore, PostgreSQL, or Supabase:

| Column | Type | Example | Description |
|---|---|---|---|
| `course_id` | string | `1DCH01-L301M` | NWU curriculum / programme version code |
| `course_name` | string | `BAdmin in Development and Mng with Local Government` | Official qualification title |
| `qualification` | string | `BCom / BAdmin` | Broad degree category |
| `faculty` | string | `Faculty of Economic and Management Sciences` | Inferred NWU faculty |
| `year_of_study` | string | `Year 1` | `Year 1`, `Year 2`, `Year 3`, `Year 4` |
| `semester` | string | `Semester 1` | `Semester 1` or `Semester 2` |
| `module_code` | string | `ALDE111` | 4-letter + 3/4-digit module code |

### C. Master Timetable: `celcat_capture/nwu_mahikeng_timetable.csv`
Complete schedule of every lecture, tutorial, and lab session:

| Column | Type | Example | Description |
|---|---|---|---|
| `celcat_event_id` | string | `187035095:49806811:14:3852:6` | Unique Celcat event ID |
| `module_code` | string | `ALDE111` | Clean module code |
| `class_group` | string | `NAS_002` | Class or cohort group |
| `start_datetime` | string | `2026-02-11T07:30:00` | ISO 8601 start datetime |
| `end_datetime` | string | `2026-02-11T09:00:00` | ISO 8601 end datetime |
| `timezone` | string | `Africa/Johannesburg` | South African Standard Time (SAST) |
| `venue_code` | string | `MC-A5_105` | Venue code without capacity brackets |
| `venue_name` | string | `MC-A5_105` | Venue name / description |
| `description` | string | `ALDE111_NAS_002\nMC-A5_105 [180]` | Cleaned raw description |
| `last_verified` | string | `2026-10-14T08:05:35Z` | Capture timestamp |

---

## 3. Integration Recipes for Copilot

### Recipe 1: Auto-Suggest Modules During Unino Sign-Up (`index.html` & `js/app.js`)

In `index.html`:
```html
<!-- Replace or augment the plain text #s-major input -->
<input type="text" id="s-major" placeholder="Your course/major (e.g. BAdmin in Development...)" list="courses-datalist" required>
<datalist id="courses-datalist"></datalist>

<!-- Container for suggested module chips -->
<div id="signup-module-suggestions" style="display:flex; flex-wrap:wrap; gap:6px; margin: 8px 0;"></div>
```

In `js/app.js`:
```javascript
// 1. Load quick suggest catalogue
let COURSES_CATALOGUE = {};
async function initCourseSuggestions() {
  try {
    const res = await fetch('celcat_capture/nwu_courses_quick_suggest.json');
    COURSES_CATALOGUE = await res.json();
    
    const datalist = document.getElementById('courses-datalist');
    if (datalist) {
      Object.keys(COURSES_CATALOGUE).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        datalist.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn('Could not load course suggestions', err);
  }
}

// 2. When major or year changes, suggest modules
function updateModuleSuggestions() {
  const major = document.getElementById('s-major')?.value.trim();
  const yearRaw = document.getElementById('s-year')?.value.trim(); // e.g. "1st Year"
  const suggestionsBox = document.getElementById('signup-module-suggestions');
  const modulesInput = document.getElementById('s-modules');
  
  if (!major || !yearRaw || !COURSES_CATALOGUE[major] || !suggestionsBox) return;

  // Convert "1st Year" -> "Year 1"
  const yrNum = yearRaw.match(/\d/)?.[0] || '1';
  const yearKey = `Year ${yrNum}`;
  const suggestedMods = COURSES_CATALOGUE[major][yearKey] || [];

  suggestionsBox.innerHTML = '';
  suggestedMods.forEach(mod => {
    const chip = document.createElement('span');
    chip.className = 'module-chip clickable';
    chip.textContent = `+ ${mod}`;
    chip.onclick = () => {
      const current = (modulesInput.value || '').split(',').map(s => s.trim()).filter(Boolean);
      if (!current.includes(mod)) {
        current.push(mod);
        modulesInput.value = current.join(', ');
        chip.style.opacity = '0.5';
      }
    };
    suggestionsBox.appendChild(chip);
  });
}

document.getElementById('s-major')?.addEventListener('change', updateModuleSuggestions);
document.getElementById('s-year')?.addEventListener('change', updateModuleSuggestions);
```

---

### Recipe 2: Show Student's Weekly Schedule from Profile Modules

Given a student user with `user.modules = ['ALDE111', 'CMPG111']`:

```javascript
// Filter timetable rows for student's enrolled modules
function getStudentTimetable(allRows, userModules) {
  const mods = new Set(userModules.map(m => m.toUpperCase().trim()));
  return allRows.filter(r => mods.has((r.module_code || '').toUpperCase()));
}

// Check for clashes
function findClashes(studentRows) {
  const clashMap = {};
  studentRows.forEach(row => {
    const key = `${row.start_datetime}`;
    clashMap[key] = clashMap[key] || [];
    clashMap[key].push(row);
  });
  return Object.values(clashMap).filter(list => list.length > 1);
}
```

---

### Recipe 3: Appwrite Collection Schema

If importing to Appwrite:
1. **Database ID**: `unino_academic`
2. **Collection 1: `courses`**
   - `course_id` (string, 32, required)
   - `course_name` (string, 128, required)
   - `qualification` (string, 64)
   - `faculty` (string, 64)
   - `year_of_study` (string, 16)
   - `semester` (string, 16)
   - `module_code` (string, 16, index)
3. **Collection 2: `timetables`**
   - `celcat_event_id` (string, 64, unique)
   - `module_code` (string, 16, index)
   - `class_group` (string, 32)
   - `start_datetime` (datetime, required)
   - `end_datetime` (datetime, required)
   - `timezone` (string, 32)
   - `venue_code` (string, 32, index)
   - `venue_name` (string, 64)

---

## 4. CLI Scripts Reference

| Command | Action |
|---|---|
| `python3 scrape_all_timetable.py --campus mahikeng` | Harvests all 91 rooms and regenerates timetable CSV in ~18s |
| `python3 build_courses_dataset.py --campus mahikeng` | Maps all 200 degrees to modules and regenerates JSON & CSV |
| `python3 search_timetable.py ALDE111` | Queries and displays weekly recurring slots in terminal |
| `python3 export_html_viewer.py` | Compiles the standalone HTML calendar viewer (`timetable_viewer.html`) |
| `open timetable_viewer.html` | Opens the visual schedule dashboard in Google Chrome |
