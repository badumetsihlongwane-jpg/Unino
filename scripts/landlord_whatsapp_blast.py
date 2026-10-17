#!/usr/bin/env python3
"""
NWU Accredited Landlord WhatsApp & SMS Campaign Generator
Generates personalized zero-auth upload links, CSV campaign sheet,
and an interactive WhatsApp Click-to-Chat Dispatcher Dashboard.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_JSON = os.path.join(ROOT_DIR, "data", "approved_residences.json")
OUTPUT_CSV = os.path.join(ROOT_DIR, "data", "landlord_campaign.csv")
OUTPUT_DASHBOARD = os.path.join(ROOT_DIR, "data", "landlord_blast_dashboard.html")
BASE_URL = "https://unibo.social/landlord.html"


def clean_phone_number(raw_phone):
    """Normalize South African phone numbers to international format (27XXXXXXXXX)."""
    if not raw_phone:
        return ""
    digits = re.sub(r"[^0-9]", "", str(raw_phone))
    if digits.startswith("0") and len(digits) == 10:
        return "27" + digits[1:]
    elif digits.startswith("27") and len(digits) == 11:
        return digits
    elif len(digits) == 9:
        return "27" + digits
    return digits


def load_residences():
    if not os.path.exists(DATA_JSON):
        print(f"Error: {DATA_JSON} not found.")
        sys.exit(1)
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("residences", [])


def group_by_landlord(residences):
    """Group residences by clean WhatsApp / Phone number."""
    landlords = {}
    no_phone_res = []

    for res in residences:
        phone_raw = res.get("whatsapp") or res.get("phone") or ""
        clean_phone = clean_phone_number(phone_raw)
        
        if not clean_phone:
            no_phone_res.append(res)
            continue

        if clean_phone not in landlords:
            landlords[clean_phone] = {
                "phone": clean_phone,
                "raw_phone": res.get("phone") or res.get("whatsapp") or clean_phone,
                "email": res.get("email") or "",
                "residences": []
            }
        
        landlords[clean_phone]["residences"].append(res)
        if not landlords[clean_phone]["email"] and res.get("email"):
            landlords[clean_phone]["email"] = res.get("email")

    return landlords, no_phone_res


def build_message(landlord_data):
    res_list = landlord_data["residences"]
    if len(res_list) == 1:
        res = res_list[0]
        res_name = res.get("name", "your residence")
        upload_link = f"{BASE_URL}?id={urllib.parse.quote(res.get('id', ''))}"
        msg = (
            f"Hello! This is NWU Student Life (Unibo).\n\n"
            f"Please tap the link below to upload 2-5 photos of your accredited residence *{res_name}* "
            f"so NWU students can view your rooms on the campus platform:\n\n"
            f"👉 {upload_link}\n\n"
            f"✓ No login or password required — simply select photos from your phone gallery and tap Save. Thank you!"
        )
    else:
        names = ", ".join([f"*{r.get('name')}*" for r in res_list[:3]])
        if len(res_list) > 3:
            names += f" and {len(res_list) - 3} others"
        first_id = res_list[0].get("id", "")
        upload_link = f"{BASE_URL}?id={urllib.parse.quote(first_id)}"
        msg = (
            f"Hello! This is NWU Student Life (Unibo).\n\n"
            f"Please tap the link below to upload photos for your accredited residences ({names}) "
            f"so students can view your rooms on the campus platform:\n\n"
            f"👉 {upload_link}\n\n"
            f"✓ No login or password required — simply select photos from your phone gallery and tap Save. Thank you!"
        )
    return msg


def generate_csv_and_dashboard():
    residences = load_residences()
    landlords, no_phone = group_by_landlord(residences)

    print(f"Loaded {len(residences)} total residences.")
    print(f"Found {len(landlords)} unique landlord phone contacts.")
    if no_phone:
        print(f"Notice: {len(no_phone)} residences have no cell/WhatsApp listed.")

    # 1. Export CSV
    records = []
    for phone, ldata in landlords.items():
        res_names = "; ".join([r.get("name", "") for r in ldata["residences"]])
        res_ids = "; ".join([r.get("id", "") for r in ldata["residences"]])
        first_res = ldata["residences"][0]
        upload_link = f"{BASE_URL}?id={urllib.parse.quote(first_res.get('id', ''))}"
        msg = build_message(ldata)
        wa_url = f"https://wa.me/{phone}?text={urllib.parse.quote(msg)}"

        records.append({
            "Landlord_WhatsApp": phone,
            "Original_Phone": ldata["raw_phone"],
            "Email": ldata["email"],
            "Residence_Count": len(ldata["residences"]),
            "Residence_Names": res_names,
            "Residence_IDs": res_ids,
            "Primary_Suburb": first_res.get("suburb", "Mahikeng"),
            "Direct_Upload_Link": upload_link,
            "WhatsApp_Click_Link": wa_url,
            "Message_Text": msg
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Landlord_WhatsApp", "Original_Phone", "Email", "Residence_Count",
            "Residence_Names", "Residence_IDs", "Primary_Suburb",
            "Direct_Upload_Link", "WhatsApp_Click_Link", "Message_Text"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✓ Saved campaign CSV to: {OUTPUT_CSV}")

    # 2. Export Interactive HTML Dashboard
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NWU Landlord WhatsApp Outreach Dispatcher</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0F172A;
      --card-bg: #1E293B;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --accent: #6C5CE7;
      --wa: #25D366;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px 20px 80px; }}
    .container {{ max-width: 1040px; margin: 0 auto; }}
    
    .header {{
      background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
      border: 1px solid #4338CA;
      border-radius: 18px;
      padding: 24px 28px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .header h1 {{ font-size: 24px; font-weight: 800; }}
    .header p {{ font-size: 14px; color: #C7D2FE; line-height: 1.5; }}
    
    .stats-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
    }}
    .stat-val {{ font-size: 22px; font-weight: 800; color: #FFFFFF; }}
    .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 700; margin-top: 2px; }}

    .search-row {{
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .search-input {{
      flex: 1;
      min-width: 260px;
      padding: 12px 18px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: #FFFFFF;
      font-size: 14px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--accent); }}

    .filter-btn {{
      padding: 10px 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }}
    .filter-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

    .landlord-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: all 0.15s;
    }}
    .landlord-card.sent {{
      opacity: 0.55;
      border-color: #059669;
      background: #064E3B;
    }}
    .landlord-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }}
    .landlord-name {{ font-size: 16px; font-weight: 800; color: #FFFFFF; }}
    .landlord-sub {{ font-size: 13px; color: var(--text-muted); margin-top: 2px; }}

    .btn-wa {{
      background: var(--wa);
      color: #FFFFFF;
      border: none;
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }}
    .btn-wa:hover {{ background: #1EBE5D; }}

    .btn-outline {{
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .btn-outline:hover {{ background: rgba(255,255,255,0.06); }}

    .toggle-sent {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }}
    .toggle-sent input {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>💬 NWU Landlord WhatsApp Outreach Dispatcher</h1>
      <p>
        One-click direct WhatsApp links for all 408 accredited landlord contacts in Mahikeng.
        Every click opens WhatsApp Web or your phone's WhatsApp with their customized residence upload link pre-filled.
      </p>
    </div>

    <div class="stats-bar">
      <div class="stat-card">
        <div class="stat-val">{len(landlords)}</div>
        <div class="stat-label">Unique Landlords</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{len(residences)}</div>
        <div class="stat-label">Total Residences</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="sent-count">0</div>
        <div class="stat-label">Messages Sent</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="remaining-count">{len(landlords)}</div>
        <div class="stat-label">Remaining</div>
      </div>
    </div>

    <div class="search-row">
      <input type="text" id="dash-search" class="search-input" placeholder="Search landlord phone, residence name, or suburb…" oninput="filterList()">
      <button type="button" class="filter-btn active" data-f="all" onclick="setDashFilter('all')">All ({len(landlords)})</button>
      <button type="button" class="filter-btn" data-f="unsent" onclick="setDashFilter('unsent')">⏳ Unsent Only</button>
      <button type="button" class="filter-btn" data-f="sent" onclick="setDashFilter('sent')">✓ Sent</button>
      <button type="button" class="btn-outline" onclick="openBatchUnsent(5)">🚀 Open Next 5 in WhatsApp Tabs</button>
    </div>

    <div id="landlord-list"></div>
  </div>

  <script>
    const DATA = {json.dumps(records)};
    let _filter = 'all';

    function getSentSet() {{
      try {{
        return new Set(JSON.parse(localStorage.getItem('unino_wa_sent_set') || '[]'));
      }} catch(e) {{ return new Set(); }}
    }}

    function saveSentSet(set) {{
      try {{
        localStorage.setItem('unino_wa_sent_set', JSON.stringify(Array.from(set)));
      }} catch(e) {{}}
      updateStats();
    }}

    function toggleSent(phone) {{
      const set = getSentSet();
      if (set.has(phone)) set.delete(phone);
      else set.add(phone);
      saveSentSet(set);
      renderList();
    }}

    function updateStats() {{
      const set = getSentSet();
      document.getElementById('sent-count').innerText = set.size;
      document.getElementById('remaining-count').innerText = Math.max(0, DATA.length - set.size);
    }}

    function setDashFilter(f) {{
      _filter = f;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.dataset.f === f));
      renderList();
    }}

    function filterList() {{
      renderList();
    }}

    function renderList() {{
      const q = (document.getElementById('dash-search')?.value || '').toLowerCase().trim();
      const sentSet = getSentSet();
      const container = document.getElementById('landlord-list');

      const filtered = DATA.filter(item => {{
        const isSent = sentSet.has(item.Landlord_WhatsApp);
        if (_filter === 'unsent' && isSent) return false;
        if (_filter === 'sent' && !isSent) return false;

        if (q) {{
          const matchPhone = item.Landlord_WhatsApp.includes(q) || item.Original_Phone.includes(q);
          const matchRes = item.Residence_Names.toLowerCase().includes(q);
          const matchSub = item.Primary_Suburb.toLowerCase().includes(q);
          if (!matchPhone && !matchRes && !matchSub) return false;
        }}
        return true;
      }});

      container.innerHTML = filtered.map(item => {{
        const isSent = sentSet.has(item.Landlord_WhatsApp);
        return `
          <div class="landlord-card ${{isSent ? 'sent' : ''}}">
            <div class="landlord-top">
              <div>
                <div class="landlord-name">${{item.Residence_Names}}</div>
                <div class="landlord-sub">
                  📞 +${{item.Landlord_WhatsApp}} • ${{item.Primary_Suburb}} • ${{item.Residence_Count}} Res
                </div>
              </div>
              <label class="toggle-sent">
                <input type="checkbox" ${{isSent ? 'checked' : ''}} onchange="toggleSent('${{item.Landlord_WhatsApp}}')">
                <span>${{isSent ? 'Sent ✓' : 'Mark Sent'}}</span>
              </label>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
              <a href="${{item.WhatsApp_Click_Link}}" target="_blank" class="btn-wa" onclick="markSentOnOpen('${{item.Landlord_WhatsApp}}')">
                💬 Open WhatsApp Chat
              </a>
              <button type="button" class="btn-outline" onclick="copyUploadLink('${{item.Direct_Upload_Link}}')">
                📋 Copy Upload Link
              </button>
              <button type="button" class="btn-outline" onclick="copyMessageText(${{DATA.indexOf(item)}})">
                📱 Copy Text Message
              </button>
            </div>
          </div>
        `;
      }}).join('');

      updateStats();
    }}

    function copyUploadLink(link) {{
      navigator.clipboard.writeText(link);
      alert('Upload link copied!');
    }}

    function copyMessageText(idx) {{
      if (DATA[idx]) {{
        navigator.clipboard.writeText(DATA[idx].Message_Text);
        alert('Message copied!');
      }}
    }}

    function markSentOnOpen(phone) {{
      const set = getSentSet();
      set.add(phone);
      saveSentSet(set);
      setTimeout(renderList, 400);
    }}

    function openBatchUnsent(count = 5) {{
      const sentSet = getSentSet();
      const unsent = DATA.filter(item => !sentSet.has(item.Landlord_WhatsApp));
      if (!unsent.length) {{
        alert('All landlords have been marked as sent! 🎉');
        return;
      }}
      const batch = unsent.slice(0, count);
      batch.forEach(item => {{
        window.open(item.WhatsApp_Click_Link, '_blank');
        sentSet.add(item.Landlord_WhatsApp);
      }});
      saveSentSet(sentSet);
      renderList();
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      renderList();
      updateStats();
    }});
  </script>
</body>
</html>
"""

    with open(OUTPUT_DASHBOARD, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print(f"✓ Saved interactive WhatsApp dispatcher dashboard to: {OUTPUT_DASHBOARD}")


def search_cli(query):
    residences = load_residences()
    landlords, _ = group_by_landlord(residences)
    q = query.lower().strip()
    q_digits = re.sub(r"[^0-9]", "", q)
    matches = []

    for phone, ldata in landlords.items():
        res_names = "; ".join([r.get("name", "") for r in ldata["residences"]])
        match_phone = False
        if q_digits and (q_digits in phone or q_digits in ldata.get("raw_phone", "")):
            match_phone = True
        if match_phone or q in phone.lower() or q in res_names.lower() or any(q in r.get("address", "").lower() for r in ldata["residences"]):
            matches.append(ldata)

    print(f"\nFound {len(matches)} matches for '{query}':\n" + "=" * 60)
    for m in matches[:10]:
        first_res = m["residences"][0]
        link = f"{BASE_URL}?id={urllib.parse.quote(first_res.get('id', ''))}"
        print(f"📞 WhatsApp: +{m['phone']} ({m['raw_phone']})")
        print(f"🏠 Residences ({len(m['residences'])}): {', '.join(r.get('name') for r in m['residences'])}")
        print(f"🔗 Direct Upload Link: {link}")
        print(f"💬 Direct WhatsApp URL: https://wa.me/{m['phone']}?text={urllib.parse.quote(build_message(m))}")
        print("-" * 60)


def print_stats():
    residences = load_residences()
    landlords, no_phone = group_by_landlord(residences)
    print("\n" + "=" * 50)
    print("NWU ACCREDITED RESIDENCES & LANDLORD STATS")
    print("=" * 50)
    print(f"Total Accredited Residences:    {len(residences)}")
    print(f"Unique WhatsApp/Cell Contacts:  {len(landlords)}")
    print(f"Residences Without Phone:       {len(no_phone)}")
    suburbs = {}
    for r in residences:
        s = r.get("suburb", "Other")
        suburbs[s] = suburbs.get(s, 0) + 1
    print("\nTop Suburbs / Corridors:")
    for s, c in sorted(suburbs.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  • {s:<24}: {c} residences")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NWU Landlord WhatsApp & SMS Campaign Tool")
    parser.add_argument("--dashboard", action="store_true", help="Generate CSV & HTML Click-to-Chat Dashboard")
    parser.add_argument("--search", type=str, help="Search for a residence or phone number")
    parser.add_argument("--stats", action="store_true", help="Display residence & landlord contact statistics")

    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.search:
        search_cli(args.search)
    else:
        # Default action: generate CSV and interactive dashboard
        generate_csv_and_dashboard()


if __name__ == "__main__":
    main()
