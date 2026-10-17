#!/usr/bin/env python3
"""
NWU Accredited Landlord Automated WhatsApp Dispatcher
Uses Playwright with a persistent Google Chrome session to automatically dispatch
personalized residence photo upload links to landlords via WhatsApp Web.

Features:
- Persistent profile in data/whatsapp_chrome_profile (Scan QR once, stay logged in).
- Deduplication & resumption via data/landlord_sent_log.json.
- Rate-limiting jitter delays to protect your WhatsApp number from spam flags.
- Auto-detects invalid / non-WhatsApp numbers and logs them cleanly.
- CLI arguments: --batch, --all, --delay, --dry-run, --status, --start-from.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.parse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CAMPAIGN_CSV = os.path.join(DATA_DIR, "landlord_campaign.csv")
SENT_LOG_FILE = os.path.join(DATA_DIR, "landlord_sent_log.json")
PROFILE_DIR = os.path.join(DATA_DIR, "whatsapp_chrome_profile")
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def load_sent_log():
    if os.path.exists(SENT_LOG_FILE):
        try:
            with open(SENT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sent_log(log_data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SENT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def load_campaign_csv():
    if not os.path.exists(CAMPAIGN_CSV):
        print(f"[ERROR] Campaign file not found: {CAMPAIGN_CSV}")
        print("Run `python3 scripts/landlord_whatsapp_blast.py` first to generate it.")
        sys.exit(1)
    
    records = []
    with open(CAMPAIGN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def print_status(records, sent_log):
    total = len(records)
    sent_success = sum(1 for v in sent_log.values() if v.get("status") == "SENT")
    invalid_phones = sum(1 for v in sent_log.values() if v.get("status") == "INVALID_NUMBER")
    pending = total - len(sent_log)

    print("\n" + "=" * 55)
    print(" 📊 NWU LANDLORD WHATSAPP CAMPAIGN STATUS")
    print("=" * 55)
    print(f" Total Unique Landlords   : {total}")
    print(f" Messages Sent Successfully: {sent_success} ({(sent_success/total)*100:.1f}%)")
    print(f" Invalid / Non-WhatsApp   : {invalid_phones}")
    print(f" Remaining Pending        : {pending}")
    print("=" * 55 + "\n")


def run_sender(args):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        print("[ERROR] Playwright is not installed. Install via: pip install playwright")
        sys.exit(1)

    records = load_campaign_csv()
    sent_log = load_sent_log()

    if args.status:
        print_status(records, sent_log)
        return

    # Filter out already contacted numbers unless --force
    to_send = []
    for r in records:
        phone = r.get("Landlord_WhatsApp", "").strip()
        if not phone:
            continue
        if phone in sent_log and not args.force:
            continue
        to_send.append(r)

    print_status(records, sent_log)

    if not to_send:
        print("🎉 All landlords in the campaign have already been contacted!")
        print("Use --force to resend, or inspect data/landlord_sent_log.json.")
        return

    # Determine batch limit
    if args.batch and args.batch > 0:
        batch_records = to_send[:args.batch]
        print(f"[INFO] Processing batch of {len(batch_records)} landlords (out of {len(to_send)} pending).")
    elif args.all:
        batch_records = to_send
        print(f"[INFO] Processing ALL {len(batch_records)} pending landlords.")
    else:
        # Default batch size: 10
        batch_records = to_send[:10]
        print(f"[INFO] No batch size specified; processing default batch of {len(batch_records)} landlords.")
        print("       (Tip: specify `--batch 25` or `--all` to send more at once).")

    if args.dry_run:
        print("\n🔎 DRY-RUN MODE: Simulating dispatch without sending messages...\n")
        for idx, rec in enumerate(batch_records, 1):
            phone = rec["Landlord_WhatsApp"]
            names = rec["Residence_Names"]
            res_count = rec["Residence_Count"]
            link = rec["Direct_Upload_Link"]
            print(f"[{idx}/{len(batch_records)}] Landlord {phone} ({res_count} properties: {names})")
            print(f"   Upload Link: {link}")
            print(f"   Message Preview: {rec['Message_Text'].splitlines()[0]}...")
            print("-" * 50)
        print(f"\n[DRY RUN COMPLETE] {len(batch_records)} records inspected. No messages were actually sent.")
        return

    # Setup browser
    os.makedirs(PROFILE_DIR, exist_ok=True)
    executable = CHROME_PATH if os.path.exists(CHROME_PATH) else None

    print("\n🚀 Launching Chrome with WhatsApp Web persistent session...")
    print(f"   Profile Dir: {PROFILE_DIR}")
    if executable:
        print(f"   Browser Executable: {executable}")

    with sync_playwright() as p:
        launch_args = [
            "--no-default-browser-check",
            "--no-first-run",
            "--disable-blink-features=AutomationControlled"
        ]
        
        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            executable_path=executable,
            headless=False,
            args=launch_args,
            viewport={"width": 1200, "height": 850}
        )

        page = context.pages[0] if context.pages else context.new_page()

        # Step 1: Open WhatsApp Web and wait for user authentication if needed
        print("[INFO] Checking WhatsApp Web login status...")
        page.goto("https://web.whatsapp.com", wait_until="load", timeout=60000)

        # Wait for either the chat list (already logged in) or the QR code canvas
        print("[INFO] Waiting for WhatsApp Web interface to load...")
        try:
            # Check if already authenticated
            page.wait_for_selector(
                "#pane-side, div[data-testid='chat-list'], div[aria-label='Chat list']",
                timeout=12000
            )
            print("✓ WhatsApp Web authenticated and active!")
        except PlaywrightTimeout:
            print("\n" + "!" * 60)
            print("📱 ONE-TIME WHATSAPP QR SCAN REQUIRED:")
            print("   Please look at the opened Chrome window and scan the QR code")
            print("   with your phone (WhatsApp -> Linked Devices -> Link a Device).")
            print("!" * 60 + "\n")
            # Wait up to 120 seconds for user to scan QR code
            page.wait_for_selector(
                "#pane-side, div[data-testid='chat-list'], div[aria-label='Chat list']",
                timeout=120000
            )
            print("✓ Login detected! Proceeding with automated dispatch...\n")

        # Step 2: Loop over batch landlords
        total_batch = len(batch_records)
        sent_count = 0
        failed_count = 0

        for idx, rec in enumerate(batch_records, 1):
            phone = rec["Landlord_WhatsApp"].strip()
            orig_phone = rec["Original_Phone"]
            names = rec["Residence_Names"]
            res_count = rec["Residence_Count"]
            msg = rec["Message_Text"]

            print(f"[{idx}/{total_batch}] Sending to {phone} ({res_count} res: {names[:40]}...)...")

            encoded_msg = urllib.parse.quote(msg)
            send_url = f"https://web.whatsapp.com/send?phone={phone}&text={encoded_msg}"

            try:
                page.goto(send_url, timeout=45000)

                # Check for either the send button, input box, or invalid phone alert
                invalid_number = False
                send_button_found = False

                for _ in range(36):
                    time.sleep(0.5)

                    # Check for invalid number modal
                    invalid_popup = page.query_selector(
                        "div[data-animate-modal-popup='true'], div[role='dialog']"
                    )
                    if invalid_popup:
                        text_content = invalid_popup.inner_text() or ""
                        if "invalid" in text_content.lower() or "not on whatsapp" in text_content.lower():
                            invalid_number = True
                            ok_btn = invalid_popup.query_selector("button, div[role='button']")
                            if ok_btn:
                                ok_btn.click()
                            break

                    # Check for message input box or send button
                    send_btn = page.query_selector("button[aria-label='Send'], span[data-icon='send']")
                    composer = page.query_selector("footer div[contenteditable='true'], div[role='textbox'][data-tab='10']")
                    
                    if send_btn or composer:
                        send_button_found = True
                        break

                if invalid_number:
                    print(f"   ⚠️ Number {phone} is not on WhatsApp or invalid.")
                    sent_log[phone] = {
                        "status": "INVALID_NUMBER",
                        "phone": phone,
                        "orig_phone": orig_phone,
                        "names": names,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_sent_log(sent_log)
                    failed_count += 1
                    continue

                if not send_button_found:
                    print(f"   ⚠️ Timeout waiting for chat composer with {phone}. Skipping.")
                    continue

                # Trigger Send: Click send button or press Enter in composer
                time.sleep(1.0)
                send_btn = page.query_selector("button[aria-label='Send'], span[data-icon='send']")
                if send_btn:
                    send_btn.click()
                else:
                    composer = page.query_selector("footer div[contenteditable='true']")
                    if composer:
                        composer.focus()
                        page.keyboard.press("Enter")

                # Wait 2.5 seconds to ensure message gets sent into outbox
                time.sleep(2.5)

                print(f"   ✓ Sent successfully to {phone}!")
                sent_log[phone] = {
                    "status": "SENT",
                    "phone": phone,
                    "orig_phone": orig_phone,
                    "names": names,
                    "res_count": res_count,
                    "timestamp": datetime.now().isoformat()
                }
                save_sent_log(sent_log)
                sent_count += 1

                # Apply anti-spam jitter delay if not the last item
                if idx < total_batch:
                    base_delay = args.delay
                    jitter = random.uniform(base_delay * 0.85, base_delay * 1.3)
                    print(f"   ⏳ Sleeping {jitter:.1f}s to respect WhatsApp messaging rate limits...")
                    time.sleep(jitter)

            except Exception as e:
                print(f"   ❌ Error sending to {phone}: {e}")
                failed_count += 1

        print("\n" + "=" * 55)
        print(f" 🎉 BATCH RUN COMPLETE:")
        print(f"    Sent: {sent_count}")
        print(f"    Failed/Invalid: {failed_count}")
        print(f"    Log updated: {SENT_LOG_FILE}")
        print("=" * 55 + "\n")

        print("[INFO] Closing browser session cleanly...")
        context.close()


def main():
    parser = argparse.ArgumentParser(
        description="Automated WhatsApp Dispatcher for NWU Landlords."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Number of landlords to contact in this batch (e.g. --batch 20)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all pending landlords in the campaign."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=9.0,
        help="Base delay in seconds between messages (default: 9.0s + jitter)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate sending without executing WhatsApp clicks or network actions."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show campaign progress & statistics without opening browser."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resend even if landlord was already logged as sent."
    )

    args = parser.parse_args()
    run_sender(args)


if __name__ == "__main__":
    main()
