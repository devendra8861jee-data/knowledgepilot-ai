"""Generate the small, fictional sample PDFs used by the example evaluation dataset.

The content is invented for this project (no real company or product), so the
labels in ``sample_dataset.jsonl`` can be verified by reading the pages below.

Usage (from the project root)::

    python evaluation/make_sample_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

HANDBOOK_PAGES = [
    # Page 1: cover page (so printed page numbers and physical pages differ in real PDFs)
    "Acme Technologies\nEmployee Handbook - Edition 2026\n\nInternal use only.\n\n"
    "Contents: Leave Policy, Remote Work Policy, Expense Policy.",
    # Page 2
    "Leave Policy\n\n"
    "Full-time employees receive 24 days of paid annual leave per calendar year. "
    "Up to 8 days of unused leave can be carried forward to the next year.\n\n"
    "Sick leave is 10 days per year. A medical certificate is required for absences longer "
    "than 2 consecutive days.\n\n"
    "Maternity leave is 26 weeks paid. Paternity leave is 10 working days.\n\n"
    "Planned leave of more than 3 days must be requested through the HR portal at least "
    "7 days in advance.",
    # Page 3
    "Remote Work Policy\n\n"
    "Employees may work remotely up to 3 days per week with manager approval. "
    "Core collaboration hours are 11:00 to 16:00 India Standard Time.\n\n"
    "The company provides a one-time home-office allowance of INR 15,000. "
    "Remote employees must use the company VPN when accessing internal systems and must be "
    "reachable on Slack during core hours.",
    # Page 4
    "Expense Policy\n\n"
    "Meals during business travel are reimbursed up to INR 1,200 per day. "
    "Expense claims must be submitted within 30 days of purchase with original receipts.\n\n"
    "Air travel must be economy class for flights shorter than 6 hours. "
    "Expenses above INR 25,000 require approval from a department head. "
    "Reimbursement is paid within 10 business days after approval.",
]

ROUTER_PAGES = [
    "Orion X200 Router - Quick Start\n\n"
    "Package contents: router, power adapter (12V, 2A), 1 Ethernet cable, quick start guide.\n\n"
    "Connect the WAN port to your modem and power on the router. Open a browser and visit "
    "192.168.50.1 to reach the admin page. The default username is admin. You must change the "
    "default password at first login. The Wi-Fi network name is printed on the bottom label.",
    "Troubleshooting\n\n"
    "If the power LED blinks red, the router has lost its internet connection. Restart the modem "
    "and wait 2 minutes.\n\n"
    "To perform a factory reset, hold the reset button for 10 seconds until the LED flashes white.\n\n"
    "If Wi-Fi is slow, change the 2.4 GHz channel from Auto to 1, 6 or 11.\n\n"
    "Firmware updates are under Admin > System > Firmware. Do not power off during an update, "
    "which takes about 5 minutes.",
    "Specifications and Warranty\n\n"
    "Wi-Fi 6 dual-band AX1800: up to 574 Mbps on 2.4 GHz and 1201 Mbps on 5 GHz. "
    "The router has 4 Gigabit LAN ports and 1 Gigabit WAN port.\n\n"
    "Warranty: 2 years from the purchase date. It covers manufacturing defects only and does not "
    "cover water damage. Support email: support@orion-example.com. "
    "Operating temperature is 0 to 40 degrees Celsius.",
]

SAMPLES = {
    "acme_employee_handbook.pdf": HANDBOOK_PAGES,
    "orion_router_manual.pdf": ROUTER_PAGES,
}


def build_pdf(pages: list[str]) -> bytes:
    """Create a simple multi-page text PDF and return its bytes."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        rect = pymupdf.Rect(60, 60, 535, 780)
        leftover = page.insert_textbox(rect, text, fontsize=12, fontname="helv")
        if leftover < 0:
            raise ValueError("Sample page text does not fit on the page.")
    data = doc.tobytes()
    doc.close()
    return data


def write_samples(out_dir: Path) -> list[Path]:
    """Write all sample PDFs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, pages in SAMPLES.items():
        path = out_dir / name
        path.write_bytes(build_pdf(pages))
        paths.append(path)
    return paths


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "sample_docs"
    for written in write_samples(target):
        print(f"Wrote {written}")
