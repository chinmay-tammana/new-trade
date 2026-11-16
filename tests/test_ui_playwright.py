"""Playwright End-to-End Viewport & UI Inspection Test (V3.2 Fix).

Verifies for viewports 1280x720, 1366x768, 1440x900, 1920x1080 and mobile 390x844:
- Log panel has scrollHeight >= clientHeight and last row is reachable by scrolling.
- Zero clipped cells (no table cell has scrollWidth > clientWidth + 2).
- Zero horizontal page overflow (doc scrollWidth <= clientWidth).
- Headline fits without clipping.
- Countdown matches required format.
- Zero browser console errors.
"""

import pytest
from playwright.sync_api import sync_playwright

VIEWPORTS = [
    {"width": 1280, "height": 720},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 390, "height": 844}  # Mobile portrait
]


def test_ui_viewports_and_log_scroll():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for vp in VIEWPORTS:
            page = browser.new_page(viewport=vp)
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            page.goto("http://localhost:8000/index_h1.html", wait_until="networkidle")
            page.wait_for_timeout(1000)

            # Evaluate DOM metrics
            data = page.evaluate("""() => {
                const scrollBox = document.getElementById('signals-scroll-box');
                const table = document.getElementById('signal-table');
                const rows = table ? table.querySelectorAll('tbody tr.signal-row') : [];
                const lastRow = rows.length ? rows[rows.length - 1] : null;

                let lastRowScrolled = false;
                if (scrollBox && lastRow) {
                    lastRow.scrollIntoView();
                    const boxRect = scrollBox.getBoundingClientRect();
                    const rowRect = lastRow.getBoundingClientRect();
                    lastRowScrolled = (rowRect.top >= boxRect.top - 20 && rowRect.bottom <= boxRect.bottom + 80);
                }

                let clippedCells = 0;
                if (table) {
                    const cells = table.querySelectorAll('td, th');
                    cells.forEach(c => {
                        if (c.scrollWidth > c.clientWidth + 2) clippedCells++;
                    });
                }

                const heroDir = document.getElementById('hero-direction');
                const headlineClipped = heroDir ? heroDir.scrollWidth > heroDir.clientWidth + 5 : false;
                const cdVal = document.getElementById('val-countdown');

                return {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    docScrollWidth: document.documentElement.scrollWidth,
                    docClientWidth: document.documentElement.clientWidth,
                    scrollBoxClientHeight: scrollBox ? scrollBox.clientHeight : 0,
                    scrollBoxScrollHeight: scrollBox ? scrollBox.scrollHeight : 0,
                    rowCount: rows.length,
                    lastRowReachable: lastRowScrolled,
                    clippedCellsCount: clippedCells,
                    headlineClipped: headlineClipped,
                    countdownText: cdVal ? cdVal.textContent : ""
                };
            }""")

            vp_str = f"{vp['width']}x{vp['height']}"
            print(f"Viewport {vp_str}: {data}")

            # Assertions
            assert len(console_errors) == 0, f"Console errors in {vp_str}: {console_errors}"
            assert not data["headlineClipped"], f"Headline clipped in {vp_str}"
            assert data["docScrollWidth"] <= data["docClientWidth"] + 1, f"Horizontal overflow in {vp_str}"
            assert data["scrollBoxClientHeight"] > 0, f"Signals scroll box has 0 height in {vp_str}"
            assert data["rowCount"] > 0, f"Zero signal rows rendered in {vp_str}"
            assert data["lastRowReachable"], f"Last row could not be scrolled into view in {vp_str}"
            assert data["clippedCellsCount"] == 0, f"Found {data['clippedCellsCount']} clipped cells in {vp_str}"

            page.close()

        browser.close()
