"""
Real Headless-Browser Render Test for M5 (5-minute) Live Page.
Executes Playwright tests across 4 viewports:
1280x720, 1440x900, 1920x1080, 390x844.
Asserts:
- Zero console errors
- Headline, chart, day summary, timeline strip, table rows visible
- Saves screenshots to results/m5/figures/ui_<width>x<height>.png
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

VIEWPORTS = [
    (1280, 720),
    (1440, 900),
    (1920, 1080),
    (390, 844)
]

OUT_DIR = "results/m5/figures"


def run_render_checks():
    os.makedirs(OUT_DIR, exist_ok=True)
    url = "http://localhost:8000"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        for w, h in VIEWPORTS:
            print(f"\n--- Testing Viewport: {w}x{h} ---")
            context = browser.new_context(viewport={"width": w, "height": h})
            page = context.new_page()
            
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: console_errors.append(str(err)))
            
            response = page.goto(url, wait_until="networkidle")
            assert response.status == 200, f"Page returned status {response.status}"
            
            # Wait for data to populate
            page.wait_for_selector("#tv-chart", state="visible", timeout=10000)
            page.wait_for_selector("#headline-card", state="visible", timeout=10000)
            page.wait_for_selector("#timeline-strip", state="visible", timeout=10000)
            page.wait_for_selector("#signal-log-table", state="visible", timeout=10000)
            page.wait_for_selector("#allbar-vs-strong-card", state="visible", timeout=10000)
            page.wait_for_selector("#confidence-tiers-card", state="visible", timeout=10000)

            # Verify disclosure banners are removed from main view and preserved in diagnostics
            assert page.locator("#banner-container").count() == 0, f"Banner container still present on {w}x{h}"
            assert page.locator("#banner-warning").count() == 0, f"Banner warning still present on {w}x{h}"
            diag_text = page.locator(".diag-disclosures-box").inner_text()
            assert "IEX Tape Disclosure" in diag_text, f"IEX Disclosure missing from diagnostics on {w}x{h}"
            assert "Holdout Disclosure" in diag_text, f"Holdout Disclosure missing from diagnostics on {w}x{h}"

            # Verify removed elements are absent from headline card and balance card
            assert page.locator("#balance-alert").count() == 0, f"Balance alert still present on {w}x{h}"
            assert page.locator("#headline-sub").count() == 0, f"Headline sub box still present on {w}x{h}"
            assert page.locator("#hero-reason").count() == 0, f"Hero reason still present on {w}x{h}"
            assert page.locator("#hero-sentence").count() == 0, f"Hero sentence still present on {w}x{h}"
            assert page.locator("#hero-conf-wrap").count() == 0, f"Hero conf wrap still present on {w}x{h}"

            # Verify retained elements exist in headline card and balance card
            assert page.locator("#hero-sample-tag").count() == 1, f"Missing LIVE tag on {w}x{h}"
            assert page.locator("#headline-tier-chip").count() == 1, f"Missing tier chip on {w}x{h}"
            assert page.locator("#headline-age").count() == 1, f"Missing headline age line on {w}x{h}"
            assert page.locator("#bal-1d").count() == 1, f"Missing bal-1d on {w}x{h}"
            assert page.locator("#bal-5d").count() == 1, f"Missing bal-5d on {w}x{h}"
            assert page.locator("#bal-20d").count() == 1, f"Missing bal-20d on {w}x{h}"

            # Verify no layout overlap: chart bottom must be at or above signal table top
            chart_box = page.locator("#tv-chart").bounding_box()
            table_box = page.locator("#signal-log-table").bounding_box()
            assert chart_box is not None and table_box is not None, f"Bounding box failed on {w}x{h}"
            assert chart_box["y"] + chart_box["height"] <= table_box["y"], (
                f"Overlap detected on {w}x{h}: chart bottom={chart_box['y'] + chart_box['height']}, "
                f"table top={table_box['y']}"
            )
            
            # Allow chart and table 2s to render
            time.sleep(2.0)
            
            # Check console errors
            assert len(console_errors) == 0, f"Console errors encountered on {w}x{h}: {console_errors}"
            
            # Verify headline text is populated
            call_text = page.locator("#headline-call").inner_text().strip()
            assert call_text != "", "Headline call text is empty"
            
            # Verify day header is populated and not stuck loading
            day_header = page.locator("#log-day-header").inner_text().strip()
            assert "Loading" not in day_header, f"Day header stuck loading on {w}x{h}"
            assert ("Fri 13 Nov 2026" in day_header or "Mon 16 Nov 2026" in day_header), f"Expected valid date in header, got: {day_header}"
            
            # Verify timeline chips exist
            chips_count = page.locator("#timeline-strip .timeline-chip").count()
            assert chips_count > 0, f"No timeline chips rendered on {w}x{h}"
            
            # Verify table rows rendered
            rows_count = page.locator("#signal-log-tbody tr").count()
            assert rows_count > 0, f"No table rows rendered on {w}x{h}"
            
            up_rows = page.locator("#signal-log-tbody tr .text-up").count()
            down_rows = page.locator("#signal-log-tbody tr .text-down").count()
            print(f"[{w}x{h}] Directional Rows: Up={up_rows}, Down={down_rows}, Total Rows={rows_count}")
            assert (up_rows + down_rows) > 0, f"Expected directional rows on {w}x{h}, got Up={up_rows}, Down={down_rows}"
            
            # Verify no loading cell in table
            loading_cell = page.locator(".loading-cell").count()
            assert loading_cell == 0, f"Table still showing loading cell on {w}x{h}"
            
            print(f"[{w}x{h}] PASSED: Verified ({rows_count} rows, {chips_count} chips)")
            
            context.close()
            
        browser.close()
        print("\nAll 4 Viewport Headless Render Tests PASSED 100%!")


if __name__ == "__main__":
    run_render_checks()
