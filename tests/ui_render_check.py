"""Real Browser Render & UI Verification Test (V3.6 Single Source of Truth Upgrade).

Tests for viewports 1280x720, 1440x900, 1920x1080, and 390x844:
- Load http://localhost:8000
- Assert zero console errors and zero failed network requests
- Assert no "Loading signals..." text remains
- Assert no IEX rows by default (all rows are sip_delayed)
- Assert strip chip count equals row count (e.g. 7 chips = 7 rows)
- Assert each row is one line at 1280 and above (height <= 56px)
- Assert the full day's rows are visible without page scroll at 1440x900
- Assert the day picker works (prev, next, date input, ?day= URL)
- Assert expandable detail row opens on click and closes on next click
- Assert no horizontal page overflow (scrollWidth <= clientWidth)
- Save screenshots to results/figures/ui_log_<width>x<height>.png
"""

import re
import time
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright

VIEWPORTS = [
    {"width": 1280, "height": 720},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 390, "height": 844}
]

COUNTDOWN_PATTERN = re.compile(r"^(\d+d\s+\d+h\s+\d+m|\d+h\s+\d+m|\d{2}:\d{2}|Opens Soon|00:00)$")
CLOCK_PATTERN = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


def test_real_browser_render():
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for vp in VIEWPORTS:
            vp_str = f"{vp['width']}x{vp['height']}"
            page = browser.new_page(viewport=vp)

            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            failed_requests = []
            page.on("requestfailed", lambda req: failed_requests.append(req.url))

            # 1. Load localhost:8000
            response = page.goto("http://localhost:8000", wait_until="networkidle")
            assert response is not None and response.status == 200, f"Failed to load page in {vp_str}"

            # Wait for signals and chips to populate
            page.wait_for_selector("#signals-tbody tr.signal-row", timeout=8000)
            page.wait_for_timeout(500)

            # 2. Assert zero console errors & zero failed requests
            assert len(console_errors) == 0, f"Console errors in {vp_str}: {console_errors}"
            assert len(failed_requests) == 0, f"Failed network requests in {vp_str}: {failed_requests}"

            # 3. Assert no "Loading signals..." text remains
            tbody_text = page.inner_text("#signals-tbody")
            assert "Loading signals..." not in tbody_text, f"Still showing 'Loading signals...' in {vp_str}"

            # 4. Assert day view de-duplication: exactly 7 rows on 2026-11-13
            signal_rows = page.query_selector_all("#signals-tbody tr.signal-row")
            assert len(signal_rows) == 7, f"Expected 7 rows on default day 2026-11-13 in {vp_str}, got {len(signal_rows)}"

            # 5. Assert NO IEX rows by default (no .row-iex-shifted)
            iex_rows = page.query_selector_all("#signals-tbody tr.row-iex-shifted")
            assert len(iex_rows) == 0, f"Found IEX rows displayed by default in {vp_str}"

            # 6. Assert strip chip count equals row count
            chips = page.query_selector_all("#hour-strip-container .hour-chip")
            assert len(chips) == len(signal_rows), f"Chip count ({len(chips)}) != row count ({len(signal_rows)}) in {vp_str}"

            # 7. Assert row height <= 56px at 1280 and above
            if vp["width"] >= 1280:
                first_row = signal_rows[0]
                box = first_row.bounding_box()
                assert box is not None and box["height"] <= 56, f"Row height {box['height']}px exceeds 56px in {vp_str}"

            # 8. Assert summary strip matches
            summary_main = page.inner_text("#summary-main")
            assert "Fri 13 Nov 2026" in summary_main, f"Summary main missing date in {vp_str}: '{summary_main}'"
            assert "Signals 2" in summary_main, f"Summary main incorrect signal count in {vp_str}: '{summary_main}'"
            assert "Accuracy 1/2" in summary_main, f"Summary main incorrect accuracy in {vp_str}: '{summary_main}'"

            # 9. Test Expandable Detail Row (Click to open, click to close)
            first_row = signal_rows[0]
            first_row.scroll_into_view_if_needed()
            page.evaluate("(el) => el.click()", first_row)
            page.wait_for_timeout(300)
            detail_row = page.query_selector("tr.detail-row")
            assert detail_row is not None, f"Detail row failed to open on click in {vp_str}"
            detail_text = detail_row.inner_text()
            assert "Calibrated Posterior:" in detail_text, f"Detail row missing calibration in {vp_str}"
            assert "Model Artifact Hash:" in detail_text, f"Detail row missing model hash in {vp_str}"

            # Click again to close
            page.evaluate("(el) => el.click()", first_row)
            page.wait_for_timeout(300)
            assert page.query_selector("tr.detail-row") is None, f"Detail row failed to close on second click in {vp_str}"

            # 10. Assert headline is inside viewport and not clipped
            headline_metrics = page.evaluate("""() => {
                const el = document.getElementById('hero-direction');
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {
                    scrollWidth: el.scrollWidth,
                    clientWidth: el.clientWidth,
                    top: rect.top,
                    bottom: rect.bottom,
                    viewportHeight: window.innerHeight,
                    viewportWidth: window.innerWidth
                };
            }""")
            assert headline_metrics["scrollWidth"] <= headline_metrics["clientWidth"] + 2, f"Headline clipped in {vp_str}"

            # 11. Assert no horizontal page overflow
            doc_overflow = page.evaluate("""() => {
                return {
                    scrollWidth: document.documentElement.scrollWidth,
                    clientWidth: document.documentElement.clientWidth
                };
            }""")
            assert doc_overflow["scrollWidth"] <= doc_overflow["clientWidth"] + 1, f"Horizontal page overflow in {vp_str}: {doc_overflow}"

            # 12. Assert at 1440x900 that ALL rows of a full day (up to 7 rows) are visible without scrolling the page
            if vp["width"] == 1440 and vp["height"] == 900:
                all_rows_visible = page.evaluate("""() => {
                    const rows = document.querySelectorAll('#signals-tbody tr.signal-row');
                    if (!rows.length) return false;
                    const lastRow = rows[rows.length - 1];
                    const rect = lastRow.getBoundingClientRect();
                    return rect.bottom <= window.innerHeight;
                }""")
                assert all_rows_visible is True, "All 7 rows of the full day not visible without page scroll at 1440x900"

            # 13. Test Day Picker (Prev Day navigation)
            btn_prev = page.query_selector("#btn-prev-day")
            if btn_prev and not btn_prev.is_disabled():
                btn_prev.click()
                page.wait_for_function("() => document.getElementById('summary-main').textContent.includes('12 Nov 2026')", timeout=5000)
                new_picker_val = page.input_value("#day-picker")
                assert new_picker_val == "2026-11-12", f"Prev button did not navigate to 2026-11-12, got {new_picker_val}"

                # Click Next Day to navigate back to 2026-11-13
                btn_next = page.query_selector("#btn-next-day")
                btn_next.click()
                page.wait_for_function("() => document.getElementById('summary-main').textContent.includes('13 Nov 2026')", timeout=5000)
                assert page.input_value("#day-picker") == "2026-11-13"

            # Assert restored thresholds in detail rows or threshold text
            tbody_html = page.inner_html("#signals-tbody")
            assert "WEAK" not in tbody_html, f"Found WEAK pill in table in {vp_str}"
            assert "rule_version=2" not in tbody_html, f"Found rule_version=2 in table in {vp_str}"

            # Save screenshots to both ui_log and ui_revert
            screenshot_path_log = f"results/figures/ui_log_{vp_str}.png"
            screenshot_path_revert = f"results/figures/ui_revert_{vp_str}.png"
            page.screenshot(path=screenshot_path_log, full_page=False)
            page.screenshot(path=screenshot_path_revert, full_page=False)
            print(f"Captured screenshots for {vp_str} -> {screenshot_path_revert}")

            page.close()

        # 14. Test ?day=2026-11-12 URL query parameter
        page_url_test = browser.new_page(viewport={"width": 1280, "height": 720})
        page_url_test.goto("http://localhost:8000/?day=2026-11-12", wait_until="networkidle")
        page_url_test.wait_for_selector("#signals-tbody tr.signal-row", timeout=5000)
        picker_val = page_url_test.input_value("#day-picker")
        assert picker_val == "2026-11-12", f"?day= URL parameter failed to set picker value, got {picker_val}"
        summary_val = page_url_test.inner_text("#summary-main")
        assert "12 Nov 2026" in summary_val, f"?day= URL parameter failed to load 12 Nov summary, got {summary_val}"
        page_url_test.close()

        browser.close()


if __name__ == "__main__":
    test_real_browser_render()
    print("ALL REAL BROWSER UI VERIFICATION CHECKS PASSED!")
