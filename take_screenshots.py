import time
from playwright.sync_api import sync_playwright

def take_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        time.sleep(5)  # wait for streamlit to load
        page.screenshot(path="docs/images/main_ui.png", full_page=True)

        # Let's perform a search to get the results UI
        page.fill("input[aria-label='What are you looking for?']", "recsys")
        page.click("button:has-text('Find Papers')")
        time.sleep(5)
        page.screenshot(path="docs/images/search_results.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    take_screenshots()