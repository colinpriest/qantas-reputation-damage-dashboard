"""Test script to debug scraping of first company only"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def test_first_company():
    output_dir = Path("hesta_voting")
    output_dir.mkdir(exist_ok=True)

    url = "https://www.hesta.com.au/about-us/what-we-do/how-we-invest/sustainable-investment"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Step 1: Expand accordion
        print("Step 1: Expanding accordion...")
        page.wait_for_timeout(5000)  # Wait for page to fully load

        # Try to find the accordion button with text
        accordion_button = page.locator('text=Share voting at company AGMs').first
        accordion_button.scroll_into_view_if_needed()
        accordion_button.click()
        page.wait_for_timeout(3000)

        # Wait for panel to expand
        panel = page.locator('#sharevotingatcompanyagms_voting')
        panel.wait_for(state="visible", timeout=10000)
        print("Accordion expanded")

        # Step 2: Switch to iframe
        print("\nStep 2: Switching to iframe...")
        iframe_element = page.frame_locator('iframe[src*="glasslewis.com"]')
        page.wait_for_timeout(2000)

        # Step 3: Click first company
        print("\nStep 3: Clicking first company...")
        first_company_link = iframe_element.locator('tbody tr td a').first
        first_company_link.click()
        page.wait_for_timeout(3000)

        # Get company name
        company_elem = iframe_element.locator('h2#detail-issuer-name').first
        company_name = company_elem.inner_text().strip()
        print(f"Company: {company_name}")

        # Get meeting dropdown
        meeting_dropdown = iframe_element.locator('select#ddl-meetings-available').first
        meeting_options_elements = meeting_dropdown.locator('option').all()

        print(f"\nFound {len(meeting_options_elements)} meetings:")
        for i, opt in enumerate(meeting_options_elements):
            print(f"  {i+1}. {opt.inner_text().strip()}")

        # Select first meeting (it should already be selected)
        if len(meeting_options_elements) > 0:
            first_meeting_value = meeting_options_elements[0].get_attribute('value')
            meeting_dropdown.select_option(first_meeting_value)
            page.wait_for_timeout(2000)

            print(f"\nSelected meeting: {meeting_options_elements[0].inner_text().strip()}")

            # Get meeting details
            try:
                meeting_date_elem = iframe_element.locator('text=/Meeting Date:/').locator('..').first
                meeting_date = meeting_date_elem.inner_text().strip().replace('Meeting Date:', '').strip()
                print(f"Meeting Date: {meeting_date}")
            except:
                meeting_date = "Unknown"

            try:
                meeting_type_elem = iframe_element.locator('text=/Meeting Type:/').locator('..').first
                meeting_type = meeting_type_elem.inner_text().strip().replace('Meeting Type:', '').strip()
                print(f"Meeting Type: {meeting_type}")
            except:
                meeting_type = "Unknown"

            # Scrape voting table
            print("\n" + "="*80)
            print("VOTING TABLE DATA:")
            print("="*80)

            # Get all td cells from the voting table
            all_cells = iframe_element.locator('table#grid tbody td').all()
            print(f"\nTotal cells found: {len(all_cells)}")

            # Show first 20 cells
            print("\nFirst 20 cells:")
            for i in range(min(20, len(all_cells))):
                text = all_cells[i].inner_text().strip()
                print(f"  Cell {i}: '{text}'")

            # Extract votes
            print("\n" + "="*80)
            print("EXTRACTED VOTES:")
            print("="*80)

            i = 0
            vote_count = 0
            while i < len(all_cells):
                first_cell_text = all_cells[i].inner_text().strip()

                # Check if this is a vote row (item number)
                if first_cell_text and first_cell_text.replace('.', '').replace('A', '').replace('B', '').replace('C', '').replace('I', '').replace('V', '').isdigit():
                    if i + 4 < len(all_cells):
                        item_num = first_cell_text
                        proposal = all_cells[i + 1].inner_text().strip()
                        mgmt_rec = all_cells[i + 2].inner_text().strip()
                        proponent = all_cells[i + 3].inner_text().strip()
                        hesta_vote = all_cells[i + 4].inner_text().strip()

                        if proposal:
                            vote_count += 1
                            print(f"\nVote #{vote_count}:")
                            print(f"  Item: {item_num}")
                            print(f"  Proposal: {proposal[:60]}...")
                            print(f"  Mgmt Rec: {mgmt_rec}")
                            print(f"  Proponent: {proponent}")
                            print(f"  HESTA Vote: {hesta_vote}")

                        # Skip next 10 cells (5 for rationale row + 5 for next vote)
                        i += 10
                    else:
                        i += 1
                else:
                    i += 1

            print(f"\n{'='*80}")
            print(f"Total votes extracted: {vote_count}")
            print("="*80)

            # Save screenshot
            screenshot_path = output_dir / "test_first_company.png"
            page.screenshot(path=str(screenshot_path))
            print(f"\nScreenshot saved to: {screenshot_path}")

        input("\nPress Enter to close browser...")
        browser.close()

if __name__ == "__main__":
    test_first_company()
