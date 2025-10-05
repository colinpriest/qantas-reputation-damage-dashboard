"""
Download HESTA quarterly voting records (XLSX and PDF formats)
Downloads from 2017 Q1 to 2025 Q3
Uses Playwright to scrape interactive table on HESTA website
"""

import os
import requests
from pathlib import Path
from typing import List, Tuple, Dict
import time
import re
import json
from playwright.sync_api import sync_playwright

class HestaVotingDownloader:
    def __init__(self):
        self.base_url = "https://www.hesta.com.au/content/dam/hesta/Documents"
        self.output_dir = Path("hesta_voting")
        self.cache_file = self.output_dir / "scraped_records_cache.json"

        # Create output directory
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            print(f"Created directory: {self.output_dir}")

    def generate_quarters(self, start_year: int, start_quarter: int,
                         end_year: int, end_quarter: int) -> List[Tuple[int, int]]:
        """Generate list of (year, quarter) tuples"""
        quarters = []

        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                # Skip quarters before start
                if year == start_year and quarter < start_quarter:
                    continue
                # Skip quarters after end
                if year == end_year and quarter > end_quarter:
                    continue

                quarters.append((year, quarter))

        return quarters

    def build_url(self, year: int, quarter: int, file_type: str, pattern: str = 'underscore') -> str:
        """Build URL for a specific quarter and file type

        Args:
            year: Year (e.g., 2022)
            quarter: Quarter number (1-4)
            file_type: 'pdf' or 'xlsx'
            pattern: 'underscore' (2022+) or 'hyphen' (2017-2021)
        """
        if pattern == 'hyphen':
            # Pattern: HESTA-share-voting-record-Q1-2017.pdf (used 2017-2021)
            filename = f"HESTA-share-voting-record-Q{quarter}-{year}.{file_type}"
        else:
            # Pattern: HESTA_Share_Voting_Record_Q4_2022.pdf (used 2022+)
            filename = f"HESTA_Share_Voting_Record_Q{quarter}_{year}.{file_type}"

        return f"{self.base_url}/{filename}"

    def download_file(self, url: str, save_path: Path) -> bool:
        """Download a file from URL to save_path"""
        try:
            # Check if file already exists and is valid
            if save_path.exists():
                # Validate existing file
                if self.validate_file(save_path):
                    print(f"  [SKIP] {save_path.name} already exists")
                    return True
                else:
                    # Delete invalid file
                    save_path.unlink()
                    print(f"  [INVALID] Deleted corrupted {save_path.name}, re-downloading...")

            # Download file
            response = requests.get(url, timeout=30, allow_redirects=True)

            # Check if successful
            if response.status_code == 200:
                # Verify content type
                content_type = response.headers.get('content-type', '').lower()

                # Check if it's the expected file type
                if save_path.suffix == '.pdf':
                    # Check for PDF magic bytes
                    if not response.content.startswith(b'%PDF'):
                        print(f"  [404] {save_path.name} - not a valid PDF")
                        return False
                elif save_path.suffix == '.xlsx':
                    # Check for ZIP magic bytes (XLSX is a ZIP file)
                    if not response.content.startswith(b'PK'):
                        print(f"  [404] {save_path.name} - not a valid XLSX")
                        return False

                # Check minimum size (avoid error pages)
                if len(response.content) < 1000:  # Less than 1KB is suspicious
                    print(f"  [404] {save_path.name} - file too small ({len(response.content)} bytes)")
                    return False

                # Save file
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                print(f"  [OK] Downloaded {save_path.name} ({len(response.content):,} bytes)")
                return True
            elif response.status_code == 404:
                print(f"  [404] {save_path.name} not found")
                return False
            else:
                print(f"  [ERROR] {save_path.name} - HTTP {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] {save_path.name}")
            return False
        except Exception as e:
            print(f"  [ERROR] {save_path.name} - {e}")
            return False

    def validate_file(self, file_path: Path) -> bool:
        """Validate that a file is not corrupted"""
        try:
            if not file_path.exists():
                return False

            # Check file size
            if file_path.stat().st_size < 1000:
                return False

            # Check magic bytes
            with open(file_path, 'rb') as f:
                header = f.read(4)

                if file_path.suffix == '.pdf':
                    return header.startswith(b'%PDF')
                elif file_path.suffix == '.xlsx':
                    return header.startswith(b'PK')

            return True
        except Exception:
            return False

    def download_quarter(self, year: int, quarter: int) -> dict:
        """Download both PDF and XLSX for a specific quarter"""
        results = {'pdf': False, 'xlsx': False}

        print(f"\n{year} Q{quarter}:")

        # Determine URL pattern based on year
        # 2017-2021 uses hyphen pattern, 2022+ uses underscore pattern
        pattern = 'hyphen' if 2017 <= year <= 2021 else 'underscore'

        # Try PDF
        pdf_url = self.build_url(year, quarter, 'pdf', pattern)
        pdf_path = self.output_dir / f"HESTA_Q{quarter}_{year}.pdf"
        results['pdf'] = self.download_file(pdf_url, pdf_path)

        # If failed and year is 2017-2021, try alternate pattern
        if not results['pdf'] and 2017 <= year <= 2021:
            time.sleep(0.3)
            alternate_pattern = 'underscore'
            pdf_url_alt = self.build_url(year, quarter, 'pdf', alternate_pattern)
            print(f"  Trying alternate URL pattern...")
            results['pdf'] = self.download_file(pdf_url_alt, pdf_path)

        # Small delay between requests
        time.sleep(0.5)

        # Try XLSX
        xlsx_url = self.build_url(year, quarter, 'xlsx', pattern)
        xlsx_path = self.output_dir / f"HESTA_Q{quarter}_{year}.xlsx"
        results['xlsx'] = self.download_file(xlsx_url, xlsx_path)

        return results

    def download_all(self, start_year: int = 2010, start_quarter: int = 1,
                     end_year: int = 2025, end_quarter: int = 3):
        """Download all voting records from start to end"""
        print("=" * 80)
        print("HESTA Voting Records Downloader")
        print("=" * 80)
        print(f"Downloading from {start_year} Q{start_quarter} to {end_year} Q{end_quarter}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print()

        quarters = self.generate_quarters(start_year, start_quarter, end_year, end_quarter)

        stats = {
            'total_quarters': len(quarters),
            'pdf_success': 0,
            'xlsx_success': 0,
            'pdf_failed': 0,
            'xlsx_failed': 0
        }

        for year, quarter in quarters:
            results = self.download_quarter(year, quarter)

            if results['pdf']:
                stats['pdf_success'] += 1
            else:
                stats['pdf_failed'] += 1

            if results['xlsx']:
                stats['xlsx_success'] += 1
            else:
                stats['xlsx_failed'] += 1

            # Small delay between quarters
            time.sleep(0.5)

        # Print summary
        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"Total quarters: {stats['total_quarters']}")
        print(f"PDF files:")
        print(f"  - Downloaded: {stats['pdf_success']}")
        print(f"  - Failed/Not found: {stats['pdf_failed']}")
        print(f"XLSX files:")
        print(f"  - Downloaded: {stats['xlsx_success']}")
        print(f"  - Failed/Not found: {stats['xlsx_failed']}")
        print()

        # List downloaded files
        all_files = sorted(self.output_dir.glob("*"))
        if all_files:
            print(f"Downloaded {len(all_files)} files to {self.output_dir.absolute()}")
        else:
            print("No files were downloaded. Check if the URL pattern is correct.")

        return stats


    def load_cache(self) -> List[Dict]:
        """Load cached voting records"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} cached records from {self.cache_file}")
                    return data
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return []

    def save_cache(self, records: List[Dict]):
        """Save voting records to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(records, f, indent=2)
            print(f"Saved {len(records)} records to cache: {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def scrape_voting_table(self, use_cache: bool = True, force_rescrape_years: List[int] = None) -> List[Dict]:
        """Scrape voting records from HESTA's interactive table using Playwright

        Args:
            use_cache: If True, load from cache and only scrape missing years
            force_rescrape_years: List of years to force rescrape even if cached
        """
        print("\n" + "=" * 80)
        print("Scraping HESTA voting records table with Playwright")
        print("=" * 80)

        # Load cache
        voting_records = []
        cached_years = set()

        if use_cache:
            voting_records = self.load_cache()
            # Track which years we already have
            for record in voting_records:
                try:
                    year = int(record['year'])
                    cached_years.add(year)
                except (ValueError, KeyError):
                    pass

            if cached_years:
                print(f"Cached years: {sorted(cached_years)}")

        if force_rescrape_years is None:
            force_rescrape_years = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Visible for debugging
            page = browser.new_page()
            page.set_viewport_size({'width': 1920, 'height': 1080})

            url = "https://www.hesta.com.au/about-us/super-with-impact/investment-excellence-with-impact#active-ownership"
            print(f"Loading page: {url}")
            page.goto(url, timeout=60000)
            page.wait_for_timeout(3000)

            # Click to expand "Share voting at company AGMs" section
            print("Expanding 'Share voting at company AGMs' section...")
            try:
                # First, scroll the expand link into view
                expand_link = page.locator('a[href="#sharevotingatcompanyagms_voting"]').first
                expand_link.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)

                # Click to expand
                expand_link.click()
                page.wait_for_timeout(2000)

                # Check if panel expanded - it should have 'in' class or 'show' class
                # Sometimes Bootstrap uses 'collapse in' or 'collapse show'
                try:
                    page.wait_for_selector('#sharevotingatcompanyagms_voting.in, #sharevotingatcompanyagms_voting.show', timeout=5000)
                except:
                    # Check if it's visible by checking the display/height
                    panel = page.locator('#sharevotingatcompanyagms_voting').first
                    if not panel.is_visible():
                        # Try clicking again
                        print("  Panel not visible, clicking again...")
                        expand_link.click()
                        page.wait_for_timeout(2000)

                print("  Section expanded")

                # Wait for iframe to load
                print("  Waiting for iframe to load...")
                page.wait_for_selector('iframe[src*="glasslewis.com"]', timeout=10000)
                page.wait_for_timeout(2000)

                # Switch to iframe
                iframe_element = page.frame_locator('iframe[src*="glasslewis.com"]')
                print("  Switched to iframe context")

            except Exception as e:
                print(f"  Error: Could not expand section or find iframe: {e}")
                # Save debug info
                debug_screenshot = self.output_dir / "debug_error.png"
                debug_html = self.output_dir / "debug_error.html"
                page.screenshot(path=str(debug_screenshot))
                with open(debug_html, 'w', encoding='utf-8') as f:
                    f.write(page.content())
                print(f"  Debug: Saved screenshot to {debug_screenshot}")
                print(f"  Debug: Saved HTML to {debug_html}")
                browser.close()
                return voting_records

            # Determine which years to scrape
            all_years = [2017, 2018, 2019, 2020, 2021]
            years_to_scrape = []

            for year in all_years:
                if year in force_rescrape_years:
                    print(f"Will rescrape {year} (forced)")
                    # Remove existing records for this year
                    voting_records = [r for r in voting_records if r.get('year') != str(year)]
                    years_to_scrape.append(year)
                elif year not in cached_years:
                    print(f"Will scrape {year} (not cached)")
                    years_to_scrape.append(year)
                else:
                    print(f"Skipping {year} (already cached)")

            if not years_to_scrape:
                print("\nAll years already cached. Use force_rescrape_years to override.")
                browser.close()
                return voting_records

            print(f"\nYears to scrape: {years_to_scrape}")

            for year in years_to_scrape:
                print(f"\n{'='*80}")
                print(f"Scraping records for {year}")
                print(f"{'='*80}")

                try:
                        # Step 3: Click on the first company in the table
                        print(f"  Clicking first company in the table...")
                        first_company_link = iframe_element.locator('tbody tr td a').first
                        first_company_link.click()
                        page.wait_for_timeout(2000)  # Wait for detail page to load

                        found_this_year = 0
                        company_count = 0
                        max_companies = 500  # Safety limit
                        processed_companies = set()  # Track processed votes to avoid duplicates

                        # Step 4-6: Loop through companies using Next button
                        while company_count < max_companies:
                            company_count += 1
                            print(f"    Company {company_count}...")

                            try:
                                # Get company name from detail page
                                try:
                                    # Company name is in specific h2 element
                                    company_elem = iframe_element.locator('h2#detail-issuer-name').first
                                    current_company = company_elem.inner_text().strip()
                                except:
                                    current_company = f"Unknown_Company_{company_count}"

                                # Step 4: Get all meeting dates from dropdown
                                meeting_dropdown = None
                                meeting_options = []
                                try:
                                    meeting_dropdown = iframe_element.locator('select').first
                                    meeting_options_elements = meeting_dropdown.locator('option').all()
                                    meeting_options = [opt.get_attribute('value') for opt in meeting_options_elements]
                                    print(f"      Found {len(meeting_options)} meetings for {current_company}")
                                except:
                                    # No dropdown, just one meeting
                                    meeting_options = [None]

                                found_for_company = 0

                                # Step 5: For each meeting date
                                for meeting_idx, meeting_value in enumerate(meeting_options):
                                    try:
                                        # Select the meeting from dropdown if there are multiple
                                        if meeting_dropdown and meeting_value:
                                            meeting_dropdown.select_option(meeting_value)
                                            page.wait_for_timeout(1000)  # Wait for page to update

                                        # Get meeting info after selection
                                        try:
                                            meeting_date_elem = iframe_element.locator('span#txt-detail-info-Meeting-Date').first
                                            current_meeting_date = meeting_date_elem.inner_text().strip()
                                        except:
                                            current_meeting_date = ''

                                        try:
                                            meeting_type_elem = iframe_element.locator('span#txt-detail-info-Meeting-Type').first
                                            current_meeting_type = meeting_type_elem.inner_text().strip()
                                        except:
                                            current_meeting_type = ''

                                        # Extract year from meeting date
                                        meeting_year = str(year)  # Default to loop year
                                        if current_meeting_date:
                                            try:
                                                # Date format is YYYY-MM-DD
                                                meeting_year = current_meeting_date.split('-')[0]
                                            except:
                                                pass

                                        # Wait for voting table to load and scrape it
                                        page.wait_for_timeout(1000)

                                        # Try multiple table selectors to find the voting results
                                        voting_rows = []
                                        table_selectors = [
                                            'table.table tbody tr',  # Bootstrap table with tbody
                                            'div.table-responsive table tbody tr',  # Responsive table wrapper
                                            'table tbody tr',  # Generic table
                                            '.voting-results tbody tr',  # Class-based selector
                                            'table tr:has(td)'  # Any table rows with td cells
                                        ]

                                        for selector in table_selectors:
                                            try:
                                                rows = iframe_element.locator(selector).all()
                                                if len(rows) > 0:
                                                    # Check if first row has multiple cells (voting data)
                                                    first_row_cells = rows[0].locator('td').all()
                                                    if len(first_row_cells) >= 3:
                                                        voting_rows = rows
                                                        break
                                            except:
                                                continue

                                        # The HTML structure has td elements directly in tbody without tr wrappers
                                        # Each vote has 2 rows: vote data (5 cells) + rationale (3 cells)
                                        # Get all td elements
                                        all_cells = iframe_element.locator('table#grid tbody td').all()

                                        # Group cells into rows of 5 (Item, Proposal, Mgmt Rec, Proponent, Vote Decision)
                                        i = 0
                                        while i < len(all_cells):
                                            try:
                                                # Check if this is a vote row (should have item number in first cell)
                                                first_cell_text = all_cells[i].inner_text().strip()

                                                # Vote rows have item number, rationale rows are empty or have "Rationale:"
                                                if first_cell_text and first_cell_text.isdigit():
                                                    # This is a vote row - extract all 5 columns
                                                    if i + 4 < len(all_cells):
                                                        item_num = first_cell_text
                                                        proposal = all_cells[i + 1].inner_text().strip()
                                                        mgmt_rec = all_cells[i + 2].inner_text().strip()
                                                        proponent = all_cells[i + 3].inner_text().strip()
                                                        hesta_vote = all_cells[i + 4].inner_text().strip()

                                                        if proposal:  # Only save if there's a proposal
                                                            record = {
                                                                'company': current_company,
                                                                'meeting_date': current_meeting_date,
                                                                'meeting_type': current_meeting_type,
                                                                'motion_number': item_num,
                                                                'proposal': proposal,
                                                                'management_recommendation': mgmt_rec,
                                                                'hesta_vote': hesta_vote,
                                                                'year': meeting_year,
                                                                'source': 'glasslewis_detail'
                                                            }

                                                            row_key = f"{current_company}|{current_meeting_date}|{item_num}|{proposal}"
                                                            if row_key not in processed_companies:
                                                                processed_companies.add(row_key)
                                                                voting_records.append(record)
                                                                found_this_year += 1
                                                                found_for_company += 1

                                                        # Skip to next potential vote row (5 vote cells + 3 rationale cells = 8 total)
                                                        i += 8
                                                    else:
                                                        i += 1
                                                else:
                                                    # Not a vote row, move to next cell
                                                    i += 1

                                            except Exception:
                                                i += 1
                                                continue

                                    except Exception as e:
                                        print(f"        Error processing meeting {meeting_idx + 1}: {e}")
                                        continue

                                if found_for_company > 0:
                                    print(f"      Found {found_for_company} total votes for {current_company}")

                                # Step 5: Click Next button
                                next_button = None
                                try:
                                    next_button = iframe_element.locator('button:has-text("Next"), a:has-text("Next")').first
                                    if next_button.is_visible():
                                        next_button.click()
                                        page.wait_for_timeout(1500)
                                    else:
                                        print(f"    No more companies (Next button not visible)")
                                        break
                                except:
                                    print(f"    No more companies (Next button not found)")
                                    break

                            except Exception as e:
                                print(f"      Error processing company {company_count}: {e}")
                                break

                        print(f"  Total records found for {year}: {found_this_year}")

                except Exception as e:
                    print(f"  Error processing {year}: {e}")

                # Small delay between years
                time.sleep(1)

            browser.close()

        print(f"\n{'='*80}")
        print(f"Total voting records found: {len(voting_records)}")
        print(f"{'='*80}")

        # Save to cache
        if voting_records:
            self.save_cache(voting_records)

        return voting_records

    def download_scraped_records(self, records: List[Dict]) -> int:
        """Download voting records from scraped URLs (only for records with URLs)"""
        print("\n" + "=" * 80)
        print("Downloading scraped voting records")
        print("=" * 80)

        # Filter records to only those with URLs
        records_with_urls = [r for r in records if 'url' in r]

        if not records_with_urls:
            print("No records with URLs to download (table data only)")
            return 0

        success_count = 0

        for i, record in enumerate(records_with_urls, 1):
            url = record['url']
            year = record.get('year', 'unknown')
            period = record.get('period', 'unknown')

            # Generate filename
            filename = f"HESTA_{year}_{period}_scraped.pdf"
            # Clean up filename
            filename = re.sub(r'[^\w\-_.]', '_', filename)

            save_path = self.output_dir / filename

            print(f"\n[{i}/{len(records_with_urls)}] {filename}:")
            if self.download_file(url, save_path):
                success_count += 1

            time.sleep(0.5)

        print(f"\nScraped records: {success_count}/{len(records_with_urls)} downloaded successfully")
        return success_count

    def export_scraped_data_to_csv(self, records: List[Dict]) -> str:
        """Export scraped table data to CSV"""
        import csv

        csv_path = self.output_dir / "hesta_voting_records.csv"

        print(f"\nExporting {len(records)} records to CSV: {csv_path}")

        if not records:
            print("No records to export")
            return str(csv_path)

        # Define field order (not alphabetical)
        fieldnames = [
            'company',
            'meeting_date',
            'meeting_type',
            'motion_number',
            'proposal',
            'management_recommendation',
            'hesta_vote',
            'year',
            'source'
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"Exported {len(records)} records to {csv_path}")
        print(f"Columns: {', '.join(fieldnames)}")
        return str(csv_path)


if __name__ == "__main__":
    downloader = HestaVotingDownloader()

    # Step 1: Scrape voting records from interactive table (2017-2021)
    scraped_records = downloader.scrape_voting_table()

    # Step 2: Export scraped data to CSV (table data doesn't have PDF links)
    if scraped_records:
        downloader.export_scraped_data_to_csv(scraped_records)
    else:
        print("\nNo records found from web scraping")

    # Step 3: Also try direct quarterly downloads for recent records (2022+)
    print("\n" + "=" * 80)
    print("Attempting direct downloads for 2022+ quarterly records")
    print("=" * 80)
    downloader.download_all(
        start_year=2022,
        start_quarter=1,
        end_year=2025,
        end_quarter=3
    )
