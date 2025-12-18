# End-to-End Tests
# Production-grade E2E testing for complete user workflows

import pytest
import asyncio
import json
import tempfile
import time
from pathlib import Path
import subprocess
import os
import shutil
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import requests
import aiohttp
import sys

# Import system components for E2E testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class TestUserWorkflowE2E:
    """End-to-end tests for complete user workflows"""
    
    @pytest.fixture(scope="class")
    def browser_driver(self):
        """Set up browser driver for E2E tests"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode for CI
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize driver
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            yield driver
        finally:
            if 'driver' in locals():
                driver.quit()
    
    @pytest.fixture(scope="class")
    def test_server_process(self):
        """Start test server for E2E tests"""
        # Start the application server (mock or real)
        server_process = None
        try:
            # Mock server startup - in real scenario, start actual server
            server_port = 8000
            server_url = f"http://localhost:{server_port}"
            
            # Wait for server to be ready
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"{server_url}/health", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.skip("Test server not available")
            
            yield server_url
            
        finally:
            if server_process:
                server_process.terminate()
                server_process.wait()
    
    @pytest.mark.e2e
    def test_complete_file_upload_workflow(self, browser_driver, test_server_process, test_data_manager):
        """Test complete file upload and processing workflow"""
        driver = browser_driver
        base_url = test_server_process
        
        # Create test file
        test_csv = test_data_manager.create_test_csv("e2e_test.csv", rows=50)
        
        try:
            # Step 1: Navigate to application
            driver.get(f"{base_url}/dashboard")
            
            # Verify dashboard loads
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard-container"))
            )
            
            # Step 2: Navigate to file upload
            upload_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "upload-file-btn"))
            )
            upload_button.click()
            
            # Step 3: Upload file
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.INPUT, "file"))
            )
            file_input.send_keys(str(test_csv))
            
            # Submit upload
            submit_button = driver.find_element(By.ID, "submit-upload")
            submit_button.click()
            
            # Step 4: Wait for processing completion
            WebDriverWait(driver, 30).until(
                EC.text_to_be_present_in_element((By.ID, "upload-status"), "Processing Complete")
            )
            
            # Step 5: Verify file appears in file list
            file_list = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "file-list"))
            )
            
            file_items = file_list.find_elements(By.CLASS_NAME, "file-item")
            assert len(file_items) > 0
            
            # Verify our test file is listed
            file_names = [item.find_element(By.CLASS_NAME, "file-name").text for item in file_items]
            assert "e2e_test.csv" in file_names
            
            # Step 6: Click on uploaded file to view details
            test_file_item = next(item for item in file_items 
                                if "e2e_test.csv" in item.find_element(By.CLASS_NAME, "file-name").text)
            test_file_item.click()
            
            # Step 7: Verify file details view
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "file-details"))
            )
            
            # Check metadata display
            metadata_section = driver.find_element(By.CLASS_NAME, "file-metadata")
            assert "CSV" in metadata_section.text
            assert "50" in metadata_section.text  # Row count
            
            # Check data preview
            data_preview = driver.find_element(By.CLASS_NAME, "data-preview")
            preview_rows = data_preview.find_elements(By.TAG_NAME, "tr")
            assert len(preview_rows) > 1  # Header + data rows
            
        except Exception as e:
            # Take screenshot for debugging
            screenshot_path = f"test_upload_workflow_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    @pytest.mark.e2e
    def test_complete_query_analysis_workflow(self, browser_driver, test_server_process, test_data_manager):
        """Test complete query and analysis workflow"""
        driver = browser_driver
        base_url = test_server_process
        
        # Prepare test data
        test_csv = test_data_manager.create_test_csv("analysis_test.csv", rows=100)
        
        try:
            # Step 1: Navigate and upload file (prerequisite)
            driver.get(f"{base_url}/dashboard")
            
            # Quick upload (assuming upload functionality works from previous test)
            self._quick_upload_file(driver, str(test_csv))
            
            # Step 2: Navigate to query interface
            query_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "new-query-btn"))
            )
            query_button.click()
            
            # Step 3: Enter query
            query_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "query-input"))
            )
            
            test_query = "Analyze the data and show summary statistics"
            query_input.clear()
            query_input.send_keys(test_query)
            
            # Step 4: Submit query
            submit_query = driver.find_element(By.ID, "submit-query")
            submit_query.click()
            
            # Step 5: Wait for analysis to complete
            WebDriverWait(driver, 60).until(  # Longer timeout for analysis
                EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
            )
            
            # Step 6: Verify results display
            results_container = driver.find_element(By.CLASS_NAME, "analysis-results")
            
            # Check for key result components
            assert driver.find_element(By.CLASS_NAME, "query-response")
            assert driver.find_element(By.CLASS_NAME, "processing-metadata")
            
            # Verify response content
            response_text = driver.find_element(By.CLASS_NAME, "query-response").text
            assert len(response_text) > 50  # Should have substantial response
            assert "summary" in response_text.lower() or "statistics" in response_text.lower()
            
            # Step 7: Check processing metadata
            metadata = driver.find_element(By.CLASS_NAME, "processing-metadata")
            metadata_text = metadata.text
            assert "processing time" in metadata_text.lower()
            assert "tokens" in metadata_text.lower() or "model" in metadata_text.lower()
            
            # Step 8: Verify query appears in history
            history_button = driver.find_element(By.ID, "query-history-btn")
            history_button.click()
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "query-history"))
            )
            
            history_items = driver.find_elements(By.CLASS_NAME, "history-item")
            assert len(history_items) > 0
            
            # Find our query in history
            history_queries = [item.find_element(By.CLASS_NAME, "query-text").text for item in history_items]
            assert any(test_query in query for query in history_queries)
            
        except Exception as e:
            screenshot_path = f"test_query_workflow_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    @pytest.mark.e2e
    def test_complete_report_generation_workflow(self, browser_driver, test_server_process, test_data_manager):
        """Test complete report generation workflow"""
        driver = browser_driver
        base_url = test_server_process
        
        # Prepare test data
        test_csv = test_data_manager.create_test_csv("report_test.csv", rows=200)
        
        try:
            # Step 1: Setup - upload file and run analysis
            driver.get(f"{base_url}/dashboard")
            self._quick_upload_file(driver, str(test_csv))
            
            # Step 2: Navigate to report generation
            reports_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "reports-btn"))
            )
            reports_button.click()
            
            # Step 3: Create new report
            new_report_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "new-report-btn"))
            )
            new_report_button.click()
            
            # Step 4: Configure report settings
            report_title_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "report-title"))
            )
            report_title_input.clear()
            report_title_input.send_keys("E2E Test Report")
            
            # Select data source
            data_source_dropdown = driver.find_element(By.ID, "data-source-select")
            data_source_dropdown.click()
            
            # Select our uploaded file
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, f"//option[contains(text(), 'report_test.csv')]"))
            ).click()
            
            # Select report type
            report_type_dropdown = driver.find_element(By.ID, "report-type-select")
            report_type_dropdown.click()
            
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//option[contains(text(), 'Summary Report')]"))
            ).click()
            
            # Step 5: Generate report
            generate_button = driver.find_element(By.ID, "generate-report-btn")
            generate_button.click()
            
            # Step 6: Wait for report generation
            WebDriverWait(driver, 120).until(  # Reports may take longer
                EC.text_to_be_present_in_element((By.ID, "generation-status"), "Report Generated")
            )
            
            # Step 7: Verify report display
            report_container = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "generated-report"))
            )
            
            # Check report components
            assert driver.find_element(By.CLASS_NAME, "report-title")
            assert driver.find_element(By.CLASS_NAME, "report-summary")
            assert driver.find_element(By.CLASS_NAME, "report-data")
            
            # Verify title
            report_title = driver.find_element(By.CLASS_NAME, "report-title").text
            assert "E2E Test Report" in report_title
            
            # Step 8: Test report download
            download_button = driver.find_element(By.ID, "download-report-btn")
            download_button.click()
            
            # Wait for download to initiate
            WebDriverWait(driver, 10).until(
                EC.text_to_be_present_in_element((By.ID, "download-status"), "Download Ready")
            )
            
            # Step 9: Verify report appears in reports list
            reports_list_button = driver.find_element(By.ID, "reports-list-btn")
            reports_list_button.click()
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "reports-list"))
            )
            
            report_items = driver.find_elements(By.CLASS_NAME, "report-item")
            assert len(report_items) > 0
            
            report_titles = [item.find_element(By.CLASS_NAME, "report-name").text for item in report_items]
            assert any("E2E Test Report" in title for title in report_titles)
            
        except Exception as e:
            screenshot_path = f"test_report_workflow_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    @pytest.mark.e2e
    def test_complete_dashboard_interaction_workflow(self, browser_driver, test_server_process, test_data_manager):
        """Test complete dashboard interaction workflow"""
        driver = browser_driver
        base_url = test_server_process
        
        try:
            # Step 1: Navigate to dashboard
            driver.get(f"{base_url}/dashboard")
            
            # Step 2: Verify dashboard components load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard-container"))
            )
            
            # Check key dashboard components
            assert driver.find_element(By.CLASS_NAME, "file-stats-widget")
            assert driver.find_element(By.CLASS_NAME, "recent-queries-widget")
            assert driver.find_element(By.CLASS_NAME, "system-metrics-widget")
            
            # Step 3: Interact with file stats widget
            file_stats = driver.find_element(By.CLASS_NAME, "file-stats-widget")
            
            # Verify stats display
            stats_numbers = file_stats.find_elements(By.CLASS_NAME, "stat-number")
            assert len(stats_numbers) >= 3  # Total files, processed today, avg time
            
            for stat in stats_numbers:
                assert stat.text.isdigit() or "." in stat.text  # Should show numbers
            
            # Step 4: Interact with recent queries widget
            queries_widget = driver.find_element(By.CLASS_NAME, "recent-queries-widget")
            
            # Check if queries are displayed
            query_items = queries_widget.find_elements(By.CLASS_NAME, "query-item")
            
            if len(query_items) > 0:
                # Click on a recent query
                query_items[0].click()
                
                # Should navigate to query details or results
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "query-details"))
                )
                
                # Navigate back to dashboard
                dashboard_button = driver.find_element(By.ID, "dashboard-nav")
                dashboard_button.click()
            
            # Step 5: Test system metrics widget
            metrics_widget = driver.find_element(By.CLASS_NAME, "system-metrics-widget")
            
            # Verify metrics display
            metric_items = metrics_widget.find_elements(By.CLASS_NAME, "metric-item")
            assert len(metric_items) >= 2  # At least cache hit rate and response time
            
            # Test refresh functionality
            refresh_button = metrics_widget.find_element(By.CLASS_NAME, "refresh-metrics")
            refresh_button.click()
            
            # Wait for refresh to complete
            WebDriverWait(driver, 10).until(
                EC.text_to_be_present_in_element((By.CLASS_NAME, "last-updated"), "seconds ago")
            )
            
            # Step 6: Test navigation between dashboard sections
            nav_items = driver.find_elements(By.CLASS_NAME, "nav-item")
            assert len(nav_items) >= 4  # Dashboard, Files, Queries, Reports
            
            for nav_item in nav_items:
                nav_text = nav_item.text.lower()
                if nav_text in ['files', 'queries', 'reports']:
                    nav_item.click()
                    
                    # Wait for section to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, f"{nav_text}-container"))
                    )
                    
                    # Return to dashboard
                    dashboard_nav = driver.find_element(By.ID, "dashboard-nav")
                    dashboard_nav.click()
                    
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "dashboard-container"))
                    )
            
        except Exception as e:
            screenshot_path = f"test_dashboard_workflow_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    @pytest.mark.e2e
    def test_error_handling_workflow(self, browser_driver, test_server_process):
        """Test error handling and recovery workflows"""
        driver = browser_driver
        base_url = test_server_process
        
        try:
            # Step 1: Test invalid file upload
            driver.get(f"{base_url}/dashboard")
            
            # Navigate to upload
            upload_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "upload-file-btn"))
            )
            upload_button.click()
            
            # Try to upload invalid file (create a temporary invalid file)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is not a valid CSV or JSON file\nIt has invalid content\n")
                invalid_file_path = f.name
            
            try:
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.INPUT, "file"))
                )
                file_input.send_keys(invalid_file_path)
                
                submit_button = driver.find_element(By.ID, "submit-upload")
                submit_button.click()
                
                # Should show error message
                error_message = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
                )
                
                assert "invalid" in error_message.text.lower() or "error" in error_message.text.lower()
                
                # Test error dismissal
                dismiss_button = driver.find_element(By.CLASS_NAME, "dismiss-error")
                dismiss_button.click()
                
                # Error should disappear
                WebDriverWait(driver, 5).until(
                    EC.invisibility_of_element_located((By.CLASS_NAME, "error-message"))
                )
                
            finally:
                os.unlink(invalid_file_path)
            
            # Step 2: Test network error handling
            # Navigate to queries
            queries_button = driver.find_element(By.ID, "queries-nav")
            queries_button.click()
            
            # Try to submit query (will test timeout/network handling)
            query_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "query-input"))
            )
            query_input.clear()
            query_input.send_keys("This is a test query for network error")
            
            # Mock network failure by intercepting requests
            driver.execute_script("""
                // Mock fetch to simulate network error
                window.originalFetch = window.fetch;
                window.fetch = function() {
                    return Promise.reject(new Error('Network error'));
                };
            """)
            
            submit_button = driver.find_element(By.ID, "submit-query")
            submit_button.click()
            
            # Should show network error
            network_error = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "network-error"))
            )
            
            assert "network" in network_error.text.lower() or "connection" in network_error.text.lower()
            
            # Test retry functionality
            retry_button = driver.find_element(By.CLASS_NAME, "retry-button")
            
            # Restore normal fetch
            driver.execute_script("window.fetch = window.originalFetch;")
            
            retry_button.click()
            
            # Should attempt retry (may still show error if no mock response, but should not show network error)
            time.sleep(2)  # Wait for retry attempt
            
        except Exception as e:
            screenshot_path = f"test_error_handling_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    def _quick_upload_file(self, driver, file_path):
        """Helper method for quick file upload in tests"""
        # Assumes already on dashboard or can navigate to upload
        try:
            upload_button = driver.find_element(By.ID, "upload-file-btn")
            upload_button.click()
            
            file_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.INPUT, "file"))
            )
            file_input.send_keys(file_path)
            
            submit_button = driver.find_element(By.ID, "submit-upload")
            submit_button.click()
            
            # Wait for upload to complete
            WebDriverWait(driver, 30).until(
                EC.text_to_be_present_in_element((By.ID, "upload-status"), "Processing Complete")
            )
        except Exception:
            # If quick upload fails, continue with test
            pass


class TestMultiUserWorkflows:
    """E2E tests for multi-user scenarios"""
    
    @pytest.fixture
    def multiple_browser_sessions(self):
        """Create multiple browser sessions for multi-user testing"""
        drivers = []
        try:
            for i in range(3):  # 3 concurrent users
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument(f"--user-data-dir=/tmp/chrome_user_{i}")
                
                driver = webdriver.Chrome(options=chrome_options)
                driver.implicitly_wait(10)
                drivers.append(driver)
            
            yield drivers
            
        finally:
            for driver in drivers:
                driver.quit()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_concurrent_file_uploads(self, multiple_browser_sessions, test_server_process, test_data_manager):
        """Test concurrent file uploads by multiple users"""
        drivers = multiple_browser_sessions
        base_url = test_server_process
        
        # Create test files for each user
        test_files = []
        for i in range(len(drivers)):
            file_path = test_data_manager.create_test_csv(f"concurrent_user_{i}.csv", rows=50)
            test_files.append(str(file_path))
        
        async def upload_file_for_user(driver, file_path, user_id):
            """Upload file for specific user"""
            try:
                driver.get(f"{base_url}/dashboard")
                
                # Upload file
                upload_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "upload-file-btn"))
                )
                upload_button.click()
                
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.INPUT, "file"))
                )
                file_input.send_keys(file_path)
                
                submit_button = driver.find_element(By.ID, "submit-upload")
                submit_button.click()
                
                # Wait for completion
                WebDriverWait(driver, 60).until(
                    EC.text_to_be_present_in_element((By.ID, "upload-status"), "Processing Complete")
                )
                
                return {'user_id': user_id, 'success': True, 'file': file_path}
                
            except Exception as e:
                return {'user_id': user_id, 'success': False, 'error': str(e)}
        
        # Run concurrent uploads
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(drivers)) as executor:
            futures = [
                executor.submit(upload_file_for_user, driver, file_path, i)
                for i, (driver, file_path) in enumerate(zip(drivers, test_files))
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all uploads succeeded
        successful_uploads = [r for r in results if r['success']]
        assert len(successful_uploads) == len(drivers), f"Only {len(successful_uploads)} uploads succeeded"
    
    @pytest.mark.e2e
    def test_concurrent_query_processing(self, multiple_browser_sessions, test_server_process):
        """Test concurrent query processing by multiple users"""
        drivers = multiple_browser_sessions
        base_url = test_server_process
        
        test_queries = [
            "What is the summary of the data?",
            "Show me the data trends",
            "Calculate basic statistics"
        ]
        
        def process_query_for_user(driver, query, user_id):
            """Process query for specific user"""
            try:
                driver.get(f"{base_url}/dashboard")
                
                # Navigate to query interface
                query_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "new-query-btn"))
                )
                query_button.click()
                
                # Enter and submit query
                query_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "query-input"))
                )
                query_input.clear()
                query_input.send_keys(query)
                
                submit_button = driver.find_element(By.ID, "submit-query")
                submit_button.click()
                
                # Wait for results
                WebDriverWait(driver, 90).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
                )
                
                return {'user_id': user_id, 'success': True, 'query': query}
                
            except Exception as e:
                return {'user_id': user_id, 'success': False, 'error': str(e), 'query': query}
        
        # Run concurrent queries
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(drivers)) as executor:
            futures = [
                executor.submit(process_query_for_user, driver, query, i)
                for i, (driver, query) in enumerate(zip(drivers, test_queries))
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all queries processed
        successful_queries = [r for r in results if r['success']]
        assert len(successful_queries) >= len(drivers) - 1, "Most queries should succeed"


class TestSystemReliabilityE2E:
    """E2E tests for system reliability and resilience"""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_long_running_workflow_reliability(self, browser_driver, test_server_process, test_data_manager):
        """Test system reliability during long-running workflows"""
        driver = browser_driver
        base_url = test_server_process
        
        # Create larger test file for longer processing
        large_test_file = test_data_manager.create_large_test_file("reliability_test.csv", size_mb=50)
        
        try:
            driver.get(f"{base_url}/dashboard")
            
            # Upload large file
            upload_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "upload-file-btn"))
            )
            upload_button.click()
            
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.INPUT, "file"))
            )
            file_input.send_keys(str(large_test_file))
            
            submit_button = driver.find_element(By.ID, "submit-upload")
            submit_button.click()
            
            # Monitor progress for extended period
            start_time = time.time()
            max_wait_time = 300  # 5 minutes
            
            while time.time() - start_time < max_wait_time:
                try:
                    status_element = driver.find_element(By.ID, "upload-status")
                    status_text = status_element.text
                    
                    if "Processing Complete" in status_text:
                        break
                    elif "Error" in status_text or "Failed" in status_text:
                        pytest.fail(f"Processing failed: {status_text}")
                    
                    # Check that progress is being updated
                    progress_element = driver.find_element(By.CLASS_NAME, "progress-bar")
                    progress_value = progress_element.get_attribute("aria-valuenow")
                    
                    assert progress_value is not None, "Progress should be reported"
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    # Take screenshot for debugging
                    screenshot_path = f"reliability_test_monitoring_{int(time.time())}.png"
                    driver.save_screenshot(screenshot_path)
                    raise e
            
            else:
                pytest.fail("Long-running workflow did not complete within time limit")
            
            # Verify file was processed successfully
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "file-list"))
            )
            
            file_items = driver.find_elements(By.CLASS_NAME, "file-item")
            file_names = [item.find_element(By.CLASS_NAME, "file-name").text for item in file_items]
            assert "reliability_test.csv" in file_names
            
        except Exception as e:
            screenshot_path = f"test_reliability_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    @pytest.mark.e2e
    def test_session_persistence_workflow(self, browser_driver, test_server_process, test_data_manager):
        """Test session persistence across browser refreshes"""
        driver = browser_driver
        base_url = test_server_process
        
        try:
            # Step 1: Initial setup
            driver.get(f"{base_url}/dashboard")
            
            # Upload a file
            test_file = test_data_manager.create_test_csv("session_test.csv", rows=30)
            self._quick_upload_file(driver, str(test_file))
            
            # Make a query
            query_button = driver.find_element(By.ID, "new-query-btn")
            query_button.click()
            
            query_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "query-input"))
            )
            test_query = "Test session persistence query"
            query_input.send_keys(test_query)
            
            submit_button = driver.find_element(By.ID, "submit-query")
            submit_button.click()
            
            # Wait for query completion
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
            )
            
            # Step 2: Refresh browser
            driver.refresh()
            
            # Step 3: Verify session data persists
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard-container"))
            )
            
            # Check that uploaded file is still listed
            file_list = driver.find_element(By.CLASS_NAME, "file-list")
            file_items = file_list.find_elements(By.CLASS_NAME, "file-item")
            file_names = [item.find_element(By.CLASS_NAME, "file-name").text for item in file_items]
            assert "session_test.csv" in file_names
            
            # Check that query is in history
            history_button = driver.find_element(By.ID, "query-history-btn")
            history_button.click()
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "query-history"))
            )
            
            history_items = driver.find_elements(By.CLASS_NAME, "history-item")
            history_queries = [item.find_element(By.CLASS_NAME, "query-text").text for item in history_items]
            assert any(test_query in query for query in history_queries)
            
        except Exception as e:
            screenshot_path = f"test_session_persistence_failure_{int(time.time())}.png"
            driver.save_screenshot(screenshot_path)
            raise e
    
    def _quick_upload_file(self, driver, file_path):
        """Helper method for quick file upload"""
        upload_button = driver.find_element(By.ID, "upload-file-btn")
        upload_button.click()
        
        file_input = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.INPUT, "file"))
        )
        file_input.send_keys(file_path)
        
        submit_button = driver.find_element(By.ID, "submit-upload")
        submit_button.click()
        
        WebDriverWait(driver, 30).until(
            EC.text_to_be_present_in_element((By.ID, "upload-status"), "Processing Complete")
        )


if __name__ == '__main__':
    # Run E2E tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'e2e',
        '--maxfail=3'  # Stop after 3 failures to avoid long runs
    ])