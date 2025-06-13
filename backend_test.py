#!/usr/bin/env python3
"""
TDS Virtual Teaching Assistant API Test Script
Tests the backend API endpoints and functionality
"""

import requests
import json
import base64
import sys
from PIL import Image
import io
import time

class TDSVirtualTAAPITester:
    def __init__(self, base_url="https://869b4550-f736-4890-ba78-42f41f6b116b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status=200, data=None, check_response=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            status_success = response.status_code == expected_status
            
            # Check response content if a check function is provided
            content_success = True
            if status_success and check_response:
                content_success = check_response(response.json())
            
            success = status_success and content_success
            
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                if response.status_code != 204:  # No content
                    print(f"Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"❌ Failed - Expected status {expected_status}, got {response.status_code}")
                if not status_success and response.status_code != 204:
                    try:
                        print(f"Error response: {json.dumps(response.json(), indent=2)}")
                    except:
                        print(f"Raw response: {response.text}")
                elif not content_success:
                    print("Response validation failed")
                    print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            self.test_results.append({
                "name": name,
                "success": success,
                "status_code": response.status_code,
                "expected_status": expected_status
            })
            
            return success, response.json() if response.status_code != 204 and response.text else {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        def check_response(data):
            return "message" in data and "TDS Virtual Teaching Assistant API" in data["message"]
        
        return self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200,
            check_response=check_response
        )

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        def check_response(data):
            return (
                "status" in data and 
                "index_loaded" in data and 
                "documents_count" in data and
                "openai_configured" in data
            )
        
        return self.run_test(
            "Health Check Endpoint",
            "GET",
            "health",
            200,
            check_response=check_response
        )

    def test_ask_endpoint_basic(self):
        """Test the ask endpoint with a basic question"""
        def check_response(data):
            return (
                "answer" in data and 
                isinstance(data["answer"], str) and
                "links" in data and
                isinstance(data["links"], list)
            )
        
        return self.run_test(
            "Ask Endpoint - Basic Question",
            "POST",
            "ask",
            200,
            data={"question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"},
            check_response=check_response
        )

    def test_ask_endpoint_missing_data(self):
        """Test the ask endpoint with a question about missing data"""
        def check_response(data):
            return (
                "answer" in data and 
                isinstance(data["answer"], str) and
                "links" in data and
                isinstance(data["links"], list)
            )
        
        return self.run_test(
            "Ask Endpoint - Missing Data Question",
            "POST",
            "ask",
            200,
            data={"question": "How do I handle missing data in my assignment?"},
            check_response=check_response
        )

    def test_ask_endpoint_visualization(self):
        """Test the ask endpoint with a question about visualization"""
        def check_response(data):
            return (
                "answer" in data and 
                isinstance(data["answer"], str) and
                "links" in data and
                isinstance(data["links"], list)
            )
        
        return self.run_test(
            "Ask Endpoint - Visualization Question",
            "POST",
            "ask",
            200,
            data={"question": "What are the best practices for data visualization?"},
            check_response=check_response
        )

    def test_ask_endpoint_ml_algorithms(self):
        """Test the ask endpoint with a question about ML algorithms"""
        def check_response(data):
            return (
                "answer" in data and 
                isinstance(data["answer"], str) and
                "links" in data and
                isinstance(data["links"], list)
            )
        
        return self.run_test(
            "Ask Endpoint - ML Algorithms Question",
            "POST",
            "ask",
            200,
            data={"question": "How do I choose between different machine learning algorithms?"},
            check_response=check_response
        )

    def test_ask_endpoint_with_image(self):
        """Test the ask endpoint with an image (should work in fallback mode)"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        def check_response(data):
            return (
                "answer" in data and 
                isinstance(data["answer"], str) and
                "links" in data and
                isinstance(data["links"], list)
            )
        
        return self.run_test(
            "Ask Endpoint - With Image",
            "POST",
            "ask",
            200,
            data={
                "question": "What does this image show?",
                "image": img_base64
            },
            check_response=check_response
        )

    def test_ask_endpoint_empty_question(self):
        """Test the ask endpoint with an empty question (should fail)"""
        return self.run_test(
            "Ask Endpoint - Empty Question",
            "POST",
            "ask",
            422,  # Validation error
            data={"question": ""}
        )

    def test_status_endpoint(self):
        """Test the status check endpoint"""
        def check_response(data):
            return isinstance(data, list)
        
        return self.run_test(
            "Status Check Endpoint - GET",
            "GET",
            "status",
            200,
            check_response=check_response
        )

    def test_create_status_check(self):
        """Test creating a status check"""
        def check_response(data):
            return (
                "id" in data and
                "client_name" in data and
                "timestamp" in data
            )
        
        return self.run_test(
            "Status Check Endpoint - POST",
            "POST",
            "status",
            200,
            data={"client_name": "API Tester"},
            check_response=check_response
        )

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting TDS Virtual TA API Tests")
        
        # Basic endpoint tests
        self.test_root_endpoint()
        self.test_health_endpoint()
        
        # Status endpoint tests
        self.test_status_endpoint()
        self.test_create_status_check()
        
        # Ask endpoint tests
        self.test_ask_endpoint_basic()
        self.test_ask_endpoint_missing_data()
        self.test_ask_endpoint_visualization()
        self.test_ask_endpoint_ml_algorithms()
        self.test_ask_endpoint_with_image()
        self.test_ask_endpoint_empty_question()
        
        # Print summary
        print("\n📊 Test Summary:")
        print(f"Tests passed: {self.tests_passed}/{self.tests_run} ({self.tests_passed/self.tests_run*100:.1f}%)")
        
        # Print failed tests
        failed_tests = [test for test in self.test_results if not test["success"]]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"- {test['name']}")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    # Get base URL from environment or use default
    base_url = "https://869b4550-f736-4890-ba78-42f41f6b116b.preview.emergentagent.com"
    
    tester = TDSVirtualTAAPITester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
