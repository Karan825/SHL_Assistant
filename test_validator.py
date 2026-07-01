import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from agent.validator import validate_response

# Mock catalog (just a subset of catalog items for testing)
mock_catalog = [
    {
        "name": "Occupational Personality Questionnaire OPQ32r",
        "link": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "keys": ["Personality & Behavior"]
    },
    {
        "name": "Java 8 (New)",
        "link": "https://www.shl.com/products/product-catalog/view/java-8-new/",
        "keys": ["Knowledge & Skills"]
    },
    {
        "name": "SVAR Spoken English (US) (New)",
        "link": "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
        "keys": ["Simulations"]
    }
]

def run_tests():
    print("=" * 60)
    print("RUNNING VALIDATOR UNIT TESTS")
    print("=" * 60)
    
    # Test Case 1: Exact matches (should work perfectly)
    input_data_1 = {
        "reply": "Here are your assessments.",
        "recommendations": [
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P"
            }
        ],
        "end_of_conversation": False
    }
    res_1 = validate_response(input_data_1, mock_catalog)
    assert len(res_1["recommendations"]) == 1
    assert res_1["recommendations"][0]["url"] == "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"
    assert res_1["recommendations"][0]["test_type"] == "P"
    print("[PASS] Test Case 1: Exact match behaves correctly.")
    
    # Test Case 2: URL with trailing slash missing, lowercase/uppercase issues, wrong protocol
    input_data_2 = {
        "reply": "Here are recommendations.",
        "recommendations": [
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "http://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r",  # http, no trailing slash
                "test_type": "personality & behavior"  # full string test type
            }
        ],
        "end_of_conversation": False
    }
    res_2 = validate_response(input_data_2, mock_catalog)
    assert len(res_2["recommendations"]) == 1
    # Check that it corrected the URL to the canonical one from the catalog
    assert res_2["recommendations"][0]["url"] == "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"
    # Check that it corrected the test_type code
    assert res_2["recommendations"][0]["test_type"] == "P"
    print("[PASS] Test Case 2: Normalized URL and test type mapping works.")

    # Test Case 3: Name variation (spacing, casing, minor punctuation changes), URL completely omitted/hallucinated
    input_data_3 = {
        "reply": "Here is Java recommendation.",
        "recommendations": [
            {
                "name": "  java 8 (new)  ",  # extra spacing, lowercase
                "url": "https://www.shl.com/hallucinated-java-url/",  # hallucinated URL
                "test_type": "K"
            }
        ],
        "end_of_conversation": True
    }
    res_3 = validate_response(input_data_3, mock_catalog)
    assert len(res_3["recommendations"]) == 1
    # Check that it corrected the hallucinated URL using the name match
    assert res_3["recommendations"][0]["url"] == "https://www.shl.com/products/product-catalog/view/java-8-new/"
    assert res_3["recommendations"][0]["name"] == "Java 8 (New)"
    assert res_3["recommendations"][0]["test_type"] == "K"
    print("[PASS] Test Case 3: Match by name corrects hallucinated URL.")

    # Test Case 4: Substring matching
    input_data_4 = {
        "reply": "Simulations",
        "recommendations": [
            {
                "name": "SVAR Spoken English",  # substring of 'SVAR Spoken English (US) (New)'
                "url": "",
                "test_type": ""
            }
        ]
    }
    res_4 = validate_response(input_data_4, mock_catalog)
    assert len(res_4["recommendations"]) == 1
    assert res_4["recommendations"][0]["name"] == "SVAR Spoken English (US) (New)"
    assert res_4["recommendations"][0]["url"] == "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/"
    assert res_4["recommendations"][0]["test_type"] == "S"  # auto-derived from catalog 'Simulations' key!
    print("[PASS] Test Case 4: Substring name matching and automatic test type derivation from catalog keys works.")

    # Test Case 5: Hallucinated item with no matching URL or name (should be stripped)
    input_data_5 = {
        "reply": "Nonsense",
        "recommendations": [
            {
                "name": "Totally Fake Test Name",
                "url": "https://www.shl.com/fake-url/",
                "test_type": "K"
            }
        ]
    }
    res_5 = validate_response(input_data_5, mock_catalog)
    assert len(res_5["recommendations"]) == 0
    print("[PASS] Test Case 5: Completely fake recommendations are stripped correctly.")

    print("\n" + "=" * 60)
    print("ALL VALIDATOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
