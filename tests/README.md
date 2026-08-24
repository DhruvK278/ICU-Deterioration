# QA Pytest Suite: ICU Inference Server

This directory contains a genuine, end-to-end Quality Automation test suite for the FastAPI inference server (`src/fog/fog_server.py`).

The tests are designed to exercise the actual `/predict` endpoint and its ML inference logic without aggressive mocking, proving that the API properly validates incoming clinical payloads, enforces logical bounds, and securely handles anomalous or missing data.

## What is tested?

1. **Valid Requests**: Verifies the 200 OK happy path and ensures the response JSON matches the exact structural shape and data types defined by the contract (including risk score bounded ranges).
2. **Missing Required Fields**: Asserts that strict Pydantic models correctly identify omitted core identifiers (`hadm_id`) and return a fast, localized 422 error.
3. **Invalid Data Types**: Validates that bad coercions (e.g., strings injected into numeric clinical fields like `acuity_score`) are securely rejected with a 422 error before reaching the ML models.
4. **Clinical Range Enforcement**: Tests bounds constraints (`age` > 120), asserting graceful 422 failure to ensure the models aren't corrupted by impossible clinical values.
5. **Malformed Payload Boundaries**: Proves that garbage JSON strings fail securely at the framework boundary, protecting the server.
6. **Unvalidated Edge Cases (QA Insight)**: Specifically highlights `admit_type`, which currently lacks Enum enforcement. It proves the ML extraction gracefully falls back without crashing, while adding an explicit comment recommending tighter bounds.

## How to Run

1. **Install Testing Dependencies**:
   Ensure you have installed the required dependencies in your environment:
   ```bash
   pip install pytest httpx
   ```

2. **Execute the Suite**:
   Run the tests from the root of the project directory:
   ```bash
   python -m pytest tests/test_inference.py -v
   ```
