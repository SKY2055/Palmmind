import requests
import json
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def wait_for_server():
    print("Waiting for server to be ready...")
    for _ in range(30):
        try:
            resp = requests.get(f"http://localhost:8000/health")
            if resp.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def test_name_collision():
    print("\n--- Testing Name Collision ---")
    
    # 1. Upload Document 1 (John Doe - Python Developer)
    doc1 = (
        "Name: John Doe\n"
        "Email: john.python@example.com\n"
        "Skills: Python, FastAPI, SQL\n"
        "Experience: 5 years at TechCorp"
    )
    with open("john1.txt", "w") as f: f.write(doc1)
    
    with open("john1.txt", "rb") as f:
        files = {"file": ("john1.txt", f, "text/plain")}
        resp = requests.post(f"{BASE_URL}/documents/upload", files=files)
    print(f"Upload 1 response: {resp.status_code}")
    
    # 2. Upload Document 2 (John Doe - Java Developer)
    doc2 = (
        "Name: John Doe\n"
        "Email: john.java@example.com\n"
        "Skills: Java, Spring Boot, Microservices\n"
        "Experience: 3 years at CodeLabs"
    )
    with open("john2.txt", "w") as f: f.write(doc2)
    
    with open("john2.txt", "rb") as f:
        files = {"file": ("john2.txt", f, "text/plain")}
        resp = requests.post(f"{BASE_URL}/documents/upload", files=files)
    print(f"Upload 2 response: {resp.status_code}")
    
    # 3. Query "Tell me about John Doe"
    chat_payload = {"message": "Tell me about John Doe"}
    resp = requests.post(f"{BASE_URL}/chat/", json=chat_payload)
    data = resp.json()
    print(f"Initial query response: {data['response']}")
    session_id = data['session_id']
    
    # 4. Clarify (Select John with Python)
    chat_payload = {"message": "The first one", "session_id": session_id}
    resp = requests.post(f"{BASE_URL}/chat/", json=chat_payload)
    print(f"Clarified query response: {resp.json()['response']}")
    
    # Cleanup
    os.remove("john1.txt")
    os.remove("john2.txt")

def test_interview_booking_flow():
    print("\n--- Testing Interview Booking Flow ---")
    
    session_id = f"test-session-{int(time.time())}"
    
    # 1. Partial booking
    chat_payload = {"message": "I want to book an interview for John Doe", "session_id": session_id}
    resp = requests.post(f"{BASE_URL}/chat/", json=chat_payload)
    print(f"Partial booking 1: {resp.json()['response']}")
    
    # 2. Add email
    chat_payload = {"message": "email is john@example.com", "session_id": session_id}
    resp = requests.post(f"{BASE_URL}/chat/", json=chat_payload)
    print(f"Partial booking 2: {resp.json()['response']}")
    
    # 3. Add date and time
    chat_payload = {"message": "2024-12-25 at 10:00 AM", "session_id": session_id}
    resp = requests.post(f"{BASE_URL}/chat/", json=chat_payload)
    print(f"Final booking: {resp.json()['response']}")

if __name__ == "__main__":
    if wait_for_server():
        test_name_collision()
        test_interview_booking_flow()
    else:
        print("Server not available.")
