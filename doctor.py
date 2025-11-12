from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests

# ==============================
# Google Maps API Config
# ==============================
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Place your API key in .env

# ==============================
# Router Setup
# ==============================
doctor_router = APIRouter(prefix="/doctor", tags=["Doctor"])

# ==============================
# Response Models
# ==============================
class DoctorResponse(BaseModel):
    name: str
    city: str
    specialization: str
    bio: str
    available_slots: int

class AppointmentResponse(BaseModel):
    message: str

class DoctorProfileResponse(BaseModel):
    name: str
    city: str
    specialization: str
    bio: str
    available_slots: int
    booked_today: int

class MessageResponse(BaseModel):
    message: str

# ==============================
# Selenium Scraper
# ==============================
def fetch_justdial_doctors(city: str, specialization: str) -> List[dict]:
    formatted_city = city.replace(" ", "-")
    formatted_spec = specialization.replace(" ", "-")
    url = f"https://www.justdial.com/{formatted_city}/{formatted_spec}-Doctors/nct-10892680"

    chrome_options = Options()
    chrome_options.headless = True  # Set to False for debugging
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service('chromedriver.exe'), options=chrome_options)
    driver.get(url)

    doctors = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.col-sm-5.col-xs-8.store-details.sp-detail-padding"))
        )

        doctor_elements = driver.find_elements(By.CSS_SELECTOR, "div.col-sm-5.col-xs-8.store-details.sp-detail-padding")
        for doc in doctor_elements:
            name = doc.find_element(By.CSS_SELECTOR, "span.lng_cont_name").text if doc.find_elements(By.CSS_SELECTOR, "span.lng_cont_name") else "Unknown"
            address = doc.find_element(By.CSS_SELECTOR, "span.cont_fl_addr").text if doc.find_elements(By.CSS_SELECTOR, "span.cont_fl_addr") else "No address found"
            rating = doc.find_element(By.CSS_SELECTOR, "span.green-box").text if doc.find_elements(By.CSS_SELECTOR, "span.green-box") else "No rating"
            contact = doc.find_element(By.CSS_SELECTOR, "p.contact-info").text if doc.find_elements(By.CSS_SELECTOR, "p.contact-info") else "Not available"

            # Optionally fetch contact from Google Maps API if needed
            if GOOGLE_MAPS_API_KEY and "No address found" not in address:
                contact = fetch_contact_from_google_maps(address)

            doctors.append({
                "name": name,
                "city": city,
                "specialization": specialization,
                "bio": f"Address: {address} | Rating: {rating} | Contact: {contact}",
                "available_slots": 5
            })

    except Exception as e:
        print(f"Scraping error: {e}")
    finally:
        driver.quit()

    return doctors

# ==============================
# Google Maps Contact Fetcher
# ==============================
def fetch_contact_from_google_maps(query: str) -> str:
    try:
        params = {
            "query": query,
            "key": GOOGLE_MAPS_API_KEY
        }
        r = requests.get("https://maps.googleapis.com/maps/api/place/textsearch/json", params=params)
        data = r.json()
        if data.get("results"):
            place_id = data["results"][0]["place_id"]
            detail_params = {
                "place_id": place_id,
                "fields": "formatted_phone_number",
                "key": GOOGLE_MAPS_API_KEY
            }
            detail_r = requests.get("https://maps.googleapis.com/maps/api/place/details/json", params=detail_params)
            details = detail_r.json()
            phone = details.get("result", {}).get("formatted_phone_number", "Not available")
            return phone
    except Exception as e:
        print(f"Google Maps API error: {e}")
    return "Not available"

# ==============================
# 1️⃣ List Doctors
# ==============================
@doctor_router.get("/list", response_model=List[DoctorResponse])
def list_doctors(city: str = Query(..., description="City name"),
                 specialization: str = Query(..., description="Specialization")):
    try:
        doctors = fetch_justdial_doctors(city, specialization)
        if not doctors:
            raise HTTPException(status_code=404, detail="No doctors found for this city and specialization")
        return doctors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch doctors: {str(e)}")

# ==============================
# 2️⃣ Dummy Profile Endpoint
# ==============================
@doctor_router.get("/profile/{doctor_id}", response_model=DoctorProfileResponse)
def doctor_profile(doctor_id: int):
    return {
        "name": f"Dr. Dummy {doctor_id}",
        "city": "N/A",
        "specialization": "N/A",
        "bio": "Dummy bio",
        "available_slots": 5,
        "booked_today": 0
    }

# ==============================
# 3️⃣ Dummy Book Appointment
# ==============================
@doctor_router.post("/book", response_model=AppointmentResponse)
def book_appointment(doctor_id: int, patient_name: str):
    return {"message": f"✅ Appointment booked for {patient_name} with Doctor ID {doctor_id}"}
