
# Smart AI Traffic Management - Vadodara

## Revolutionizing Urban Mobility for a Smarter City

 **Project Status:** Prototype

## 1\. Project Overview

The "Smart AI Traffic Management - Vadodara" is a pioneering prototype of a comprehensive, AI-driven web application designed to address critical urban mobility challenges in Vadodara, Gujarat, India. Born from a personal observation of daily traffic congestion, inefficient signal timings, and prevalent waterlogging issues in my city, this project aims to create a more intelligent, efficient, and safer traffic ecosystem through cutting-edge technology and citizen engagement.

It provides a holistic platform for smart route planning, predictive signal timing, emergency routing, and proactive hazard reporting, tailored specifically for the dynamic urban environment of Vadodara.

## 2\. Problem Statement

Urbanization presents significant challenges for traffic management, leading to:

  * **Severe Traffic Congestion:** Resulting in lost time, increased fuel consumption, and environmental pollution.
  * **Inefficient Traffic Signal Operations:** Static timings fail to adapt to real-time traffic flows, causing bottlenecks.
  * **Delayed Emergency Response:** Hindered by unpredictable traffic, impacting critical services.
  * **Pervasive Road Hazards:** Specifically, waterlogging (a common issue in Vadodara) often goes unreported, leading to accidents and disruptions.
  * **Lack of Real-time Information:** Preventing citizens from making informed travel decisions.

## 3\. Solution

The "Smart AI Traffic Management - Vadodara" system tackles these problems by leveraging:

  * **Predictive AI:** For intelligent route optimization and dynamic traffic signal timing.
  * **Real-time Data Integration:** To inform decisions and provide up-to-date information.
  * **Community Engagement:** Through a citizen-driven hazard reporting system.
  * **Intuitive Interfaces:** Including a voice assistant for enhanced accessibility.

## 4\. Key Features & Functionalities

This project offers a robust suite of features designed to enhance urban traffic:

  * **AI-Driven Smart Route Planning & Suggestion:** Predicts traffic density using ML models and utilizes graph algorithms (`networkx`) to suggest optimal routes, considering real-time conditions.
  * **Predictive Traffic Signal Timing:** Implements ML models to dynamically predict ideal red, green, and yellow light durations, significantly reducing intersection delays.
  * **Real-time Emergency Route Finder:** Instantly identifies the fastest and clearest routes to nearby hospitals, crucial for emergency services.
  * **Citizen-Powered Water Hazard Reporting System:** Empowers users to report waterlogging incidents with location and image uploads, enabling proactive city response and safer routes.
  * **Intelligent Peak Hours Travel Suggestions:** Provides strategic advice and alternative travel options for navigating high-traffic periods.
  * **Interactive Voice Assistant (Powered by Gemini API):** Offers hands-free navigation queries, traffic updates, and system interaction.
  * **Live News & Alerts:** Aggregates and displays real-time news and critical traffic alerts relevant to Vadodara.

## 5\. Technical Stack

The project is developed using a robust combination of modern web and AI technologies:

  * **Backend:** Python (Flask, Flask-SQLAlchemy, `joblib`, `requests`, `feedparser`, `dotenv`).
  * **Machine Learning:** Pre-trained models for traffic prediction and signal timing.
  * **Graph Processing:** `networkx` for complex road network analysis.
  * **Geocoding:** Nominatim OpenStreetMap API.
  * **Frontend:** HTML, CSS (Bootstrap 5.3.3), JavaScript, Leaflet.js (for interactive mapping), Font Awesome.
  * **Generative AI:** Google's Gemini API (`google.generativeai`).
  * **Data Handling:** Pandas for data manipulation.

## 6\. File Structure

The project follows a logical and organized file structure for clarity and maintainability:

# Project Structure

```
Traffic-Management-System/
├── Backend/
│   ├── Flask File/
│   │   └── app.py                # Main Flask application
│   ├── Dataset/
│   │   ├── Full_Vadodara_Traffic_Signal_Timings.csv
│   │   ├── Vadodara_Smart_Route_Suggestions.csv
│   │   ├── Vadodara_Traffic_Hourly_Expanded.csv
│   │   └── traffic_processed.csv
│   └── Frontend/
│       └── main.html             # Frontend template
│
├── ML Models/
│   ├── green_time_model.pkl      # Green signal duration model
│   ├── label_encoders.pkl        # Label encoders for categorical data
│   ├── red_time_model.pkl        # Red signal duration model
│   ├── traffic_model.pkl         # Main traffic prediction model
│   ├── traffic_prediction_model.pkl  # Secondary traffic model
│   └── yellow_time_constant.txt  # Fixed yellow signal duration
│
├── LICENSE                       # Project license
└── README.md                     # Project documentation (this file)
```

## Key Files Description

### Backend
- `app.py`: Main Flask application containing all routes and business logic
- Datasets:
  - Traffic signal timings
  - Route suggestions data
  - Hourly traffic data
  - Processed traffic data

### ML Models
- Pretrained models for traffic signal timing predictions
- Label encoders for categorical data
- Traffic prediction models

### Frontend
- `main.html`: Primary frontend template

## 7\. Installation & Setup (Conceptual - Details to be added)

  * **Prerequisites:** Python 3.x, pip.
  * **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd Smart-AI-Traffic-Management-Vadodara
    ```
  * **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
  
  * **Environment Variables:** Create a `.env` file in the root directory and add your `GEMINI_API_KEY`.
    ```
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```
  * **Run the application:**
    ```bash
    python app.py
    ```
    (The application will typically run on `http://127.0.0.1:5000/`)

## 8\. Known Issues & Future Enhancements (Currently Under Development)

During development, I encountered challenges, including an API cache issue, which I am actively addressing. My current focus for enhancement includes:

  * **Dynamic Alternate Route Finding:** Enhancing the route planner to dynamically calculate and suggest alternate routes that actively bypass detected waterlogging locations.
  * **Proximity-Based Hazard Alerts:** Implementing real-time notifications for users whose current location is near reported water hazards.
  * **AI-Powered Image Verification:** Integrating a CNN-based AI model to automatically verify waterlogging levels from uploaded images, improving data reliability.

## 9\. Real-World Impact & Vision for Vadodara City

This project directly addresses tangible problems faced by Vadodara's citizens daily, with a profound real-world impact:

  * **Solves Congestion:** Achieves smoother commutes and reduced travel times through smart routes and optimized traffic signals.
  * **Enhances Safety & Reduces Accidents:** Mitigates risks from flooded roads via the water hazard reporting system, dynamic routing, and image verification.
  * **Improves Emergency Response:** Ensures quicker access to critical services by offering immediate, clear routes to hospitals.
  * **Empowers Citizens:** Provides residents with a voice in urban management and real-time, actionable travel intelligence.
  * **Drives Smart City Initiatives:** Showcases Vadodara's potential for technological innovation in urban planning, setting a precedent for future smart infrastructure.

## 10\. Demo & Connect

  * **Demo Video:** Watch a quick demonstration of the project's functionalities here: https://drive.google.com/file/d/1vmOhfAF7MRoSB_4vkKvDBLA7mFhVX9mX/view?usp=drive_link
  * **Connect with me on LinkedIn:** https://www.linkedin.com/in/dhrumil-pawar/


