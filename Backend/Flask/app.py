# import Library : 
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS
from functools import lru_cache
import logging
import numpy as np
import google.generativeai as genai
from math import radians, sin, cos, sqrt, atan2
import random
from datetime import datetime
import json
import pandas as pd # Import pandas to read CSV
import networkx as nx # for Smart Route Suggestion Feature .
from flask_sqlalchemy import SQLAlchemy  # use for interact with data Sql database 
# for Hazard Report  :
from werkzeug.utils import secure_filename  
from datetime import datetime, date
import re # for text processing (remove tag from News)
# for fatch the data from RSS News Type (vadodara)
import feedparser
import sys

from logging.handlers import RotatingFileHandler

# --- Nominatim Geocoding API ---
NOMINATIM_URL = "https://nominatim.openstreetmap.com/search" # <--- Define it here GLOBALLY


# Remove all existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)  # File output
    ]
)

# Get the specific logger named 'traffic_app' and ensure its level is DEBUG
logger = logging.getLogger('traffic_app')
logger.setLevel(logging.DEBUG)


# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#----------------------------------------------------  Water HAZARD FEATURE ----------------------------------------------------------------------
#------------------------------------------- Setup For hazard Reports (Water logging Report)-------------------------------------------------------------------------------

# define models_dir (path of the model folder of system) :
base_dir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(base_dir, 'models')  

data_dir = os.path.join(base_dir, 'data')      
database_dir = os.path.join(base_dir, 'database') 


# This line will create the 'database' folder if it doesn't already exist.
os.makedirs(database_dir, exist_ok=True)
logger.info(f"Ensured database directory exists: {database_dir}")

# Database Creation in the specified directory:
db_path = os.path.join(database_dir, 'site.db') 
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # disable a flask sqlAlchemy warning .
db = SQLAlchemy(app)
logger.info(f"Database configured at {db_path}") # Log the exact path for verification

# Database Model for Water Hazard Report :
class WaterHazardReport(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    latitude = db.Column(db.Float, nullable = True)  
    longitude = db.Column (db.Float, nullable = True) 
    location_name = db.Column(db.String(255),nullable = True)
    image_filename = db.Column(db.String(255),nullable = False)
    timestamp = db.Column(db.DateTime, default = datetime.utcnow)
    
# column for Day, Date and time Category : 
    report_date = db.Column(db.Date, default = datetime.utcnow) # Store the Date only 
    report_day_of_week = db.Column(db.String(20), default=lambda:datetime.utcnow().strftime('%A'))  #  Stores day name ex. 'Monday'
    time_of_day_category = db.Column(db.String(20),nullable = True) # show 'morning', 'afternoon', 'evening', 'night'
    
    status = db.Column(db.String(50), default = 'active')
    
    
    def __repr__(self):
        return f'<WaterhazardReport {self.id} - {self.location_name} - {self.status} >'
    
    def serialize(self):
        return{
            'id' : self.id,
            'latitude' : self.latitude,
            'longitude' : self.longitude,
            'location_name' : self.location_name,
            'image_url' : f'/uploads/{self.image_filename}',
            'timestamp' : self.timestamp.isoformat(),
            'report_date' : self.report_date.isoformat() if self.report_date else None,
            'report_day_of_week' :self.report_day_of_week,
            'time_of_day_category':self.time_of_day_category,
            'status' : self.status
        }
        
# Create the new Endpoint route for report_water_hazard :
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png','jpg', 'jpeg','gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logger.info(f"Upload folder configured at : {UPLOAD_FOLDER}")

# Function for Correct Extension :
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#-------------------------------------- EndPoint FOr Water Hazard Report ----------------------------------
@app.route('/report-water-hazard', methods=['POST'])
def report_water_hazard():
    try:
        # --- Initial Checks for the uploaded file ---
        if 'hazard_image' not in request.files:
            logger.warning("No 'hazard_image' file part in request.")
            return jsonify({"status": "error", "message": "No image file provided."}), 400
        
        file = request.files['hazard_image'] # Correctly reference the uploaded file here
        
        if file.filename == '':
            logger.warning("Empty filename received for image upload.")
            return jsonify({"status": "error", "message": "No image selected."}), 400
        
        if not allowed_file(file.filename): # Check if file type is allowed
            logger.warning(f"File type not allowed or invalid file: {file.filename}")
            return jsonify({"status": "error", "message": "Invalid file type. Allowed: png, jpg, jpeg, gif."}), 400


        # --- Securely save the image  ---
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename) 
        
        try:
            file.save(image_path) # Changed from image_file to file
            logger.info(f"Image saved to {image_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return jsonify({"status": "error", "message": f"Failed to save image: {e}"}), 500

        # --- Get other form data and parse latitude/longitude ---
        location_name = request.form.get('location_name')


        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        try:
            latitude = float(latitude) if latitude else None
            longitude = float(longitude) if longitude else None
        except ValueError:
            latitude = None
            longitude = None
            logger.warning(f"Invalid latitude/longitude received: lat={request.form.get('latitude')}, lon={request.form.get('longitude')}")
        
        # --- Determine time of day category ---
        current_hour = datetime.utcnow().hour
        time_of_day_category = ""
        if 5 <= current_hour < 12 :
            time_of_day_category = "morning"
        elif 12 <= current_hour < 17:
            time_of_day_category = "afternoon"
        elif 17 <= current_hour < 21 :
            time_of_day_category = "evening"
        else:
            time_of_day_category = "night"
            

        new_report = WaterHazardReport(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            image_filename=unique_filename,
            timestamp=datetime.utcnow(),
            report_date=datetime.utcnow().date(),
            report_day_of_week=datetime.utcnow().strftime('%A'),
            time_of_day_category=time_of_day_category,
            status='active', 
        )
        
        try:
            db.session.add(new_report)
            db.session.commit()
            logger.info(f"Water hazard report saved: {new_report.id}")
            return jsonify({"status": "success", "message": "Water hazard report submitted successfully!", "report_id": new_report.id}), 201
        except Exception as e:
            db.session.rollback() # Revert changes if anything goes wrong
            logger.error(f"Failed to save water hazard report to DB: {e}")
            return jsonify({"status": "error", "message": f"Failed to save report: {e}"}), 500
            
    except Exception as e: # This outer try-except catches any unhandled errors in the function
        db.session.rollback() # Ensure rollback if any error occurs before session.add
        logger.error(f"Unhandled error in report_water_hazard: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
    
# ------------------------------ Load ML Models and Data (This block runs once when the app starts) -----------------------------------------

traffic_prediction_model = None
label_encoders = None
df = None # Initialize df as None globally
label_encoders = {} # Dictionary to hold label encoders
unique_signal_locations = [] # To hold original labels for mapping
signal_location_mapping = {} # Maps encoded values (str) to original labels

# ------------------------------------------------------------- Load ML Models Path and CSV Path  -------------------------------------------

try:
    models_dir = "models"
    data_dir = "data"
    
    # Traffic Prediction Model (for Smart Route Suggestion) & Traffic Signal Prediction -----------------------------------------
    traffic_model = joblib.load(os.path.join(models_dir, r"traffic_model.pkl"))


#--------------------------------------------------- Model For Traffic Signal Time Prediction---------------------------------------------------

    # Signal Timing Models
    signal_red_model = joblib.load(os.path.join(models_dir, r"red_time_model.pkl"))
    signal_green_model = joblib.load(os.path.join(models_dir, r"green_time_model.pkl"))
    with open(os.path.join(models_dir, r"yellow_time_constant.txt"), 'r') as f:
        YELLOW_SIGNAL_DURATION = float(f.read())

    logger.info("All specified ML models and constants loaded successfully.")

except FileNotFoundError as e:
    logger.error(f"Error loading model: {e}. Please ensure all model files are in the 'models' directory.")
    exit(1) # Exit if essential models are missing
except Exception as e:
    logger.error(f"An unexpected error occurred during model loading: {e}")
    exit(1)



# ------------------------------------------- Smart Route Planner  Feature  Using Network Graph --------------------------------------------

# 1. Loading Models and Road Network Graph
# --- Load ML Models and Data ---
traffic_prediction_model = None

# Define consistent paths (at top of your loading section)
base_dir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(base_dir, 'models')  # Standardized path
data_dir = os.path.join(base_dir, 'data')      # Standardized path

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# For model loading
try:
    traffic_model_path = os.path.join(models_dir, "traffic_model.pkl")
    road_network_path = os.path.join(data_dir, "Vadodara_Smart_Route_Suggestions.csv")
    
    logger.info(f"Loading traffic model from: {traffic_model_path}")
    logger.info(f"Loading road network from: {road_network_path}")
    
    if not os.path.exists(traffic_model_path):
        raise FileNotFoundError(f"Traffic model not found at {traffic_model_path}")
    if not os.path.exists(road_network_path):
        raise FileNotFoundError(f"Road network CSV not found at {road_network_path}")

    # Load models and data
    traffic_model = joblib.load(traffic_model_path)
    road_network_df = pd.read_csv(road_network_path)
    
    # Verify CSV has required columns
    required_columns = {'source', 'destination', 'distance_km', 'road_type'}
    if not required_columns.issubset(road_network_df.columns):
        missing = required_columns - set(road_network_df.columns)
        raise ValueError(f"Road network CSV missing columns: {missing}")

    # Create NetworkX graph
    road_network_graph = nx.DiGraph()
    for _, row in road_network_df.iterrows():
        road_network_graph.add_edge(
            row['source'], 
            row['destination'],
            distance_km=row['distance_km'],
            road_type=row['road_type']
        )
    
    logger.info(f"Road network graph loaded: {road_network_graph.number_of_nodes()} nodes, {road_network_graph.number_of_edges()} edges")

except Exception as e:
    logger.error(f"Error loading ML models or road network data: {e}")
    traffic_prediction_model = None
    traffic_model = None
    road_network_graph = None
    road_network_df = None  # Ensure this is set to None on error


# --- RoutePlanner Class  ---

class RoutePlanner:
    def __init__(self, graph_data, traffic_model_path, db):
        self.db = db
        self.graph = nx.DiGraph()
        
        # Build graph from DataFrame
        for _, row in graph_data.iterrows():
            self.graph.add_edge(
                row['source'],
                row['destination'],
                distance_km=row['distance_km'],
                road_type=row['road_type']
            )
        
        # Load traffic model
        try:
            self.traffic_model = joblib.load(traffic_model_path)
            logger.info(f"Traffic model loaded from {traffic_model_path}")
        except Exception as e:
            logger.error(f"Failed to load traffic model: {e}")
            self.traffic_model = None
            
            
            
        # ------------------------------------------------ Give the Bug  this code snippet ---------------------------------------------
        # Get WaterHazardReport model - NEW PROPER WAY
        try:
            self.WaterHazardReport = WaterHazardReport  # You don't need to import it — it's defined in the same file
            logger.debug("WaterHazardReport assigned directly.")
        except NameError:
            logger.warning("Direct assignment of WaterHazardReport failed. Trying fallback registry method.")
        try:
            # SQLAlchemy fallback approach
            self.WaterHazardReport = next(
            cls for cls in db.Model.registry._class_registry.values()
            if hasattr(cls, '__name__') and cls.__name__ == 'WaterHazardReport'
        )
            logger.debug("WaterHazardReport resolved from SQLAlchemy registry.")
        except StopIteration:
            logger.error("WaterHazardReport not found in SQLAlchemy registry.")
            self.WaterHazardReport = None
 # ------------------------------------------------ Give the Bug   Above        this code snippet ---------------------------------------------

        # Traffic parameters
        self.traffic_multipliers = {
            'Low': 1.0,
            'Medium': 1.5,
            'High': 2.5,
            'Very High': 4.0
        }
        self.road_type_base_speeds_kmph = {
            'Main road': 40,
            'Local road': 25,
            'Highway': 80,
            'Residential': 20
        }
        
    # predict traffic_density :
    def _predict_traffic_density(self, hour, is_night, is_peak_hour):
        """
        Predicts traffic density using the loaded model with the correct features
        and ensures the feature order matches the model's training.
        """
        if self.traffic_model is None:
            logger.warning("Traffic model not loaded. Returning 'Medium' traffic level.")
            return 'Medium' # Default to a medium traffic if model isn't available

        try:
            correct_feature_order = ['hour', 'is_peak_hour', 'is_night']

            # Determine is_weekend for current query time
            is_weekend = 1 if datetime.now().weekday() >= 5 else 0 # Monday=0, Sunday=6
            
            # Create a dictionary for the data to ensure correct mapping to columns
            data_dict = {
                'hour': hour,
                'is_peak_hour': is_peak_hour,
                'is_night': is_night,
                
            }
            
            # Create the DataFrame, explicitly ordering the columns
            features_df = pd.DataFrame([data_dict])[correct_feature_order]
            
            prediction = self.traffic_model.predict(features_df)[0]
            
            if prediction not in self.traffic_multipliers:
                logger.warning(f"Unknown traffic density prediction: {prediction}. Falling back to 'Low'.")
                return 'Low'
            return prediction
        except Exception as e:
            logger.error(f"Error predicting traffic density: {e}")
            return 'Low' # Fallback in case of prediction error
        
    #  calculate_travel_time 
    def calculate_travel_time(self, from_node, to_node, edge_data, hour, is_night, is_peak_hour):
        """
        Calculates estimated travel time for an edge considering predicted traffic.
        Passes the correct features to the traffic model.
        """
        distance_km = edge_data.get('distance_km', 0) # Use .get for safety
        road_type = edge_data.get('road_type', 'Local road') # Use .get for safety

        if distance_km == 0: # Avoid division by zero
            return 0

        try:
            # Call _predict_traffic_density with the features the model expects
            predicted_density = self._predict_traffic_density(
                hour=hour,
                is_night=is_night,
                is_peak_hour=is_peak_hour
            )
        except Exception as e:
            logger.warning(f"Failed to prepare features for traffic prediction: {e}. Using 'Low' density.")
            predicted_density = 'Low'

        traffic_multiplier = self.traffic_multipliers.get(predicted_density, 1.0)
        base_speed_kmph = self.road_type_base_speeds_kmph.get(road_type, 30)
        if base_speed_kmph == 0:
            base_speed_kmph = 30 # Prevent division by zero if road_type not found or speed is 0

        base_time_hours = distance_km / base_speed_kmph
        estimated_time_hours = base_time_hours * traffic_multiplier
        estimated_time_min = estimated_time_hours * 60

        return estimated_time_min


    # find_optimal_route :
    def find_optimal_route(self, start_node, end_node, hour):
        
        """
        Finds the optimal (lowest estimated time) route using Dijkstra's algorithm.
        Now includes derivation of 'is_night' and 'is_peak_hour' from 'hour'
        and checks for water logging alerts along the route.
        """
        if start_node not in self.graph or end_node not in self.graph:
            return {"error": "Start or end node not found in the road network."}

        try:
            is_night = 1 if (hour < 6 or hour > 20) else 0 # Assuming 8 PM (20) is start of night for traffic purposes
            is_peak_hour = 1 if ((hour >= 7 and hour < 10) or (hour >= 17 and hour < 20)) else 0 # Assuming 5 PM (17) to 8 PM (20)

            path_nodes = nx.dijkstra_path(
                self.graph,
                source=start_node,
                target=end_node,
                weight=lambda u, v, d: self.calculate_travel_time(u, v, d, hour, is_night, is_peak_hour)
                
            )
            print(f"DEBUG: Path nodes found: {path_nodes}") # See what your nodes are named

            total_distance_km = 0
            estimated_time_min = 0
            route_details = []
        
            # --- Check for Water Logging along the route ---
            water_logging_alerts_on_route = []
            try:
             
                # --- Debug Statements ---
                print(f"DEBUG: self.db is: {self.db}")
                print(f"DEBUG: self.WaterHazardReport is: {self.WaterHazardReport}")
                print(f"DEBUG: Type of self.WaterHazardReport: {type(self.WaterHazardReport)}")


                # Fetch active water logging reports from the database
                active_water_logging_reports = self.db.session.execute(
                    self.db.select(WaterHazardReport).filter(
                        WaterHazardReport.status.in_(['active']) # Check your actual statuses, 'active' was used previously
                    )
                ).scalars().all()
                
              
                logger.debug(f"Query for all active reports returned {len(active_water_logging_reports)} reports.")
                if active_water_logging_reports:
                    for report in active_water_logging_reports:
                        logger.error(f"Active Report Found: ID: {report.id}, Location: '{report.location_name}'")
                      
                processed_alert_locations = set()

                for node in path_nodes:
                    logger.error(f"Checking node '{node}' for alerts.")
                    
                    for report in active_water_logging_reports:
                        # Case-insensitive name match
                        if report.location_name and report.location_name.lower() == node.lower():
                            logger.error(f"DEBUG: MATCH FOUND: Node '{node}' matched report '{report.location_name}'")
                            
                            if report.location_name not in processed_alert_locations: # Avoid duplicates
                                alert_message = f"Water logging reported at {report.location_name}."
            
                                water_logging_alerts_on_route.append({
                                    "location_name": report.location_name,
                                    "image_url": f"/uploads/{report.image_filename}", 
                                    "message": alert_message,
                                    "status": report.status,
                                    "report_date": report.report_date.isoformat()
                                })
                                processed_alert_locations.add(report.location_name)
                            # Once a report is found for this node, move to the next node in the path
                            break 

            except Exception as e:
                logger.error(f"Error checking for water logging alerts: {e}")
                # Continue route calculation even if there's an error fetching alerts

            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i+1]
                edge_data = self.graph.get_edge_data(u, v)

                distance = edge_data.get('distance_km', 0)
                time = self.calculate_travel_time(u, v, edge_data, hour, is_night, is_peak_hour)

                total_distance_km += distance
                estimated_time_min += time

                route_details.append({
                    "from": u,
                    "to": v,
                    "distance_km": round(distance, 2),
                    "estimated_time_min": round(time, 2),
                    "road_type": edge_data.get('road_type', 'N/A')
                })

            return {
                "status": "success",
                "start_node": start_node,
                "end_node": end_node,
                "hour": hour,
                "is_night": is_night,
                "is_peak_hour": is_peak_hour,
                "path_nodes": path_nodes,
                "route_details": route_details,
                "total_distance_km": round(total_distance_km, 2),
                "estimated_time_min": round(estimated_time_min, 2),
                "water_logging_alerts": water_logging_alerts_on_route 
            }

        except nx.NetworkXNoPath:
            return {"error": "No path found between the specified nodes."}
        except Exception as e:
            logger.error(f"An error occurred during route calculation: {e}")
            return {"error": f"An error occurred during route calculation: {str(e)}"}
        
        
 # Initialize RoutePlanner after all models and data are loaded
route_planner_instance = None
if road_network_df is not None and os.path.exists(traffic_model_path):
    try:
        logger.info("Initializing RoutePlanner...")
        route_planner_instance = RoutePlanner(
            graph_data=road_network_df,
            traffic_model_path=traffic_model_path,
            db=db
        )
        logger.info("RoutePlanner initialized successfully")
    except Exception as e:
        logger.error(f"RoutePlanner initialization failed: {e}", exc_info=True)
else:
    logger.error("RoutePlanner not initialized - missing requirements")
    
#--------------------------------- Peak hour Travel Suggestion Feature ------------------------------------------------------------------------- 

# Paths to your model and data files - Ensure these paths are correct for your setup
model_path = os.path.join(app.root_path, 'data', 'traffic_prediction_model.pkl')
label_encoders_path = os.path.join(app.root_path, 'data', 'label_encoders.pkl')
traffic_hourly_data_path = os.path.join(app.root_path, 'data', 'Vadodara_Traffic_Hourly_Expanded.csv')

try:
    # Load the machine learning model
    traffic_prediction_model = joblib.load(model_path)
    logger.info(f"Traffic prediction model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading traffic prediction model from {model_path}: {e}")
    traffic_prediction_model = None # Ensure it's None on failure

try:
    # Load the label encoders
    label_encoders = joblib.load(label_encoders_path)
    logger.info(f"Label encoders loaded successfully from {label_encoders_path}")
    # Log classes for debugging purposes
    for key, encoder in label_encoders.items():
        if hasattr(encoder, 'classes_'):
            logger.info(f"Encoder '{key}' has classes: {encoder.classes_.tolist()}")
except Exception as e:
    logger.error(f"Error loading label encoders from {label_encoders_path}: {e}")
    label_encoders = {} # Ensure it's an empty dict on failure

try:
    # Load the historical traffic data (CSV)
    df = pd.read_csv(traffic_hourly_data_path)
    logger.info(f"Traffic data loaded successfully from {traffic_hourly_data_path}. Shape: {df.shape}")


 # ----------------------------- ENCODE CATEGORICAL COLUMNS IN DF ---------------------------------
    # These are the columns that should be numerically encoded for the model
    categorical_cols_to_encode = ['Location', 'Road_Type', 'Traffic_Level']

    for col in categorical_cols_to_encode:
        if col in df.columns and col in label_encoders:
            # Only transform if the column is currently in string/object format
            if df[col].dtype == 'object':
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                    logger.info(f"Successfully transformed column '{col}' in DataFrame to numerical.")
                except ValueError as ve:
                    logger.error(f"Error transforming column '{col}' using loaded encoder: {ve}. This might mean new categories in CSV or a mismatch between CSV and encoder.")
            elif df[col].dtype != 'int64':
                # If it's not an object type (e.g., float) but not int, attempt to transform
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                    logger.info(f"Successfully transformed non-integer column '{col}' to encoded integers.")
                except ValueError as ve:
                    logger.error(f"Error transforming non-int column '{col}': {ve}")
        elif col in df.columns:
            logger.warning(f"Label encoder for '{col}' not found. Cannot transform this column in DataFrame.")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame for transformation.")

    # Ensure 'Is_Weekend' and 'Hour' are integer types if they exist
    if 'Is_Weekend' in df.columns:
        df['Is_Weekend'] = df['Is_Weekend'].astype(int)
    if 'Hour' in df.columns:
        df['Hour'] = df['Hour'].astype(int)

    # Populate unique signal locations for dashboard/mapping.
    # We map encoded value to original label for client-side use (e.g., Leaflet markers)
    if 'Location' in label_encoders and hasattr(label_encoders['Location'], 'classes_'):
        # unique_signal_locations will contain the original string labels
        unique_signal_locations = label_encoders['Location'].classes_.tolist()
        # signal_location_mapping maps the integer-encoded value (as string) to the original label
        for i, label in enumerate(label_encoders['Location'].classes_):
            signal_location_mapping[str(i)] = label
    else:
        logger.warning("Location encoder not available for populating unique signal locations for map/dashboard.")

except Exception as e:
    logger.error(f"Error loading or processing traffic data from {traffic_hourly_data_path}: {e}")
    df = None # Ensure df is None on failure

# Final check if essential components are loaded for the app to function
if traffic_prediction_model is None or df is None or not label_encoders:
    logger.critical("Server startup failed: Essential components (model, data, encoders) are not loaded.")
else:
    logger.info("Server is ready to receive requests.")


# --- Load Locations from CSV ---
# Ensure 'Vadodara_Traffic_Hourly_Expanded' is the correct filename and it's in your 'data' directory
traffic_hourly_data_path = os.path.join(app.root_path,data_dir,"Vadodara_Traffic_Hourly_Expanded.csv")
 
 
if os.path.exists(traffic_hourly_data_path):   
    df = pd.read_csv(traffic_hourly_data_path)
    logger.info(f"DataFrame loaded Successfully from {traffic_hourly_data_path}. shape : {df.shape}")
    logger.info(f"DataFrame columns:{df.columns.tolist()}")
    
    #  Verify essential columns exist for peak hors feature 
    required_cols = ['Location' , 'Road_Type' , 'Traffic_Level']
    for col in required_cols :
        if col not in df.columns :
            logger.error(f"Missing required column in CSV :'{col}' from {traffic_hourly_data_path}")
            df = None # Invaluable df if critical column is missing 
            break 
    
    if df is not None :
        logger.info("Required columns for peak hours feature are  present." )
        
else:
    logger.error(f"Hourly traffic CSV file not found at: {traffic_hourly_data_path}")
    df = None


# Assuming Full_Vadodara_Traffic_Signal_Timings.csv is in a 'data' directory
SIGNAL_LOCATIONS = []
try:
    csv_path = os.path.join(data_dir, "Full_Vadodara_Traffic_Signal_Timings.csv")
    df_signals = pd.read_csv(csv_path)
    # Ensure 'location' column exists and is used
    if 'location' in df_signals.columns:
        SIGNAL_LOCATIONS = sorted(df_signals['location'].dropna().unique().tolist())
        logger.info(f"Loaded {len(SIGNAL_LOCATIONS)} unique signal locations from CSV.")
    else:
        logger.warning(f"'{csv_path}' does not contain a 'location' column. Signal locations will be empty.")
except FileNotFoundError:
    logger.error(f"Error: {csv_path} not found. Cannot load signal locations.")
except Exception as e:
    logger.error(f"An error occurred while loading signal locations from CSV: {e}")

    


# ----------------------------------------------------------- Helper FUNCTION STARTS --------------------------------------------------------------

# ----------------------------------------------------- Helper Functions  For Direction of Routes ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def geocode_nominatim(query):
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "SmartTrafficApp/1.0"}
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            return {
                "display_name": data[0]["display_name"],
                "lat": float(data[0]["lat"]),
                "lng": float(data[0]["lon"])
            }
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Nominatim geocoding failed: {e}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# TrueWay Directions API Configuration
TRUEWAY_API_HOST = "trueway-directions2.p.rapidapi.com"
TRUEWAY_API_KEY = os.getenv('TRUEWAY_API_KEY') # Replace with your actual key

def get_route_info(origin_coords, destination_coords):
    """
    Fetches detailed route information using the TrueWay Directions API.
    origin_coords and destination_coords are lists/tuples like [latitude, longitude].
    """
    logger.info(f"get_route_info received origin_coords: {origin_coords}, type: {type(origin_coords)}")
    logger.info(f"get_route_info received destination_coords: {destination_coords}, type: {type(destination_coords)}")

    url = f"https://{TRUEWAY_API_HOST}/FindDrivingRoute"

    headers = {
        'x-rapidapi-host': TRUEWAY_API_HOST,
        'x-rapidapi-key': TRUEWAY_API_KEY
    }

    try:
        if not isinstance(origin_coords, (list, tuple)) or len(origin_coords) < 2:
            raise ValueError(f"Invalid origin_coords format: {origin_coords}")
        if not isinstance(destination_coords, (list, tuple)) or len(destination_coords) < 2:
            raise ValueError(f"Invalid destination_coords format: {destination_coords}")

        origin_lat, origin_lon = float(origin_coords[0]), float(origin_coords[1])
        dest_lat, dest_lon = float(destination_coords[0]), float(destination_coords[1])

        stops_param = f"{origin_lat},{origin_lon};{dest_lat},{dest_lon}"

        params = {
            'stops': stops_param,
            
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

      
        if data and 'route' in data: # Check if 'route' key exists and it's not None
            route_data = data['route'] # Access 'route' directly as an object, not an array
            path = []
            for point in route_data.get('geometry', {}).get('coordinates', []):
                path.append(point) 
            

            distance_m = route_data.get('distance', 0)
            duration_s = route_data.get('duration', 0)

            logger.info(f"Final path array to be returned: {path}") 

            return {
                "path": path,
                "distance": distance_m / 1000, # Convert meters to kilometers
                "duration": duration_s / 60   # Convert seconds to minutes
            }
            
        return {"path": [], "distance": 0, "duration": 0, "message": "No route found from API or empty route data"}
    except requests.exceptions.RequestException as e:
        logger.error(f"TrueWay API route calculation failed: {e}")
        if response is not None:
             logger.error(f"TrueWay API raw error response: {response.text}")
        return {"path": [], "distance": 0, "duration": 0, "message": f"API error: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during route calculation: {e}, Type: {type(e)}")
        return {"path": [], "distance": 0, "duration": 0, "message": f"Internal error: {e}"}
    
    
        
    

# ------------------------------------------- :For The Pick Hour Travel Suggestion  Feature Helper Function : -----------------------------------------------------------------------------------

# Helper function for generating travel advice:
def generate_travel_advice(location_encoded, road_type_encoded, is_weekend):
    """Generate best/worst times based on historical data and provide hourly predictions."""
    
    if df is None: 
        logger.error("DataFrame (df) not loaded for generate_travel_advice. Returning default values.")
        return "N/A", "N/A", [] # Return empty list for predictions

    # Inverse transform encoded values to get human-readable names for filtering
    # This assumes label_encoders have been correctly loaded and contain 'Location' and 'Road_Type'
    try:
        loc_name_for_filter = label_encoders['Location'].inverse_transform([location_encoded])[0]
        road_name_for_filter = label_encoders['Road_Type'].inverse_transform([road_type_encoded])[0]
    except KeyError as e:
        logger.error(f"Label encoder for {e} not found. Cannot inverse transform for filtering.")
        return "N/A", "N/A", []
    except ValueError as e:
        logger.error(f"Value error during inverse_transform: {e}. Encoded value might be out of range for encoder.")
        return "N/A", "N/A", []

    logger.info(f"Filtering data for Location: '{loc_name_for_filter}' (encoded: {location_encoded}), Road Type: '{road_name_for_filter}' (encoded: {road_type_encoded}), Is_Weekend: {is_weekend}")

    # Filter the historical data based on selection (using inverse-transformed names)
    subset = df[
        (df['Location'] == loc_name_for_filter) & 
        (df['Road_Type'] == road_name_for_filter) &
        (df['Is_Weekend'] == is_weekend)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if len(subset) == 0:
        logger.warning(f"No historical data found for: Location='{loc_name_for_filter}', RoadType='{road_name_for_filter}', IsWeekend={is_weekend}. Cannot generate advice.")
        return "No specific advice available.", "No specific advice available.", []
    
    # Map categorical Traffic_Level to numerical for mean calculation
    # This mapping must align with how your Traffic_Level categories are ordered 
    # (e.g., 'Low' is best, 'Very High' is worst)
    # Ensure 'Traffic_Level' encoder exists before mapping
    if 'Traffic_Level' not in label_encoders:
        logger.error("Label encoder for 'Traffic_Level' not found. Cannot map traffic levels.")
        return "N/A", "N/A", []

    traffic_level_map = {level: i for i, level in enumerate(label_encoders['Traffic_Level'].classes_)}
    subset['Traffic_Level_Numerical'] = subset['Traffic_Level'].map(traffic_level_map)

    traffic_by_hour = subset.groupby('Hour')['Traffic_Level_Numerical'].mean()
    
    # Get hours with lowest average traffic (best times)
    best_hours_numerical = traffic_by_hour.nsmallest(4).index.tolist()
    # Get hours with highest average traffic (worst times)
    worst_hours_numerical = traffic_by_hour.nlargest(4).index.tolist()
    
    best_times_str = ", ".join(f"{h:02d}:00" for h in sorted(best_hours_numerical))
    worst_times_str = ", ".join(f"{h:02d}:00" for h in sorted(worst_hours_numerical))

    # Also generate hourly predictions for the selected parameters
    hourly_predictions = create_hourly_predictions(location_encoded, road_type_encoded, is_weekend)
    
    return best_times_str, worst_times_str, hourly_predictions



 #Helper function for hourly predictions :
def create_hourly_predictions(location_encoded, road_type_encoded, is_weekend):
    """Generate hourly predictions for a given location, road type, and day type."""
    predictions = []
    
    if traffic_prediction_model is None or not label_encoders: 
        logger.error("Traffic prediction model or label encoders not loaded for create_hourly_predictions. Returning empty predictions.")
        return []

    for hour in range(24):
        input_data = pd.DataFrame({
            'Location': [location_encoded],
            'Hour': [hour],
            'Is_Weekend': [is_weekend],
            'Road_Type': [road_type_encoded],
        })
        
        try:
            # Use the correct model variable name: traffic_prediction_model
            prediction_encoded = traffic_prediction_model.predict(input_data)[0] 
            traffic_level = label_encoders['Traffic_Level'].inverse_transform([prediction_encoded])[0]
            predictions.append({'hour': f"{hour:02d}:00", 'traffic_level': traffic_level})
        except Exception as e:
            logger.error(f"Error predicting for hour {hour}: {e}", exc_info=True)
            predictions.append({'hour': f"{hour:02d}:00", 'traffic_level': "Error"})
    
    return predictions




# ------------------------------------------------------- All THE END POINTS START FROM HERE -----------------------------------------------------------------------------------------------
# ---- ENd point For Saving FIle in Database ----------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Attempting to serve file from: {filepath}") 
    if not os.path.exists(filepath):
        print(f"File not found at the calculated path: {filepath}") 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ------------------------------------------------------ End Point of Smart Route  Planner ---------------------------------------------------------

@app.route('/suggest-smart-route', methods=['POST'])
def suggest_smart_route():
    if route_planner_instance is None:
        return jsonify({"status": "error", "message": "Route planner not initialized. Check model and graph loading."}), 500

    data = request.json
    start_node = data.get('start_node')
    end_node = data.get('end_node')
    hour = data.get('hour') # Hour of the day (e.g., 9 for 9 AM)

    if not all([start_node, end_node, hour is not None]):
        return jsonify({"status": "error", "message": "Missing start_node, end_node, or hour in request."}), 400

    try:
        hour = int(hour)
        if not (0 <= hour <= 23):
             return jsonify({"status": "error", "message": "Hour must be between 0 and 23."}), 400
    except ValueError:
         return jsonify({"status": "error", "message": "Hour must be an integer."}), 400


    logger.info(f"Received smart route request: start={start_node}, end={end_node}, hour={hour}")
    result = route_planner_instance.find_optimal_route(start_node, end_node, hour)

    if "error" in result:
        return jsonify({"status": "error", "message": result["error"]}), 404 if "No path found" in result["error"] or "not found" in result["error"] else 500
    else:
        return jsonify(result)
    
    
    
    
# -------------------------------------------- API Endpoints  For Pick Hour Travel Suggestion ------------------------------------------------------------

@app.route('/get-prediction-data', methods=['GET'])
def get_prediction_data():
    try:
        global df, traffic_prediction_model, label_encoders # Ensure these are accessible

        if df is None or traffic_prediction_model is None or not label_encoders:
            logger.error("DataFrame (df), prediction model, or label encoders not loaded. Cannot provide dropdown options.")
            return jsonify({'status': 'error', 'message': 'Server not ready: Historical data or models not available.'}), 500

        formatted_locations = []
        if 'Location' in label_encoders and hasattr(label_encoders['Location'], 'classes_'):
            # Iterate through the encoder's classes to get original labels and their corresponding integer values (0, 1, 2...)
            for i, label in enumerate(label_encoders['Location'].classes_):
                formatted_locations.append({'label': str(label), 'value': i})
            logger.info(f"Populated {len(formatted_locations)} locations for dropdown.")
        else:
            logger.warning("Label encoder for 'Location' not available or invalid. Cannot populate locations dropdown.")
            # Fallback for locations if encoder is missing/invalid - you might want to adjust this for production
            # Example: use unique values from df if df is loaded, assuming they are numerical
            if df is not None and 'Location' in df.columns:
                unique_numeric_locations = sorted(df['Location'].unique().tolist())
                for val in unique_numeric_locations:
                    formatted_locations.append({'label': f"Location {val}", 'value': val})


        formatted_road_types = []
        if 'Road_Type' in label_encoders and hasattr(label_encoders['Road_Type'], 'classes_'):
            # Iterate through the encoder's classes to get original labels and their corresponding integer values
            for i, label in enumerate(label_encoders['Road_Type'].classes_):
                formatted_road_types.append({'label': str(label), 'value': i})
            logger.info(f"Populated {len(formatted_road_types)} road types for dropdown.")
        else:
            logger.warning("Label encoder for 'Road_Type' not available or invalid. Cannot populate road types dropdown.")
            # Fallback for road types if encoder is missing/invalid
            if df is not None and 'Road_Type' in df.columns:
                unique_numeric_road_types = sorted(df['Road_Type'].unique().tolist())
                for val in unique_numeric_road_types:
                    formatted_road_types.append({'label': f"Road Type {val}", 'value': val})


        # Prepare hours for dropdown
        hours = [{"label": f"{h:02d}:00", "value": h} for h in range(24)]

        is_weekend_options = [
            {"label": "Weekday", "value": 0},
            {"label": "Weekend", "value": 1}
        ]

        return jsonify({
            'status': 'success',
            'locations': formatted_locations,
            'road_types': formatted_road_types,
            'hours': hours,
            'is_weekend_options': is_weekend_options
        })
    except Exception as e:
        logger.error(f"Error getting prediction data: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500







#------------------------------------------------------------ End Point For Pick Hour Travel Suggestion Report  ---------------------------------------------------------------------------------
# ---  get_peak_hours_report route to use the updated helper function ---
@app.route('/get-peak-hours-report', methods=['POST'])
def get_peak_hours_report():
    data = request.get_json()
    location_encoded = data.get('location') # This is the encoded integer
    road_type_encoded = data.get('road_type') # This is the encoded integer
    is_weekend = data.get('is_weekend')

    if location_encoded is None or road_type_encoded is None or is_weekend is None:
        logger.error("Missing data for peak hours report.")
        return jsonify({"status": "error", "message": "Missing location, road type, or day type."}), 400

    try:
        # Cast to int (they should already be integers from frontend)
        location_encoded = int(location_encoded)
        road_type_encoded = int(road_type_encoded)
        is_weekend = int(is_weekend) # 0 for weekday, 1 for weekend

        # Call the updated generate_travel_advice
        best_times, avoid_times, hourly_predictions = generate_travel_advice(location_encoded, road_type_encoded, is_weekend)

        if best_times == "N/A": # Check for the specific error return from generate_travel_advice
            logger.error("Failed to generate travel advice due to data issues or missing encoders.")
            return jsonify({"status": "error", "message": "Could not generate travel advice due to missing data or server configuration issues."}), 500

        # Inverse transform inputs back to human-readable names for the report response
        # Ensure 'Location' and 'Road_Type' encoders exist before inverse transforming
        location_name = label_encoders['Location'].inverse_transform([location_encoded])[0] if 'Location' in label_encoders else f"Unknown Location ({location_encoded})"
        road_type_name = label_encoders['Road_Type'].inverse_transform([road_type_encoded])[0] if 'Road_Type' in label_encoders else f"Unknown Road Type ({road_type_encoded})"
        day_type_name = "Weekend" if is_weekend == 1 else "Weekday"

        return jsonify({
            "status": "success",
            "location_name": location_name,
            "road_type_name": road_type_name,
            "day_type_name": day_type_name,
            "best_times": best_times,
            "avoid_times": avoid_times,
            "hourly_predictions": hourly_predictions
        })
    except ValueError as e:
        logger.error(f"Invalid input data type: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Invalid input data type. Expected integers for encoded values."}), 400
    except KeyError as e:
        logger.error(f"Label encoder missing or failed for: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Server configuration error: missing label encoder for {e}. Ensure label_encoders.pkl is correct."}), 500
    except Exception as e:
        logger.critical(f"An unexpected error occurred in /get-peak-hours-report: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500



#--------------------------------------------------------- End Point For Get the Route B/W Source - Destination -----------------------------------------------------------

@app.route('/predict-route', methods=['POST'])
def predict_route():
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')

    if not all([origin, destination]):
        return jsonify({"status": "error", "message": "Missing origin or destination"}), 400

    try:
        # Assuming origin and destination are in [lat, lng] format
        route_info = get_route_info(origin, destination)
        if route_info.get("path"):
            return jsonify({
                "status": "success",
                "path": route_info["path"],
                "distance": round(route_info["distance"], 2),
                "duration": round(route_info["duration"], 2)
            })
        else:
            return jsonify({"status": "error", "message": route_info.get("message", "Could not find a route")}), 500
    except Exception as e:
        logger.error(f"Error predicting route: {e}")
        return jsonify({"status": "error", "message": f"Route prediction failed: {e}"}), 500


#--------------------------------------------------------------- End Point For Finding the Route --------------------------------------------------

@app.route('/get-route', methods=['POST'])
def get_route():
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')

    if not all([origin, destination]):
        return jsonify({"status": "error", "message": "Missing origin or destination"}), 400

    try:
        route_info = get_route_info(origin, destination)
        if route_info.get("path"):
            return jsonify({
                "status": "success",
                "route": route_info["path"],
                "distance": round(route_info["distance"], 2),
                "duration": round(route_info["duration"], 2)
            })
        else:
            return jsonify({"status": "error", "message": route_info.get("message", "Could not find a route")}), 500
    except Exception as e:
        logger.error(f"Error getting route: {e}")
        return jsonify({"status": "error", "message": f"Route calculation failed: {e}"}), 500
    
    
    
# -------------------------------------------------------- End Route For Search Location (for Route Suggestion )-----------------------------------------------------------------------------------------

@app.route('/search-location', methods=['POST'])
def search_location():
    data = request.json
    query = data.get('query')
    as_dict = data.get('as_dict', False)

    if not query:
        return jsonify([]), 200

    vadodara_locations = {
        "sursagar": {
            "name": "Sursagar Lake",
            "lat": 22.2987,
            "lng": 73.1950,
            "address": "Sursagar Lake, Vadodara",
            "type": "location"
        },
        "vadodara": {
            "name": "Vadodara City Center",
            "lat": 22.3072,
            "lng": 73.1812,
            "address": "Vadodara, Gujarat",
            "type": "location"
        },
        "akota": {
            "name": "Akota",
            "lat": 22.2975,
            "lng": 73.1650,
            "address": "Akota, Vadodara",
            "type": "location"
        },
        "alampura": {
            "name": "Alampura",
            "lat": 22.3200,
            "lng": 73.1900,
            "address": "Alampura, Vadodara",
            "type": "location"
        }
    }

    query_lower = query.lower()
    matched_locations = []

    for key, location in vadodara_locations.items():
        if key in query_lower:
            if as_dict:
                matched_locations.append(location)
            else:
                matched_locations.append({
                    "display_name": location["address"],
                    "lat": str(location["lat"]),
                    "lon": str(location["lng"])
                })

    if matched_locations:
        return jsonify(matched_locations), 200

    nominatim_result = geocode_nominatim(query)
    if nominatim_result:
        if as_dict:
            return jsonify([{
                "name": nominatim_result["display_name"],
                "lat": nominatim_result["lat"],
                "lng": nominatim_result["lng"],
                "address": nominatim_result["display_name"],
                "type": "geocode"
            }]), 200
        else:
            return jsonify([{
                "display_name": nominatim_result["display_name"],
                "lat": str(nominatim_result["lat"]),
                "lon": str(nominatim_result["lng"])
            }]), 200

    return jsonify([]), 200

    

    
#-------------------------------------------------- End Point For Traffic Signal Timing ------------------------------------------------------------

@app.route('/predict-signal-timings', methods=['POST'])
def predict_signal_timings():
    data = request.json
    location = data.get('location') # Now expecting location
    hour = data.get('hour')
    day_type = data.get('day_type')

    if not all([location, hour is not None, day_type]):
        return jsonify({"status": "error", "message": "Missing input data (location, hour, or day_type)"}), 400

    try:
        # Encode inputs as per your predict_timing function
        day_type_encoded = 0 if day_type == 'Weekday' else 1

        if 5 <= hour < 12:
            time_of_day_encoded = 0  # Morning
        elif 12 <= hour < 17:
            time_of_day_encoded = 1  # Afternoon
        elif 17 <= hour < 22:
            time_of_day_encoded = 2  # Evening
        else:
            time_of_day_encoded = 3  # Night

        is_peak_hour = 1 if ((day_type == 'Weekday') and ((7 <= hour < 10) or (17 <= hour < 20))) else 0

        features = np.array([[
            hour,
            is_peak_hour,
            day_type_encoded,
            time_of_day_encoded
        ]])

        red_time = max(20, int(signal_red_model.predict(features)[0]))
        green_time = max(15, int(signal_green_model.predict(features)[0]))
        yellow_time = YELLOW_SIGNAL_DURATION # Your constant yellow time

        # You had a location_to_signal mapping in your snippet, you might want to add that here
        # For now, just use the location name
        signal_id = location # Placeholder if no specific ID mapping in backend

        return jsonify({
            "status": "success",
            "signal_timings": {
                "red": round(red_time),
                "green": round(green_time),
                "yellow": round(yellow_time)
            },
            "location": location,
            "signal_id": signal_id,
            "current_hour": hour,
            "day_type": day_type,
            "confidence": 0.85 # Placeholder, you can replace with actual model confidence if available
        })
    except Exception as e:
        logger.error(f"Error predicting signal timings: {e}")
        return jsonify({"status": "error", "message": f"Signal timing prediction failed: {e}"}), 500

# ------------------------------------------------------------ End Point For Get the Signal Location -------------------------------------------------------

@app.route('/get-signal-locations', methods=['GET'])
def get_signal_locations():
    return jsonify(SIGNAL_LOCATIONS)




# ----------------------------------------------------  End Point OF Predict All the Data -------------------------------------------------------------------------------

@app.route('/predict-all', methods=['POST'])
def predict_all():
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')
    hour = data.get('hour')
    is_peak = data.get('is_peak')
    weather = data.get('weather')
    traffic_volume = data.get('traffic_volume') # Still included, but not for signal model
    is_night = data.get('is_night')
    day_type = data.get('day_get')
    road_type = data.get('road_type')
    location = data.get('location') # For signal timing prediction

    results = {}
    errors = []

    # Traffic Prediction
    try:
        if all([hour is not None, is_peak is not None, is_night is not None, road_type]):
            if 'road_type' in label_encoders:
                road_type_encoded = label_encoders['road_type'].transform([road_type])[0]
            else:
                road_type_encoded = 0
            features = np.array([[hour, is_peak, is_night, road_type_encoded]])
            prediction_proba = traffic_prediction_model.predict_proba(features)[0]
            prediction_class = np.argmax(prediction_proba)
            traffic_levels = ['Light', 'Moderate', 'Heavy']
            predicted_traffic_level = traffic_levels[prediction_class]
            results['traffic_prediction'] = prediction_class.item()
            results['traffic_level'] = predicted_traffic_level
        else:
            errors.append("Missing data for traffic prediction.")
    except Exception as e:
        errors.append(f"Traffic prediction failed: {e}")
        logger.error(f"Error in /predict-all (traffic): {e}")


    # Route Prediction
    try:
        if all([origin, destination]):
            route_info = get_route_info(origin, destination)
            if route_info.get("path"):
                results['optimal_route'] = {
                    "path": route_info["path"],
                    "distance": round(route_info["distance"], 2),
                    "duration": round(route_info["duration"], 2)
                }
            else:
                errors.append(f"Route prediction failed: {route_info.get('message', 'No route found')}")
        else:
            errors.append("Missing origin or destination for route prediction.")
    except Exception as e:
        errors.append(f"Route prediction failed: {e}")
        logger.error(f"Error in /predict-all (route): {e}")


    # Signal Timings Prediction
    try:
        if all([location, hour is not None, day_type]):
            day_type_encoded = 0 if day_type == 'Weekday' else 1
            if 5 <= hour < 12:
                time_of_day_encoded = 0
            elif 12 <= hour < 17:
                time_of_day_encoded = 1
            elif 17 <= hour < 22:
                time_of_day_encoded = 2
            else:
                time_of_day_encoded = 3
            is_peak_hour = 1 if ((day_type == 'Weekday') and ((7 <= hour < 10) or (17 <= hour < 20))) else 0
            features = np.array([[
                hour,
                is_peak_hour,
                day_type_encoded,
                time_of_day_encoded
            ]])

            red_time = signal_red_model.predict(features)[0]
            green_time = signal_green_model.predict(features)[0]
            yellow_time = YELLOW_SIGNAL_DURATION
            results['signal_timings'] = {
                "red": round(red_time),
                "green": round(green_time),
                "yellow": round(yellow_time)
            }
        else:
            errors.append("Missing data for signal timings prediction.")
    except Exception as e:
        errors.append(f"Signal timings prediction failed: {e}")
        logger.error(f"Error in /predict-all (signal): {e}")

    if errors:
        return jsonify({"status": "error", "messages": errors, "results": results}), 500
    return jsonify({"status": "success", "results": results})







#------------------------------------------------------------- End Point For Emergency Route ----------------------------------------------------
@app.route('/emergency-routes', methods=['POST'])
def emergency_routes():
    data = request.json
    current_location = data.get('current_location')

    if not current_location:
        return jsonify({"status": "error", "message": "Current location not provided"}), 400

    emergency_facilities = [
        {"name": "Sterling Hospital", "lat": 22.3115, "lng": 73.1558, "type": "hospital"},
        {"name": "Apollo Hospitals", "lat": 22.3160, "lng": 73.2005, "type": "hospital"},
        {"name": "SSG Hospital", "lat": 22.3023, "lng": 73.1906, "type": "hospital"},
        {"name": "Sayaji Hospital", "lat": 22.3069, "lng": 73.1788, "type": "hospital"},
    ]

    nearest_facility = None
    min_distance = float('inf')

    # Find the nearest facility using haversine distance
    for facility in emergency_facilities:
        dist = haversine_distance(current_location[0], current_location[1], facility["lat"], facility["lng"])
        if dist < min_distance:
            min_distance = dist
            nearest_facility = facility

    if nearest_facility:
        # Use TrueWay API for the emergency route
        route_info = get_route_info(current_location, [nearest_facility["lat"], nearest_facility["lng"]])

        if route_info.get("path"):
            return jsonify({
                "status": "success",
                "destination": nearest_facility,
                "route": route_info["path"],
                "distance": round(route_info["distance"], 2),
                "duration": round(route_info["duration"], 2)
            })
        else:
            return jsonify({"status": "error", "message": route_info.get("message", "Could not find an emergency route")}), 500
    else:
        return jsonify({"status": "error", "message": "No emergency facilities found"}), 404
    
# --------------------------------------------------- End Point For Fetch the Get Alert Data ---------------------------------------------------------------------
@app.route('/get-alerts', methods=['GET'])
def get_alerts():
    alerts = [
        {
            "id": "alert_001",
            "type": "accident",
            "location": "Near Fatehgunj Circle",
            "lat": 22.3200,
            "lng": 73.1800,
            "severity": "high",
            "timestamp": "2025-05-27T10:00:00Z",
            "description": "Major accident reported, expect significant delays."
        },
        {
            "id": "alert_002",
            "type": "roadwork",
            "location": "Alkapuri Road",
            "lat": 22.3100,
            "lng": 73.1700,
            "severity": "medium",
            "timestamp": "2025-05-27T09:30:00Z",
            "description": "Road maintenance work in progress, minor diversions."
        }
    ]
    return jsonify(alerts)

# ------------------------------------ End Point For Voice Assistance -------------------------------------------------------------

@app.route('/assistant-command', methods=['POST'])
def assistant_command():
    data = request.json
    command = data.get('command', '')

    if not command:
        return jsonify({"error": "Command not provided"}), 400

    try:
        response = gemini_model.generate_content(
            f"Analyze the following command and extract intent and location. "
            f"If it's a navigation command, identify origin (if specified, otherwise infer 'current location') and destination. "
            f"If it's a traffic query, identify the location. "
            f"Return a JSON object with 'intent', 'location', and a 'message' summarizing the command. "
            f"Example for navigation: {{'intent': 'navigate', 'destination': 'Sursagar Lake', 'message': 'Navigating to Sursagar Lake'}}. "
            f"Example for traffic: {{'intent': 'traffic_query', 'location': 'Akota', 'message': 'Checking traffic near Akota'}}. "
            f"Command: '{command}'"
        )
        gemini_output = response.text.strip().replace("```json", "").replace("```", "")
        parsed_response = json.loads(gemini_output)
        return jsonify(parsed_response)
    except Exception as e:
        logger.error(f"Error processing assistant command with Gemini: {e}")
        return jsonify({"error": "Failed to process command", "message": str(e)}), 500

# ---------------------------------------------------- Gemini Setup ---------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("Using fallback Gemini API key - recommend setting GEMINI_API_KEY in .env")
   
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")



#--------------------------------------------------------- Live News Rss End Point -----------------------------------------------------------------------------------
# News Articlae fatch from the Rss Link :
@app.route('/api/vadodara-news', methods=['GET'] ) # Add this line
def get_vadodara_news():
    feed_url = "https://sandesh.com/rss/vadodara-city.xml"  # Replace with the actual URL
    feed = feedparser.parse(feed_url)
    news_items = []
    
    # Function to strip HTML tags 
    def strip_html_tags(text):
        if text:
            clean = re.compile('<.*?>') # regx to match any html tag 
            return re.sub(clean, '', text).strip() # replace with empty string and strip whitespace 
        return ''
    
    for entry in feed.entries:
        cleaned_summary = strip_html_tags(entry.summary if hasattr(entry, 'summary')else '')
        cleaned_title = strip_html_tags(entry.title if hasattr(entry, 'title') else '')
        
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "summary":  cleaned_summary,
            "published":  entry.published if hasattr(entry, 'published') else ''
        })
    return news_items



#---------------------------------------------------------- API End Points-------------------------------------------------------------------------------------------------

# ---  render the frontend Templates ---
@app.route('/')
def index():
    return render_template('main.html')


if __name__ == '__main__':
    
    # Create database tables if they don't exist :
    with app.app_context():
        
        db.create_all()
        logger.info("Database tables created or already exist .")
        logger.debug("DEBUG: Logger is working.")  # Should show
        sys.stdout.flush()  # Ensure print shows up
    app.run(debug=True, host='0.0.0.0', port=5000)
    
