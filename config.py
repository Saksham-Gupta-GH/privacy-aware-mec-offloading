import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# ==========================================
# Cloud Tier (Azure VM) Configuration
# ==========================================
AZURE_HOST = os.getenv("AZURE_HOST", "20.219.130.205")
AZURE_USER = os.getenv("AZURE_USER", "azureuser")
AZURE_API_PORT = int(os.getenv("AZURE_API_PORT", "8000"))
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "prism_sec_healthcare_mec_2026")
AZURE_URL = f"http://{AZURE_HOST}:{AZURE_API_PORT}"

# ==========================================
# Edge Tier (MEC Server) Configuration (Simulated)
# ==========================================
EDGE_HOST = os.getenv("EDGE_HOST", "127.0.0.1")
EDGE_API_PORT = int(os.getenv("EDGE_API_PORT", "8001"))
EDGE_URL = f"http://{EDGE_HOST}:{EDGE_API_PORT}"

# ==========================================
# Privacy Levels
# ==========================================
PRIVACY_LEVEL_PUBLIC = 0       # Level 0: Device telemetry, battery, ward ambient
PRIVACY_LEVEL_RESTRICTED = 1   # Level 1: Anonymized vitals, equipment logs
PRIVACY_LEVEL_CRITICAL = 2     # Level 2: Raw ECG, MRI segmentation, patient PII

PRIVACY_LABELS = {
    PRIVACY_LEVEL_PUBLIC: "Public / Telemetry",
    PRIVACY_LEVEL_RESTRICTED: "Restricted / Operational",
    PRIVACY_LEVEL_CRITICAL: "Critical / Confidential PII"
}

# ==========================================
# Default Objective Weights (Sum to 1.0)
# ==========================================
DEFAULT_WEIGHTS = {
    "w_L": float(os.getenv("WEIGHT_LATENCY", "0.35")),   # Execution Latency
    "w_E": float(os.getenv("WEIGHT_ENERGY", "0.30")),    # Mobile Device Energy
    "w_C": float(os.getenv("WEIGHT_COST", "0.10")),      # Financial Cost
    "w_P": float(os.getenv("WEIGHT_PRIVACY", "0.25")),   # Privacy Risk
}

# ==========================================
# Hardware & Network System Parameters
# ==========================================
# Local Mobile Device
LOCAL_CPU_FREQ_GHZ = 1.8            # Mobile CPU clock (GHz)
LOCAL_POWER_ACTIVE_W = 1.2          # Energy drain when computing (Watts)
LOCAL_POWER_TX_W = 0.8              # Energy drain when transmitting (Watts)
LOCAL_BATTERY_CAPACITY_J = 15000.0  # Mobile battery reserve (Joules)

# Edge Server (Simulated near-premises MEC gateway, e.g. 5G/WiFi-6 small cell)
EDGE_CPU_FREQ_GHZ = 3.6             # Edge CPU clock (GHz)
EDGE_RTT_MS = 12.0                  # Low RTT (ms)
EDGE_UPLINK_MBPS = 100.0            # High local wireless bandwidth (Mbps)
EDGE_DOWNLINK_MBPS = 200.0          # Mbps
EDGE_FINANCIAL_COST_PER_MFLOP = 0.0 # On-premise infrastructure (free/amortized)
EDGE_PRIVACY_RISK_FACTOR = 0.15     # Trusted local hospital gateway

# Cloud Tier (Azure B1s / Standard VM over WAN)
# Default values before calibration:
CLOUD_CPU_FREQ_GHZ = 4.2            # Azure Cloud CPU capability
CLOUD_DEFAULT_RTT_MS = 65.0         # Typical WAN latency (calibrated dynamically)
CLOUD_DEFAULT_UPLINK_MBPS = 35.0    # Internet uplink bandwidth
CLOUD_DEFAULT_DOWNLINK_MBPS = 75.0  # Internet downlink bandwidth
CLOUD_FINANCIAL_COST_PER_MFLOP = 0.000008  # Micro-charge for compute
CLOUD_PRIVACY_RISK_FACTOR = 0.95    # Untrusted third-party public cloud

# Security: Tiers where Critical (L2) data is allowed
# Critical PII can ONLY execute on Local or Trusted Edge; strictly forbidden on Cloud
ALLOWED_TIERS_FOR_CRITICAL = {"local", "edge"}
