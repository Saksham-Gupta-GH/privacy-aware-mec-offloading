import random
import csv
from pathlib import Path
from typing import List, Dict
import config

# Set random seed for reproducibility
random.seed(42)

# Templates for Level 0: Public / Device Telemetry
L0_TEMPLATES = [
    "Ambient room temperature sensor check in Ward {ward}: current reading {temp}C.",
    "Battery status diagnostic for wearable node {dev_id}: battery level {val}%.",
    "Hospital corridor air humidity monitor: relative humidity measured at {val}%.",
    "BLE beacon RSSI signal strength scan from location anchor {dev_id}.",
    "Sanitization station liquid dispenser telemetry: remaining level {val}ml.",
    "Smart bed tilt angle and mechanical motor position check for unit {dev_id}.",
    "Firmware version verification and uptime log for telemetry gateway {dev_id}.",
    "Oxygen tank mechanical pressure regulator telemetry: {val} PSI remaining.",
    "Corridor lighting and UV sterilization status beacon: status nominal.",
    "WiFi access point RSSI and channel interference log from ward ceiling node.",
    "Backup generator voltage frequency diagnostic: frequency {val} Hz stable.",
    "HVAC sterile ward air filter differential pressure gauge log: {val} Pa.",
    "Smart IV pole wheel lock brake sensor telemetry: engaged status true.",
    "Medication cabinet ambient door contact magnetic sensor status ping."
]

# Templates for Level 1: Restricted / Operational / De-identified
L1_TEMPLATES = [
    "Aggregated ward {ward} mean vital sign index across {val} de-identified beds.",
    "Nurse shift scheduling and automated duty allocation model for ward {ward}.",
    "Medication cart RFID batch scanner reconciliation: {val} unit doses logged.",
    "Anonymized cohort blood pressure statistical distribution summary (Cohort-{dev_id}).",
    "Dialysis machine operational diagnostic telemetry and pump rotation speed log.",
    "Infusion pump flow rate calibration telemetry for anonymous unit {dev_id}.",
    "De-identified clinical trial placebo group physiological variance report.",
    "Surgical theater positive pressure differential airflow telemetry cycle.",
    "Hospital pharmacy automated drug inventory stock alert for batch #{dev_id}.",
    "Anonymized heart rate variability (HRV) aggregated summary over 4-hour window.",
    "Biomedical equipment preventive maintenance vibration analysis for scanner #{dev_id}.",
    "Cleanroom HEPA filtration particulate count sensor stream: particles {val} ppm.",
    "Sterilization autoclave temperature-pressure cycle log for tray set {dev_id}."
]

# Templates for Level 2: Critical / Confidential PII
L2_TEMPLATES = [
    "Real-time 12-lead ECG waveform arrhythmia detection for patient MRN-{dev_id}.",
    "Raw DICOM brain MRI T1-weighted slice 3D segmentation for subject #{dev_id}.",
    "Continuous Holter monitor raw cardiac telemetry and ST-elevation classification.",
    "Genomic sequence variant analysis and targeted oncology mutation mapping.",
    "Patient facial landmark analysis and symmetry detection for acute stroke screening.",
    "High-resolution chest X-ray radiograph pneumonia detection with patient metadata.",
    "Pediatric ICU arterial blood pressure invasive catheter real-time waveform.",
    "Psychiatric intake consultation speech transcript analysis for patient ID #{dev_id}.",
    "Fetal heart rate Doppler ultrasound audio analysis and distress alarm trigger.",
    "Electronic Health Record (EHR) clinical progression notes for patient #{dev_id}.",
    "Continuous blood glucose monitor telemetry linked to patient insulin pump #{dev_id}.",
    "Emergency department trauma triage vital signs and biometric facial scan.",
    "Post-operative telemetry feed: raw SpO2 plethysmograph and end-tidal CO2 trace."
]

# Ambiguous / Edge cases to challenge the classifier
AMBIGUOUS_CASES = [
    ("Research study de-identified pulse ox data with residual hospital admission code.", 1, 2),
    ("Anonymized patient vital signs with un-redacted room number and time of admission.", 1, 2),
    ("Device serial number log containing embedded patient medical record ID tag.", 0, 2),
    ("Ward air quality log with temporary note mentioning patient quarantined status.", 0, 1),
    ("Calibration ECG signal on test phantom device with synthetic cardiac rhythm.", 0, 1),
    ("De-identified genomic research benchmark file from open public repository.", 0, 1),
    ("Aggregate ward statistical summary that inadvertently includes single-patient outlier spike.", 1, 2)
]


def generate_task_dataset(num_samples: int = 1200, output_path: Path = None) -> List[Dict]:
    """
    Generates a realistic healthcare task dataset with controlled noise and varied computational characteristics.
    """
    if output_path is None:
        output_path = config.BASE_DIR / "data" / "healthcare_tasks.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    
    samples_per_class = num_samples // 3

    for task_id in range(num_samples):
        # Determine intended true class
        if task_id < samples_per_class:
            true_level = 0
            template = random.choice(L0_TEMPLATES)
            # Public tasks: small payload, light compute, relaxed deadlines
            payload_kb = round(random.uniform(2.0, 50.0), 1)
            output_kb = round(random.uniform(0.5, 5.0), 1)
            cpu_mflop = round(random.uniform(5.0, 100.0), 1)
            deadline_ms = round(random.uniform(400.0, 2000.0), 1)
        elif task_id < 2 * samples_per_class:
            true_level = 1
            template = random.choice(L1_TEMPLATES)
            # Restricted tasks: moderate payload, medium compute, moderate deadlines
            payload_kb = round(random.uniform(20.0, 500.0), 1)
            output_kb = round(random.uniform(5.0, 50.0), 1)
            cpu_mflop = round(random.uniform(50.0, 400.0), 1)
            deadline_ms = round(random.uniform(200.0, 1000.0), 1)
        else:
            true_level = 2
            template = random.choice(L2_TEMPLATES)
            # Critical tasks: high payload (waveforms/images), heavy compute, strict deadlines
            payload_kb = round(random.uniform(100.0, 4000.0), 1)
            output_kb = round(random.uniform(10.0, 150.0), 1)
            cpu_mflop = round(random.uniform(200.0, 1500.0), 1)
            deadline_ms = round(random.uniform(100.0, 600.0), 1)

        # Populate template parameters
        ward = random.choice(["ICU-A", "CCU-3", "ER-South", "Ward-4B", "Neuro-2", "Peds-1"])
        dev_id = random.randint(1000, 9999)
        val = random.randint(15, 95)
        temp = round(random.uniform(19.5, 25.0), 1)

        desc = template.format(ward=ward, dev_id=dev_id, val=val, temp=temp)

        # Controlled label noise (~6% realistic ambiguity)
        noisy_level = true_level
        if random.random() < 0.06:
            # Shift by +/- 1 level to simulate real annotator disagreement / ambiguous boundaries
            noise_delta = random.choice([-1, 1])
            noisy_level = max(0, min(2, true_level + noise_delta))

        samples.append({
            "task_id": task_id,
            "description": desc,
            "payload_size_kb": payload_kb,
            "output_size_kb": output_kb,
            "cpu_cycles_mflop": cpu_mflop,
            "deadline_ms": deadline_ms,
            "true_privacy_level": true_level,
            "labeled_privacy_level": noisy_level  # Used for training to reflect real noisy data
        })

    # Shuffle the dataset
    random.shuffle(samples)

    # Save to CSV
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "description", "payload_size_kb", "output_size_kb",
            "cpu_cycles_mflop", "deadline_ms", "true_privacy_level", "labeled_privacy_level"
        ])
        writer.writeheader()
        writer.writerows(samples)

    print(f"Generated {len(samples)} realistic healthcare tasks saved to: {output_path}")
    return samples


if __name__ == "__main__":
    generate_task_dataset()

