"""Locked project constants for the AMR federated KG capstone.

Every module and every federated client imports from here, so the graph schema
stays identical across the federation (required for weight aggregation to work).
"""
import os
from pathlib import Path


def _resolve_data_dir() -> Path:
    """Find the ARMD CSVs no matter whose machine this runs on.

    Priority:
      1. ARMD_DIR environment variable (each teammate sets their own).
      2. Standard Colab Drive mount, if present.
      3. Local data/raw/ fallback.
    """
    env = os.environ.get("ARMD_DIR")
    if env:
        return Path(env)
    colab_mount = Path("/content/drive/MyDrive/ARMD")
    if colab_mount.exists():
        return colab_mount
    return Path("data/raw")


# --- Paths ---
DATA_DIR = _resolve_data_dir()  # set ARMD_DIR to your shared-Drive ARMD folder

# --- ARMD tables we actually use (10 of 16) ---
ARMD_TABLES = {
    "cohort":        "microbiology_cultures_cohort.csv",                      # organism, antibiotic, susceptibility (target)
    "demographics":  "microbiology_cultures_demographics.csv",               # patient: age bin, gender
    "adi":           "microbiology_cultures_adi_scores.csv",                  # patient: socioeconomic
    "comorbidity":   "microbiology_cultures_comorbidity.csv",                # comorbidity nodes
    "procedures":    "microbiology_cultures_priorprocedures.csv",            # procedure nodes
    "ward":          "microbiology_cultures_ward_info.csv",                  # partitioning + edge features
    "labs":          "microbiology_cultures_labs.csv",                       # edge features
    "vitals":        "microbiology_cultures_vitals.csv",                     # edge features
    "abx_class_exp": "microbiology_cultures_antibiotic_class_exposure.csv",  # patient->antibiotic prior exposure
    "resistance":    "microbiology_cultures_microbial_resistance.csv",       # organism->antibiotic history prior
}

# --- Join keys (link every table back to a culture order) ---
KEYS = {
    "patient":   "anon_id",
    "encounter": "pat_enc_csn_id_coded",
    "culture":   "order_proc_id_coded",
    "time":      "order_time_jittered_utc",
}

# --- Heterogeneous graph schema: 5 node types ---
NODE_TYPES = ["patient", "organism", "antibiotic", "comorbidity", "procedure"]

# (src, relation, dst)
EDGE_TYPES = [
    ("organism", "tested",          "antibiotic"),  # <-- PREDICTION TARGET (S/I/R)
    ("patient",  "grew",            "organism"),     # ward/labs/vitals/time = edge features
    ("patient",  "has",             "comorbidity"),
    ("patient",  "underwent",       "procedure"),
    ("patient",  "prior_exposure",  "antibiotic"),
    ("organism", "known_resistant", "antibiotic"),   # historical prior, NOT a label
]

TARGET_EDGE = ("organism", "tested", "antibiotic")

# --- Label handling (decided from EDA on the real cohort) ---
# Raw `susceptibility` has 6 values. Keep only true outcomes; drop the rest.
LABEL_COLUMN = "susceptibility"
VALID_LABELS = ["Susceptible", "Resistant", "Intermediate"]
DROP_LABELS = ["Null", "Inconclusive", "Synergism"]  # Null = negative culture (was_positive=0)
# Class balance is ~80% S / 16% R / 3% I -> use the BINARY target + class weights.
BINARY_TARGET = True
RESISTANT_LABELS = ["Resistant", "Intermediate"]      # I folded into R for the binary task
# binary classes: 0 = not_R (Susceptible), 1 = R (Resistant or Intermediate)

# --- Federated partitioning: simulate hospitals by ward (ARMD is single-site) ---
# Ward flags are separate binary columns and can overlap; assign one ward per culture
# by this priority (sickest setting wins). ICU is small (~51k) but kept — it's the
# clinically distinct, high-resistance client the topology-aware method should help most.
WARD_FLAG_COLUMNS = {
    "ICU": "hosp_ward_ICU",
    "ER":  "hosp_ward_ER",
    "IP":  "hosp_ward_IP",
    "OP":  "hosp_ward_OP",
}
WARD_PRIORITY = ["ICU", "ER", "IP", "OP"]  # used to pick a single ward when flags overlap
WARD_PARTITIONS = ["ICU", "ER", "IP", "OP"]

# --- Verified column names (differ from the README in places) ---
COLUMNS = {
    "organism":            "organism",
    "antibiotic":          "antibiotic",
    "was_positive":        "was_positive",
    "culture_description": "culture_description",
    "ordering_mode":       "ordering_mode",          # extra usable cohort feature
    "procedure":           "procedure_description",   # README said procedure_name
    "comorbidity":         "comorbidity_component",
    "age":                 "age",
    "gender":              "gender",
    "adi_score":           "adi_score",
}

# --- Reproducibility ---
SEED = 42