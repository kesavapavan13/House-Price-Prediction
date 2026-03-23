import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
/* ── Blue gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 40%, #2e6da4 75%, #1a91c8 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}

/* ── Content area ── */
.block-container {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(6px) !important;
}

/* ── Hide sidebar ── */
section[data-testid="stSidebar"] { display: none !important; }

/* ── Global text ── */
h1,h2,h3,h4,h5,h6      { color: #e8f4fd !important; }
p, label, span, div, li { color: #cce0f5 !important; }
.stCaption, small       { color: #90b8d8 !important; }

/* ── Input boxes — white bg, black text ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] > div > div > div,
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #c0d4ea !important;
    border-radius: 8px !important;
    color: #111111 !important;
}
div[data-testid="stSelectbox"] *,
div[data-baseweb="select"] *,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #111111 !important;
    background-color: transparent !important;
}
div[data-testid="stSelectbox"] div[class*="placeholder"],
div[data-testid="stSelectbox"] div[class*="singleValue"],
div[data-testid="stSelectbox"] div[class*="ValueContainer"] {
    color: #111111 !important;
}
div[data-testid="stNumberInput"] input { color: #111111 !important; }
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #1e88e5 !important;
    box-shadow: 0 0 0 3px rgba(30,136,229,0.2) !important;
}

/* ── Dropdown popup — white bg, black text ── */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
li[role="option"],
li[role="option"] * {
    background: #ffffff !important;
    color: #111111 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: #e3f2fd !important;
    color: #0d47a1 !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] label p { color: #cce0f5 !important; }

/* ── Select slider ── */
div[data-testid="stSlider"] * { color: #cce0f5 !important; }

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0 0%, #1e88e5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 18px rgba(30,136,229,0.45) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%) !important;
    box-shadow: 0 6px 24px rgba(66,165,245,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
}
div[data-testid="stMetricLabel"] p { color: #90b8d8 !important; font-size: 12px !important; }
div[data-testid="stMetricValue"]   { color: #e8f4fd !important; }
div[data-testid="stMetricDelta"]   { color: #81d4fa !important; }

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background: rgba(30,100,200,0.18) !important;
    border-radius: 10px !important;
    border-left: 4px solid #4fc3f7 !important;
    color: #cce0f5 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
LOCATIONS = sorted([
    'thane','navi-mumbai','nagpur','mumbai','ahmedabad','bangalore','chennai',
    'gurgaon','hyderabad','indore','jaipur','kolkata','lucknow','new-delhi',
    'noida','pune','agra','ahmadnagar','allahabad','aurangabad','badlapur',
    'belgaum','bhiwadi','bhiwandi','bhopal','bhubaneswar','chandigarh',
    'coimbatore','dehradun','durgapur','ernakulam','faridabad','ghaziabad',
    'goa','greater-noida','guntur','guwahati','gwalior','haridwar','jabalpur',
    'jamshedpur','jodhpur','kalyan','kanpur','kochi','kozhikode','ludhiana',
    'madurai','mangalore','mohali','mysore','nashik','navsari','nellore',
    'palakkad','palghar','panchkula','patna','pondicherry','raipur',
    'rajahmundry','ranchi','satara','shimla','siliguri','solapur','sonipat',
    'surat','thrissur','tirupati','trichy','trivandrum','udaipur','udupi',
    'vadodara','vapi','varanasi','vijayawada','visakhapatnam','vrindavan','zirakpur'
])
LOC_DISPLAY  = [l.replace('-', ' ').title() for l in LOCATIONS]
TRANSACTIONS = ['Resale', 'New Property', 'Other']
STATUSES     = ['Ready', 'Not Specified']
FURNISHINGS  = ['Unfurnished', 'Semi-Furnished', 'Furnished', 'Not Specified']
FACINGS      = ['Not Specified', 'East', 'West', 'North', 'North - East', 'North - West', 'South', 'South -West', 'South - East']
OWNERSHIPS   = ['Not Specified', 'Freehold', 'Co-operative Society', 'Power Of Attorney', 'Leasehold']

# ── Session State ─────────────────────────────────────────────────────────
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "price" not in st.session_state:
    st.session_state.price = None
if "inputs" not in st.session_state:
    st.session_state.inputs = None

p = st.session_state.inputs or {}

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
st.title("🏠 House Price Predictor")
st.caption("Fill in the property details and click **Predict Price** to get an instant estimate.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOCATION & DEAL TYPE
# ══════════════════════════════════════════════════════════════════════════
st.subheader("🏙️ Location & Deal Type")

c1, c2, c3 = st.columns(3)

saved    = p.get("location", "").replace('-', ' ').title()
location = c1.selectbox(
    "City / Area",
    ["— Select a city —"] + LOC_DISPLAY,
    index=LOC_DISPLAY.index(saved) + 1 if saved in LOC_DISPLAY else 0
)
transaction = c2.radio(
    "Transaction Type", TRANSACTIONS,
    index=TRANSACTIONS.index(p["transaction"]) if p.get("transaction") in TRANSACTIONS else 0,
    horizontal=True
)
status = c3.radio(
    "Status", STATUSES,
    index=STATUSES.index(p["status"]) if p.get("status") in STATUSES else 0,
    horizontal=True
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — UNIT SPECIFICATIONS
# ══════════════════════════════════════════════════════════════════════════
st.subheader("🏠 Unit Specifications")

c1, c2, c3, c4 = st.columns(4)

bhk         = c1.number_input("BHK",               min_value=1,     max_value=20,     step=1,   value=int(p.get("bhk", 2)))
bathroom    = c2.number_input("Bathrooms",          min_value=1,     max_value=20,     step=1,   value=int(p.get("bathroom", 2)))
balcony     = c3.number_input("Balconies",          min_value=0,     max_value=10,     step=1,   value=int(p.get("balcony", 1)))
carpet_area = c4.number_input("Carpet Area (sqft)", min_value=100.0, max_value=50000.0, step=50.0, value=float(p.get("carpet_area", 1000.0)))

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — PROPERTY ATTRIBUTES
# ══════════════════════════════════════════════════════════════════════════
st.subheader("🏷️ Property Attributes")

c1, c2, c3 = st.columns(3)

furnishing = c1.selectbox(
    "Furnishing", FURNISHINGS,
    index=FURNISHINGS.index(p["furnishing"]) if p.get("furnishing") in FURNISHINGS else 0
)
ownership = c2.selectbox(
    "Ownership", OWNERSHIPS,
    index=OWNERSHIPS.index(p["ownership"]) if p.get("ownership") in OWNERSHIPS else 0
)
facing = c3.selectbox(
    "Facing", FACINGS,
    index=FACINGS.index(p["facing"]) if p.get("facing") in FACINGS else 0
)

c4, c5 = st.columns(2)
total_floors = c4.number_input("Total Floors",              min_value=1,  max_value=200, step=1, value=int(p.get("total_floors", 10)))
floor_number = c5.number_input("Floor Number (0 = Ground)", min_value=-2, max_value=200, step=1, value=int(p.get("floor_number", 3)))

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    if location == "— Select a city —":
        st.error("Please select a city.")
    elif floor_number > total_floors:
        st.error("Floor number cannot exceed total floors.")
    else:
        st.session_state.inputs = {
            "location":     LOCATIONS[LOC_DISPLAY.index(location)],
            "transaction":  transaction,
            "status":       status,
            "bhk":          bhk,
            "bathroom":     bathroom,
            "balcony":      balcony,
            "carpet_area":  carpet_area,
            "furnishing":   furnishing,
            "ownership":    ownership,
            "facing":       facing,
            "total_floors": total_floors,
            "floor_number": floor_number,
        }

        try:
            preprocessor = joblib.load("preprocessor.pkl")
            model        = joblib.load("xgb_model.pkl")

            # ── Feature Engineering (must match training) ──────────────────
            floor_number_adj = max(floor_number, 0)
            is_basement      = 1 if floor_number < 0 else 0
            floor_ratio      = floor_number_adj / total_floors if total_floors > 0 else 0
            parking_count    = 1
            area_log         = np.log1p(carpet_area)

            input_df = pd.DataFrame([{
                "location":          LOCATIONS[LOC_DISPLAY.index(location)],
                "Transaction":       transaction,
                "Status":            status,
                "Furnishing":        furnishing,
                "facing":            facing,
                "Bathroom":          bathroom,
                "Balcony":           balcony,
                "Ownership":         ownership,
                "BHK":               bhk,
                "area":              area_log,
                "parking_count":     parking_count,
                "is_basement":       is_basement,
                "floor_number_adj":  floor_number_adj,
                "floor_ratio":       floor_ratio,
                "total_floors":      total_floors,
            }])

            pred_log = model.predict(preprocessor.transform(input_df))[0]
            price    = float(np.expm1(pred_log))

            st.session_state.price     = price
            st.session_state.predicted = True

        except FileNotFoundError:
            st.error("⚠️ Place `preprocessor.pkl` and `xgb_model.pkl` in the same folder as `app.py`.")
            st.session_state.predicted = False
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.session_state.predicted = False

# ══════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.predicted and st.session_state.price is not None:
    inp = st.session_state.inputs

    st.divider()
    st.subheader("📋 Selected Features")

    c = st.columns(6)
    c[0].metric("📍 Location",    inp["location"].replace('-', ' ').title())
    c[1].metric("🔄 Transaction", inp["transaction"])
    c[2].metric("🏗️ Status",     inp["status"])
    c[3].metric("🛏️ BHK",        inp["bhk"])
    c[4].metric("🛁 Bathrooms",   inp["bathroom"])
    c[5].metric("🏗️ Balconies",  inp["balcony"])

    c = st.columns(5)
    c[0].metric("📐 Carpet Area", f"{inp['carpet_area']:,.0f} sqft")
    c[1].metric("🛋️ Furnishing", inp["furnishing"])
    c[2].metric("📜 Ownership",   inp["ownership"])
    c[3].metric("🧭 Facing",      inp["facing"])
    c[4].metric("🏢 Floor",       f"{inp['floor_number']} of {inp['total_floors']}")

    st.divider()
    st.subheader("💰 Price Estimate")
    st.success("✅ Prediction Successful")
    st.metric(
        label="Estimated Market Price",
        value=f"₹ {st.session_state.price:,.2f} Lakhs",
        delta=f"{inp['location'].replace('-', ' ').title()} · {inp['bhk']} BHK · {inp['carpet_area']:,.0f} sqft",
        delta_color="off"
    )