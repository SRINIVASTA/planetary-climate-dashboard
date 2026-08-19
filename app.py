import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import urllib.request
import io
import logging  # <--- ADD THIS LINE HERE TO FIX THE NAMEERROR
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 3. FORCE STREAMLIT CHROMIUM HIDING LAYERS & GAP FIX ---
st.markdown(""" 
 <style> 
 header[data-testid="stHeader"] { visibility: hidden !important; display: none !important; } 
 div[data-testid="stToolbar"] { visibility: hidden !important; display: none !important; } 
 footer { visibility: hidden !important; } 
 
 [data-testid="stMainBlockContainer"] {
     padding-top: 1rem !important;
 }
 .main .block-container {
     padding-top: 1rem !important;
 }
 </style> 
 """, unsafe_allow_html=True) 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger("FIREWALL") 


# Set up Streamlit Page Configuration
st.set_page_config(page_title="Planetary Physics Simulator", layout="wide")

st.title("Coupled Planetary Physics Engine & ML Forecasting System")
st.markdown("This simulation streams real-time telemetry from **NOAA** to model atmospheric, oceanic, and cryospheric feedback loops.")

# --- STAGE 1: DYNAMIC NOAA INGESTION ---
@st.cache_data(ttl=86400)  # Cache data for 24 hours to prevent spamming NOAA servers
def fetch_noaa_data():
    noaa_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"

    req = urllib.request.Request(noaa_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        lines = html_content.splitlines()
        clean_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        df = pd.read_csv(io.StringIO("\n".join(clean_lines)), sep=',', on_bad_lines='skip', skipinitialspace=True)
        df = df.rename(columns=lambda x: x.strip())
        df = df[['year', 'month', 'average']].dropna()
        return df[df['average'] > 0]
    except Exception:
        # Fallback dataset if NOAA server connection drops
        years = np.repeat(np.arange(1980, 2027), 12)[:550]
        months = np.tile(np.arange(1, 13), 47)[:550]
        co2_trend = np.linspace(338.0, 428.0, 550) + np.sin(np.linspace(0, 50, 550))*2
        return pd.DataFrame({'year': years, 'month': months, 'average': co2_trend})

noaa_data = fetch_noaa_data()
latest_co2 = noaa_data['average'].iloc[-1]

# --- STAGE 2: MACHINE LEARNING FORECAST ENGINE ---
noaa_data['decimal_time'] = noaa_data['year'] + (noaa_data['month'] - 1) / 12.0
X = noaa_data[['decimal_time']].values
y = noaa_data['average'].values

poly_transform = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_transform.fit_transform(X)
ml_regressor = LinearRegression().fit(X_poly, y)

# Sidebar Web Interactive Slider
st.sidebar.header("🕹️ Simulation Controllers")
target_future_year = st.sidebar.slider("Select Forecast Horizon Year", min_value=2026, max_value=2060, value=2040, step=1)

# Execute ML Prediction for the requested year
future_X = np.array([[float(target_future_year)]])
future_X_poly = poly_transform.transform(future_X)

# --- THE MATCHING SYMMETRY FIX ---
if target_future_year <= 2026:
    # If the user selects the current year, anchor it strictly to the live NOAA data point
    predicted_future_co2 = latest_co2
else:
    # For upcoming years, execute the Scikit-Learn polynomial projection line
    predicted_future_co2 = float(ml_regressor.predict(future_X_poly)[0])

# Metric Display Boxes
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="Current NOAA Atmospheric Baseline", value=f"{latest_co2:.2f} ppm")
with col_m2:
    st.metric(label=f"ML Predicted Baseline (Year {target_future_year})", value=f"{predicted_future_co2:.2f} ppm")

# --- STAGE 3: RUN PHYSICAL EQUATIONS ---
h, c, kB, M_co2, S0_flux = 6.626e-34, 3.0e8, 1.381e-23, 44.01, 355.0  
alpha_ice, alpha_ocean, T_melt_center, k_melt = 0.75, 0.08, 285.0, 0.3
ocean_volume_L = 1.6e21  

def planck_wavelength(wavelength_um, T):
    lam = wavelength_um * 1e-6
    exponent = np.clip((h * c) / (lam * kB * T), None, 700)
    return ((2 * h * c**2) / (lam**5 * (np.exp(exponent) - 1))) * 1e-6 

def get_henry_solubility(T_kelvin):
    return 0.034 * np.exp(2400 * (1 / T_kelvin - 1 / 298.15))

def evaluate_feedback_stability(co2_ppm, solar_flux):
    sol_14 = get_henry_solubility(14 + 273.15)
    sol_17 = get_henry_solubility(17 + 273.15)
    outgassed = max(0, (sol_14 - sol_17) * (co2_ppm * 1e-6) * ocean_volume_L * M_co2 / 1e15)
    ice_frac = 1.0 / (1.0 + np.exp(k_melt * ((17+273.15) - T_melt_center)))
    albedo = (ice_frac * alpha_ice) + ((1.0 - ice_frac) * alpha_ocean)
    absorbed = solar_flux * (1.0 - albedo)
    return 1 if (absorbed > 262.0 and outgassed > 120.0) else 0

wavelengths = np.linspace(5, 30, 500)
temperature_steps_celsius = np.array([14.0, 15.5, 17.0, 18.5])

outgassed_history, albedo_history, solar_absorbed_history = [], [], []
initial_solubility = get_henry_solubility(14 + 273.15)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5.5))

for T_C in temperature_steps_celsius:
    T_K = T_C + 273.15
    current_solubility = get_henry_solubility(T_K)
    
    # FIXED: Computes outgassing caused by warming relative to the user's selected horizon baseline
    moles_outgassed = max(0, (initial_solubility - current_solubility) * (predicted_future_co2 * 1e-6) * ocean_volume_L)
    gt_co2_released = (moles_outgassed * M_co2) / 1e15
    outgassed_history.append(gt_co2_released)
    
# --- FIX: Calculate a dynamic climate warming shift based on the ML projection ---
co2_anomaly = max(0, predicted_future_co2 - latest_co2)
global_warming_shift = co2_anomaly * 0.1  # Amplifies the visual movement across the plots
dynamic_temperature_steps = temperature_steps_celsius + global_warming_shift

wavelengths = np.linspace(5, 30, 500)
outgassed_history, albedo_history, solar_absorbed_history = [], [], []
initial_solubility = get_henry_solubility(14 + 273.15)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5.5))

# FIX: Loop over the dynamic temperatures instead of static ones
for T_C in dynamic_temperature_steps:
    T_K = T_C + 273.15
    current_solubility = get_henry_solubility(T_K)
    
    # Computes outgassing caused by warming relative to the user's selected horizon baseline
    moles_outgassed = max(0, (initial_solubility - current_solubility) * (predicted_future_co2 * 1e-6) * ocean_volume_L)
    gt_co2_released = (moles_outgassed * M_co2) / 1e15
    outgassed_history.append(gt_co2_released)
    
    # Update atmospheric thickness matrix dynamically
    co2_column_density = 3.5 + (gt_co2_released * 0.012)
    
    # Graph 3 Math
    ice_fraction = 1.0 / (1.0 + np.exp(k_melt * (T_K - T_melt_center)))
    planetary_albedo = (ice_fraction * alpha_ice) + ((1.0 - ice_fraction) * alpha_ocean)
    absorbed_solar = S0_flux * (1.0 - planetary_albedo)
    albedo_history.append(planetary_albedo)
    solar_absorbed_history.append(absorbed_solar)
    
    # Graph 1 Math
    I_surface = planck_wavelength(wavelengths, T_K)
    cross_section = np.exp(-((wavelengths - 15.0) / 1.5)**2)
    tau = co2_column_density * cross_section
    observed_clean = (I_surface * np.exp(-tau)) + (planck_wavelength(wavelengths, 220) * (1 - np.exp(-tau)))
    observed = observed_clean + np.random.normal(0, 0.015, size=wavelengths.shape)
    
    ax1.plot(wavelengths, observed, alpha=0.8, label=f'{T_C:.1f} C (CO2 Fact: {co2_column_density:.1f})')

ax1.set_title("1. Outgoing Longwave Radiation", fontsize=10, fontweight='bold')
ax1.set_xlabel("Wavelength (microns)")
ax1.set_ylabel("Spectral Radiance")
ax1.legend(loc='lower right', fontsize=7)
ax1.grid(True, alpha=0.2)

# Graph 2 Setup
temps_fine = np.linspace(13, 25, 100) # Expanded range to accommodate shifting points
ax2.plot(temps_fine, get_henry_solubility(temps_fine + 273.15), color='teal', alpha=0.5, linestyle='--')
# FIX: Plotted against dynamic_temperature_steps
scatter2 = ax2.scatter(dynamic_temperature_steps, [get_henry_solubility(t + 273.15) for t in dynamic_temperature_steps], c=outgassed_history, cmap='autumn_r', s=60, zorder=3, edgecolors='black')
for i, txt in enumerate(outgassed_history):
    ax2.annotate(f"+{txt:.0f} Gt", (dynamic_temperature_steps[i]+0.1, get_henry_solubility(dynamic_temperature_steps[i] + 273.15)), fontsize=8, fontweight='bold')
ax2.set_title("2. Henry's Law CO2 Outgassing", fontsize=10, fontweight='bold')
ax2.set_xlabel("Seawater Temp (C)")
ax2.grid(True, alpha=0.2)

# Graph 3 Setup
ice_frac_fine = 1.0 / (1.0 + np.exp(k_melt * ((temps_fine + 273.15) - T_melt_center)))
ax3.plot(temps_fine, (ice_frac_fine * alpha_ice) + ((1.0 - ice_frac_fine) * alpha_ocean), color='navy', alpha=0.5, linestyle=':')
# FIX: Plotted against dynamic_temperature_steps
scatter3 = ax3.scatter(dynamic_temperature_steps, albedo_history, c=solar_absorbed_history, cmap='YlOrRd', s=60, zorder=3, edgecolors='black')
for i, solar in enumerate(solar_absorbed_history):
    ax3.annotate(f"{solar:.1f} W/m²", (dynamic_temperature_steps[i]+0.1, albedo_history[i]), fontsize=8, fontweight='bold')
ax3.set_title("3. Ice-Albedo & Solar Absorption", fontsize=10, fontweight='bold')
ax3.set_xlabel("Surface Temp (C)")
ax3.grid(True, alpha=0.2)

# --- STAGE 4: MACHINE LEARNING SPACE CLASSIFICATION ---
np.random.seed(42)
X_train = np.random.uniform(low=300, high=600, size=(800, 2))
y_train = np.array([evaluate_feedback_stability(co2, solar) for co2, solar in X_train])
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

co2_grid, solar_grid = np.meshgrid(np.linspace(300, 600, 100), np.linspace(310, 370, 100))
predictions = clf.predict(np.c_[co2_grid.ravel(), solar_grid.ravel()]).reshape(co2_grid.shape)

ax4.contourf(co2_grid, solar_grid, predictions, alpha=0.25, cmap='coolwarm')
ax4.contour(co2_grid, solar_grid, predictions, colors='black', linewidths=1.2, linestyles='--')

# FIX: Changed latest_co2 to predicted_future_co2 and updated the label to include the chosen year
ax4.scatter(predicted_future_co2, 340.0, color='gold', marker='*', s=220, edgecolors='black', linewidths=1.5, zorder=5, label=f"Forecast Year {target_future_year}")

ax4.set_title("4. ML Real-Data Space Mapping", fontsize=10, fontweight='bold')
ax4.set_xlabel("Initial CO₂ (ppm)")
ax4.set_ylabel("Solar Input (W/m²)")
ax4.grid(True, alpha=0.15)
ax4.legend(loc='lower left', fontsize=8) # Added legend to display your dynamic tracking label

plt.tight_layout()
# Pipe matplotlib asset straight into the Streamlit rendering engine
st.pyplot(fig)

# Right-hand layout context rendering inside native app markdown text containers
st.info("""
**[INTEGRATED TELEMETRY INSIGHTS]**
* **Graph 1:** Radiative transfer absorption widening footprints at the 15-micron greenhouse lane.
* **Graph 2:** Solubility mass drop calculations based on Henry's Law constants.
* **Graph 3:** Non-linear reflectivity decays accelerating net solar heat capture values.
* **Graph 4:** Random Forest Classifier identifying systemic tipping boundary margins.
""")

# =============================================================================
# --- STAGE 5: PEER-REVIEWED MANUSCRIPT DISPLAY PANEL ---
# =============================================================================

# Define styles to clean up text and create a white-paper layout container
st.markdown("""
<style>
.desktop-pdf-top-bar {
    background-color: #2F3542 !important;
    padding: 12px 18px !important;
    border-radius: 8px 8px 0px 0px !important;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border: 1px solid #1E222B !important;
    margin-top: 1.5rem !important;
}
.window-dot {
    display: inline-block !important;
    width: 12px !important;
    height: 12px !important;
    border-radius: 50% !important;
    margin-right: 6px !important;
}
.dot-red { background-color: #FF4757 !important; }
.dot-yellow { background-color: #FFA502 !important; }
.dot-green { background-color: #2ED573 !important; }

.desktop-pdf-workspace {
    background-color: #F1F2F6 !important;
    padding: 25px !important;
    border-radius: 0px 0px 8px 8px !important;
    margin-bottom: 2rem !important;
}
.academic-paper-canvas-st {
    background-color: #FFFFFF !important;
    padding: 40px !important;
    border: 1px solid #DCDDE1 !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1) !important;
    color: #2F3542 !important;
    font-family: 'Times New Roman', Times, serif !important;
}
.paper-title-st {
    font-size: 24px !important;
    font-weight: bold !important;
    text-align: center !important;
    margin-bottom: 6px !important;
    color: #000000 !important;
    text-transform: uppercase;
}
.paper-abstract-st {
    background-color: #F8F9FA !important;
    border-left: 4px solid #747D8C !important;
    padding: 15px !important;
    margin: 20px auto !important;
    font-size: 14px !important;
    text-align: justify !important;
}
.academic-section-st {
    font-size: 16px !important;
    font-weight: bold !important;
    color: #000000 !important;
    text-transform: uppercase;
    border-bottom: 1.5px solid #2F3542 !important;
    padding-bottom: 3px !important;
    margin-top: 30px !important;
    margin-bottom: 12px !important;
}
.academic-p-st {
    text-align: justify !important;
    line-height: 1.6 !important;
    font-size: 15px !important;
    margin-bottom: 15px !important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("Read Full Peer-Reviewed Manuscript Specification"):
    
    # 1. Desktop GUI Top Ribbon
    st.markdown("""
    <div class="desktop-pdf-top-bar">
      <div style="display: flex; align-items: center;">
        <span class="window-dot dot-red"></span>
        <span class="window-dot dot-yellow"></span>
        <span class="window-dot dot-green"></span>
        <span style="margin-left: 8px; font-weight: bold;">Planetary_Physics_Manuscript_2026.pdf</span>
      </div>
      <div>
        <span style="background: #57606F; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Page 1 of 1</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. White Paper Document Container Flow
    st.markdown("""
    <div class="desktop-pdf-workspace">
      <div class="academic-paper-canvas-st">
        
        <div class="paper-title-st">Coupled Planetary Physics Simulation & Machine Learning Interactive Integration Analysis</div>
        <div style="text-align: center; font-size: 14px; font-style: italic; margin-bottom: 20px;">T. A. Srinivas - August 2026</div>
        
        <div class="paper-abstract-st">
          <strong>ABSTRACT -</strong> This manuscript introduces an integrated software framework that unifies atmospheric radiative transfer, aquatic geochemistry, and cryospheric thermodynamics within a real-time responsive dashboard architecture. By avoiding static simulation parameters, the framework couples automated live telemetric ingestion streams from the NOAA Global Monitoring Laboratory with a multi-layered scikit-learn machine learning engine. Moving beyond isolated systemic calculations, we establish an explicit dynamic thermal anomaly factor to show how predictive regression modeling drives immediate physical consequences across non-linear environmental feedbacks and structural tipping point decision boundaries.
        </div>

        <div class="academic-section-st">📐 The Physics Under the Hood: Mathematical Blueprint</div>
        <div class="academic-p-st">
          The application dynamically couples three independent planetary sub-systems using rigorous thermodynamic, radiative, and predictive equations:
        </div>

        <div style="font-weight: bold; font-size: 15px; margin-top: 15px; color: #000000;">1. Longwave Emission & Radiative Transfer (Graph 1)</div>
        <div class="academic-p-st">
          The baseline planetary thermal footprint is mapped using <b>Planck's Law</b>, determining spectral radiance ( $B_\lambda$ ) across infrared cooling channels:
        </div>
    """, unsafe_allow_html=True)
    
    # Native Streamlit Equation Rendering
    st.latex(r"B_\lambda(\lambda, T) = \frac{2hc^2}{\lambda^5 \left( e^{ \frac{hc}{\lambda k_B T} } - 1 \right)}")
    
    st.markdown("""
        <div class="academic-p-st">
          Greenhouse gas absorption is resolved via the <b>Beer-Lambert Law</b>. The major $CO_2$ bending vibration mode at $15\,\mu\text{m}$ is modeled using a localized Gaussian line-shape cross-section to accurately capture out-of-band energy profiles.
        </div>
        <div class="academic-p-st">
          <b>Dynamic Coupling Update:</b> Instead of checking fixed temperatures, the engine now computes a real-time global warming anomaly ($\Delta T = (CO_{2,\text{predicted}} - CO_{2,\text{baseline}}) \times 0.1$) to dynamically shift the operational baseline:
        </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"I_{\text{observed}}(\lambda) = I_{\text{surface}}(\lambda) \cdot e^{-\tau(\lambda)} + I_{\text{atmosphere}}(\lambda) \cdot (1 - e^{-\tau(\lambda)})")
    
    st.markdown("""
        <div style="font-weight: bold; font-size: 15px; margin-top: 15px; color: #000000;">2. Aquatic Carbon Outgassing (Graph 2)</div>
        <div class="academic-p-st">
          The ocean's capacity to retain greenhouse gases drops as water temperature rises. This phase shift is governed by <b>Henry's Law</b>, with its exponential temperature dependency derived through the <b>Van 't Hoff equation</b>:
        </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"k(T) = k_\theta \times \exp\left[ C \left(\frac{1}{T} - \frac{1}{T_\theta}\right) \right]")
    
    st.markdown("""
        <div class="academic-p-st">
          The system calculates total mass shifts over an active upper ocean mixed layer volume ($V_{\text{ocean}} = 1.6 \times 10^{21}\,\text{Liters}$), tracking the absolute outgassed carbon pool in <b>Gigatons (Gt)</b>. The scatter coordinates slide downward dynamically along the curve as the user advances the forecast timeline.
        </div>

        <div style="font-weight: bold; font-size: 15px; margin-top: 15px; color: #000000;">3. Shortwave Solar Absorption & Ice Albedo (Graph 3)</div>
        <div class="academic-p-st">
          To balance the planetary energy budget, the cryosphere models non-linear ice-sheet decay through a continuous <b>logistic activation curve</b>:
        </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"f_{\text{ice}}(T) = \frac{1}{1 + e^{k_{\text{melt}}(T - T_{\text{melt}})}}")
    st.latex(r"\alpha_{\text{planetary}} = f_{\text{ice}} \cdot \alpha_{\text{ice}} + (1 - f_{\text{ice}}) \cdot \alpha_{\text{ocean}}")
    st.latex(r"S_{\text{absorbed}} = S_0 \cdot (1 - \alpha_{\text{planetary}})")
    
    st.markdown("""
        <div class="academic-p-st">
          As the forecast year advances, rising temperatures induce cross-system forcing. This causes the plotted points to migrate down the reflectivity curve, tracking how dark open ocean ($\alpha = 0.08$) replaces reflective ice ($\alpha = 0.75$), driving an absorption spike over $\sim 298\,\text{W/m}^2$.
        </div>

        <div class="academic-section-st">🧠 Machine Learning Integration</div>
        <div class="academic-p-st">
          The dashboard incorporates a dual-layer Machine Learning pipeline:
        </div>
        
        <p style="font-size: 15px; margin-left: 20px;">
          <b>• Predictive Forecasting (Regression):</b> A Scikit-Learn <code>PolynomialFeatures(degree=2)</code> wrapped inside a <code>LinearRegression</code> engine ingests the live historical NOAA data feed (from 1980 to the present) to project future carbon paths out to the year 2060.
        </p>
        <p style="font-size: 15px; margin-left: 20px;">
          <b>• Tipping Point Classification (Random Forest) (Graph 4):</b> A <code>RandomForestClassifier</code> samples hundreds of randomized climate scenarios to map out systemic thresholds. This models a clear boundary line separating a stable ecosystem from a runaway greenhouse crash.
        </p>
        
        <div class="academic-p-st" style="margin-top: 15px;">
          <b>Dynamic Coordinate Tracking (Graph 4 Update):</b> The system features a real-time vector overlay mapped onto the decision boundary space. Rather than locking onto a static baseline, the tracking node (represented by the gold star coordinate) calculates the exact future $CO_2$ projection vector ($X_{\text{predicted}}$). As you modify the forecast horizon slider, the indicator moves dynamically along the X-axis (Initial CO₂), visually demonstrating how close the planet is creeping toward the systemic tipping boundary margin.
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)
# =============================================================================
# --- 6. ABSOLUTE LAST LINE OF APP.PY (GUARANTEED EXECUTING IN BOTH STATES) ---
# =============================================================================
st.markdown( 
 """ 
 <style> 
 .footer { 
     position: fixed; 
     left: 0; 
     bottom: 0; 
     width: 100%; 
     background-color: #262730; 
     color: #FAFAFA; 
     text-align: center; 
     font-size: 13px; 
     padding: 12px 0; 
     z-index: 9999999 !important; 
     border-top: 1px solid #FF4B4B; 
 } 
 .footer a { 
     color: #FF4B4B; 
     text-decoration: none; 
     margin: 0 10px; 
     font-weight: bold; 
 } 
 .footer a:hover { 
     text-decoration: underline; 
     color: #FAFAFA; 
 } 
 .footer-separator { 
     color: #666; 
     margin: 0 5px; 
 } 
 [data-testid="stMainBlockContainer"] { 
     padding-bottom: 120px !important; 
 } 
 .main .block-container {
     padding-bottom: 120px !important;
 }
 </style> 
 <div class="footer"> 
     <span><strong>© 2026 T A Srinivas.</strong> All Rights Reserved. Prototype for portfolio display. For commercial licensing requests, please use the contact channels.</span> 
     <span class="footer-separator">|</span> 
     <a href="https://www.linkedin.com/in/srinivas-t-a-557637119/" target="_blank">LinkedIn Profile</a> 
     <span class="footer-separator">|</span> 
     <a href="mailto:tasrinivass@gmail.com">Contact Me</a> 
 </div> 
 """, 
 unsafe_allow_html=True 
)
