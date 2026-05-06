## All the imports

#Gaia Imports
import pandas as pd
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord

#ML Imports
import umap
import hdbscan
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


########################################################
# Gaia Code - Data import

'''
## Key note: 
# I choose to take the data in the G band which is a filter.
# 	Gaia measures in three bands/filters and G band is the white light which covers most of the stars
#	and has less missing values. As well as Gaia database provides the extinction in the G band.  

## Some convention to know: 
# mas = milli arcsecond (angular measurment)
# G band - G filter on the telescope, measures white light 330–1050 nm
# parallax - distance measurement (mas)
# parsec - unit of length used in astronomy. 3.26 light years ~ 1 parsec.
# proper motion - apparent angular motion of the star across the sky over time.
# apparent magnitude (m) - Brightness as it appears to us. Depends on the band.
# Absolute magnitude (M) - Brightness of the object if it was 10 parsecs away from Earth.
#						   Normalized magnitude. Depends on the band.
# Extinction - how much light gets absorbed between us and the object. Depends on the band.
# radial velocity - how much the star is moving towards or away from us.
'''

"""
Fetch Gaia DR3 data for NGC 2516 open cluster.
Retrieves photometric + astrometric features.
"""

# Cluster parameters
	# NGC 2516 centre (J2000) and search cone
CLUSTER_NAME   = "NGC 2516"
RA_DEG         = 119.517     # degrees
DEC_DEG        = -60.752     # degrees
SEARCH_RADIUS  = 1.0         # degrees (Huge area, more below)

# Loose proper-motion (pm) & parallax priors from textbook
# (pmra ~ -4.7, pmdec ~ 11.2 mas/yr; parallax ~ 2.41 mas -> ~415 pc)
PM_RA_CENTER   = -4.7        # mas/yr
PM_DEC_CENTER  =  11.2       # mas/yr
PM_SIGMA       =  3.0        # mas/yr — half-width of membership box
PARALLAX_MIN   =  1.5        # mas
PARALLAX_MAX   =  3.5        # mas

OUTPUT_FILE    = "gaia_ngc2516_raw.csv"

#Astronomical Data Query Language (ADQL) query
	# Columns chosen to support:
	#   Absolute magnitude calculation        (phot_g_mean_mag, parallax, ag_gspphot)
		# Need the distance through parallax, the apparent brightness and extinction 
		# to calculate the absolute magnetide through this equation (m-M = 5*log_10(d/5) + 5 - A)

	#   Color index / effective temperature  (bp_rp, teff_gspphot)
		# bp_rp - color index of the stars, teff_gspphot - effective surface temperature in Kelvin

	#   HR diagram axes                       (MG derived later, bp_rp)
		# Temperature/color on x-axis and magnitude on the y-axis

	#   Membership filtering                  (pmra, pmdec, parallax)
		# This is the process of figuring out which stars actually belong to NGC 2516

	#   Quality control                       (ruwe, astrometric_excess_noise)
		# This means filtering out stars whose measurements Gaia couldn't pin down reliably
		# ruwe - Renormalised Unit Weight Error, 
		# astrometric_excess_noise - A companion metric that measures how much extra unexplained noise 
		#	had to be added to the astrometric model to make it fit the data. Units in milli arcseconds.

ADQL = f"""
SELECT
    source_id,
    ra, dec,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    bp_rp,
    phot_g_mean_flux_over_error,
    phot_bp_mean_flux_over_error,
    phot_rp_mean_flux_over_error,
    ag_gspphot,
    ebpminrp_gspphot,
    teff_gspphot,
    logg_gspphot,
    mh_gspphot,
    ruwe,
    astrometric_excess_noise,
    radial_velocity,
    radial_velocity_error,
    non_single_star
FROM
    gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA_DEG}, {DEC_DEG}, {SEARCH_RADIUS})
    )
    AND parallax  BETWEEN {PARALLAX_MIN} AND {PARALLAX_MAX}
    AND pmra      BETWEEN {PM_RA_CENTER  - PM_SIGMA} AND {PM_RA_CENTER  + PM_SIGMA}
    AND pmdec     BETWEEN {PM_DEC_CENTER - PM_SIGMA} AND {PM_DEC_CENTER + PM_SIGMA}
    AND parallax_error < 0.5
    AND phot_g_mean_mag IS NOT NULL
"""

# The function below actually connects to the Gaia database and downloads the star data.
def fetch_data() -> pd.DataFrame:
    print(f"Querying Gaia DR3 for {CLUSTER_NAME} ...\n")
    print(f"  Cone: RA={RA_DEG}°, Dec={DEC_DEG}°, r={SEARCH_RADIUS}° \n")
    print(f"  Parallax: {PARALLAX_MIN}–{PARALLAX_MAX} mas \n")
    print(f"  pmra: {PM_RA_CENTER}±{PM_SIGMA} mas/yr,\n"
          f"  pmdec: {PM_DEC_CENTER}±{PM_SIGMA} mas/yr\n")

    # Configuring to Gaia connection
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1

    job = Gaia.launch_job_async(ADQL, dump_to_file=False, verbose=False)
    table = job.get_results()
    df = table.to_pandas()

    print(f"  → {len(df):,} stars returned before cleaning.\n")
    return df

########################################################
# Data Cleaning

# The function below takes the raw DataFrame from fetch_data() 
# and removes bad/unusable rows, and creates new columns that your model will actually need.
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Drop rows where the core photometric/astrometric columns are absent
    required_cols = [
        "phot_g_mean_mag", "parallax", "bp_rp",
        "pmra", "pmdec",
    ]
    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"  Dropped {before - len(df):,} rows missing core columns.")

    # 2. Astrometric quality cut (RUWE < 1.4)
    before = len(df)
    df = df[df["ruwe"] < 1.4]
    print(f"  Dropped {before - len(df):,} rows with RUWE ≥ 1.4.")

    # 3. Absolute magnitude  M_G = m_G - 5·log_10(1000/d) + 5 - A_G
    #       (distance modulus; d in mas → distance in pc = 1000/d)
    df["distance_pc"] = 1000.0 / df["parallax"]

    # Use A_G when available; fall back to 0 else
    ag = df["ag_gspphot"].fillna(0.0)
    df["abs_g_mag"] = (
        df["phot_g_mean_mag"]
        - 5.0 * np.log10(df["distance_pc"] / 10.0)
        - ag
    )

    # 4. Corrected color index  (BP-RP)_0 = (BP-RP) - E(BP-RP)
    ebr = df["ebpminrp_gspphot"].fillna(0.0)
    df["bp_rp_0"] = df["bp_rp"] - ebr

    # 5. Drop columns that are > 60% missing
    threshold = 0.60
    frac_missing = df.isnull().mean()
    sparse_cols  = frac_missing[frac_missing > threshold].index.tolist()
    if sparse_cols:
        print(f"  Dropping {len(sparse_cols)} sparse columns (>{threshold*100:.0f}% NaN): "
              f"{sparse_cols}")
        df = df.drop(columns=sparse_cols)

    print(f"  Final dataset: {len(df):,} stars, {df.shape[1]} columns.\n")

    # 6. Explicitly drop known sparse GSP-Phot columns
    gspphot_cols = ['ag_gspphot', 'ebpminrp_gspphot', 'teff_gspphot', 'logg_gspphot', 'mh_gspphot']
    gspphot_cols = [c for c in gspphot_cols if c in df.columns]
    df = df.drop(columns=gspphot_cols)
    print(f"  Dropped GSP-Phot columns: {gspphot_cols}")
    print(f"  Remaining NaNs: {df.isnull().sum().sum()}")
    return df


def summarise(df: pd.DataFrame) -> None:
    print("─" * 55)
    print("SUMMARY")
    print("─" * 55)
    print(df[["phot_g_mean_mag", "bp_rp_0", "abs_g_mag",
              "parallax", "pmra", "pmdec"]].describe().round(3).to_string())
    print()
    print("Missing-value fractions:")
    missing = df.isnull().mean().sort_values(ascending=False)
    print(missing[missing > 0].round(3).to_string())
    print("─" * 55)

########################################################
# Learning part

# Config 
INPUT_FILE  = "gaia_ngc2516_raw.csv"
 
# Features the model will cluster on
FEATURE_COLS = [
    "phot_g_mean_mag",               # raw apparent magnitude — NOT absolute
    "phot_bp_mean_mag",              # raw BP magnitude
    "phot_rp_mean_mag",              # raw RP magnitude
    "parallax",                      # distance — model derives depth itself
    "pmra",
    "pmdec"
]

# PCA — how many components to keep before passing to UMAP
N_PCA_COMPONENTS = 2 


# 1. Load data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
 
    # Drop sparse GSP-Phot columns if still present
    drop = ["ag_gspphot", "ebpminrp_gspphot", "teff_gspphot",
            "logg_gspphot", "mh_gspphot"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
 
    print(f"Loaded {len(df):,} stars, {df.shape[1]} columns.")
    print(f"Feature columns: {FEATURE_COLS}\n")
    return df
 
# 2. Standardise 
def standardise(df: pd.DataFrame):
    """
    Z-score normalise all features so no single column dominates due to scale.
    e.g. flux SNR can be in the thousands while bp_rp_0 is 0–3.
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Standardised feature matrix: {X_scaled.shape}")
    return X_scaled, scaler
 
 
# 3. PCA 
def run_pipeline(df):
    X = df[FEATURE_COLS].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print("\nVariance explained:")
    for i, ev in enumerate(explained):
        print(f"  PC{i+1}: {ev*100:.1f}%  (cumulative: {np.cumsum(explained)[i]*100:.1f}%)")

    return X_pca, pca

# 4. Annotation
LABEL_KW = dict(fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="grey", alpha=0.75, lw=0.8))
 
def annotate_pc_space(ax):
    ax.annotate("Giants /\nBright stars", xy=(-5.5, 0.2),
                xytext=(-7.2, 1.8), arrowprops=dict(arrowstyle="->", color="0.3"),
                **LABEL_KW)
    ax.annotate("Main\nSequence", xy=(0.8, 0.0),
                xytext=(-1.5, 2.8), arrowprops=dict(arrowstyle="->", color="0.3"),
                **LABEL_KW)
 
def annotate_hr(ax):
    ax.annotate("Main Sequence", xy=(0.8, 4.5),
                xytext=(1.8, 2.0), arrowprops=dict(arrowstyle="->", color="0.3"),
                **LABEL_KW)
    ax.annotate("Low-mass\nstars", xy=(2.5, 11.0),
                xytext=(0.3, 11.5), arrowprops=dict(arrowstyle="->", color="0.3"),
                **LABEL_KW)


# 5. Plotting Area 
def plot_results(df: pd.DataFrame, X_pca: np.ndarray):
    SCATTER_KW = dict(s=3, alpha=0.7, linewidths=0)
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("PCA Rediscovery of the HR Diagram — NGC 2516", fontsize=14, y=1.01)
 
    # PC1 vs PC2 colored by absolute magnitude
    ax = axes[0, 0]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["abs_g_mag"],
                    cmap="plasma", **SCATTER_KW)
    plt.colorbar(sc, ax=ax, label="Absolute Magnitude $M_G$")
    ax.set(xlabel="PC1", ylabel="PC2",
           title="PC1 vs PC2\nColored by Absolute Magnitude")
    annotate_pc_space(ax)
 
    # PC1 vs PC2 colored by color index
    ax = axes[0, 1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["bp_rp_0"],
                    cmap="RdYlBu_r", **SCATTER_KW)
    plt.colorbar(sc, ax=ax, label="Color Index (BP−RP)₀")
    ax.set(xlabel="PC1", ylabel="PC2",
           title="PC1 vs PC2\nColored by Color Index")
    annotate_pc_space(ax)
 
    # HR diagram colored by PC1
    ax = axes[1, 0]
    sc = ax.scatter(df["bp_rp_0"], df["abs_g_mag"], c=X_pca[:, 0],
                    cmap="plasma", **SCATTER_KW)
    plt.colorbar(sc, ax=ax, label="PC1 value")
    ax.invert_yaxis()
    ax.set(xlabel="Color Index (BP−RP)₀", ylabel="Absolute Magnitude $M_G$",
           title="HR Diagram\nColored by PC1")
    annotate_hr(ax)
 
    # HR diagram colored by PC2
    ax = axes[1, 1]
    sc = ax.scatter(df["bp_rp_0"], df["abs_g_mag"], c=X_pca[:, 1],
                    cmap="RdYlBu_r", **SCATTER_KW)
    plt.colorbar(sc, ax=ax, label="PC2 value")
    ax.invert_yaxis()
    ax.set(xlabel="Color Index (BP−RP)₀", ylabel="Absolute Magnitude $M_G$",
           title="HR Diagram\nColored by PC2")
    annotate_hr(ax)
 
    plt.tight_layout()
    plt.savefig("hr_diagram_pca.png", dpi=150, bbox_inches="tight")
    print("\nSaved → hr_diagram_pca.png")
    plt.show()

def main():
    df_raw   = fetch_data()
    df_clean = clean_and_engineer(df_raw)
    summarise(df_clean)
 
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}\n")
 
    df = load_data(OUTPUT_FILE)
    X_pca, _ = run_pipeline(df)
    plot_results(df, X_pca)

if __name__ == "__main__":
    main()