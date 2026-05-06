# Machine Learning Stellar Classification
### Rediscovering the HR Diagram — NGC 2516

**Krupa Pothiwala · Florida Institute of Technology · May 2026**

---

## Overview

This project investigates whether unsupervised machine learning can rediscover the structure of the Hertzsprung-Russell (HR) diagram from raw, unlabelled stellar data — with no prior knowledge of stellar physics provided to the model.

Star data for the open cluster NGC 2516 was retrieved from the ESA Gaia DR3 catalog. PCA was applied to four raw photometric and astrometric features. Without being told anything about stellar physics, the model recovered the primary structure of the HR diagram, with the two principal components mapping directly onto stellar brightness and distance.

---

## Research Question

> Can unsupervised machine learning rediscover the structure of the HR diagram from raw, unlabelled stellar data with no prior knowledge of stellar physics?

---

## Data

- **Source:** ESA Gaia Data Release 3 (DR3)
- **Retrieved via:** `astroquery.gaia` (Python)
- **Target cluster:** NGC 2516 (RA = 119.517°, Dec = −60.752°)
- **Search radius:** 1.0°
- **Membership filters applied:**
  - Parallax: 1.5 – 3.5 mas
  - pmRA: −4.7 ± 3.0 mas/yr
  - pmDec: 11.2 ± 3.0 mas/yr
  - Parallax error < 0.5 mas
- **Final sample:** 2,670 member stars

---

## Pipeline

### Cleaning
- Dropped rows missing apparent magnitude, parallax, colour index, or proper motion
- Applied RUWE quality cut (RUWE < 1.4) to remove unreliable astrometric solutions
- Dropped columns with more than 60% missing values
- Removed GSP-Phot derived columns (too incomplete)
- Derived absolute magnitude: `M_G = m_G - 5·log10(d) + 5 - A_G`
- Derived corrected colour index: `(BP-RP)_0 = (BP-RP) - E(BP-RP)`

### Machine Learning
| Step | Method | Purpose |
|------|--------|---------|
| 1 | Z-score normalisation | Rescale features to equal footing |
| 2 | PCA (2 components) | Compress 4 features, capture structure |
| 3 | UMAP + HDBSCAN | Explored, discarded (see below) |

**Features used:** apparent magnitudes in G, BP, and RP bands + parallax

### Why not UMAP + HDBSCAN?
UMAP and HDBSCAN were applied on top of the PCA output but added no meaningful scientific insight. HDBSCAN traced UMAP's layout geometry rather than genuine density structure in the data — a form of **pipeline bias**. Since the goal was to recover a continuous physical structure rather than partition stars into discrete groups, PCA alone was adopted as the final method.

---

## Results

PCA retained **99.7% of total variance** in just 2 components.

| Component | Physical Meaning |
|-----------|-----------------|
| PC1 | Brightness gradient — traces the full main sequence from hot blue stars to faint red stars. Giants separate naturally at negative PC1. |
| PC2 | Residual scatter — largely flat across the main sequence, picks up minor parallax depth variation. No new astrophysical information. |

The model recovered the HR diagram structure entirely without supervision — rediscovering over a century of stellar astronomy from raw numbers alone.

---

## Requirements

```
astroquery
astropy
pandas
numpy
matplotlib
scikit-learn
umap-learn
hdbscan
```

Install with:
```bash
pip install astroquery astropy pandas numpy matplotlib scikit-learn umap-learn hdbscan
```

---

## File Structure

```
.
├── gaia_ngc2516_raw.csv          # Raw data downloaded from Gaia DR3
├── gaia_ngc2516_clustered.csv    # Cleaned data with PCA coordinates
├── hr_diagram_pca.png            # Result plot (PCA + HR diagram)
├── main.py                       # Full pipeline script
├── report/
│   └── main.tex                  # Written report (LaTeX)
└── slides/
    └── presentation.tex          # Beamer presentation (LaTeX)
```

---

## References

- European Space Agency (2022). *Gaia Archive: Extract Data.* https://www.cosmos.esa.int/web/gaia-users/archive/extract-data
- European Space Agency (2022). *Gaia Data Release 3 (DR3).* https://www.cosmos.esa.int/web/gaia/dr3
- Ginsburg, A. et al. (2023). *Astroquery: Gaia Module Documentation.* https://astroquery.readthedocs.io/en/latest/gaia/gaia.html
- Pearce, L. (2022). *RUWE as an Indicator of Multiplicity.* http://www.loganpearcescience.com/research/RUWE_as_an_indicator_of_multiplicity.pdf
