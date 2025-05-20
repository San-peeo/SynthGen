# SynthGen: A Gravitational Simulator for Planetary Interior Modelling

Determining the internal structure of planetary bodies from gravitational observations is a key challenge in planetary geophysics. Gravity inversion techniques make it possible to estimate mass distribution by combining information on a body's shape, gravitational field, and rotational dynamics. However, gravity data alone present a well-known ambiguity between mass magnitude and depth, making the interpretation of internal layering a complex inverse problem. 

We present SynthGen, a code developed to simulate the gravitational response of planetary bodies based on parametric interior models. It exploits the spherical harmonics framework described in Wieczorek [1], through the computation of gravitational harmonic coefficients [C_{nm}, S_{nm}] which characterise the global gravitational field, thanks to the SHTools [2] routines. SynthGen takes as input a simplified multi-layer interior model, assuming them homogenous. Model parameters consist of the number of internal layers, their mean thickness and density, and eventually, the topography of internal interfaces. On the latter, several ways are implemented: sphere, polar/equatorial flattened ellipsoid, randomly generated topography, downwarded Bouguer anomaly (see Wieczorek & Phillips [2], avoiding isostacy assumptions) and finally an input user grid. Given these inputs, SynthGen computes the corresponding gravitational potential, Free-Air anomalies, and Bouguer anomaly fields for the modelled body, generating full-resolution global maps.

SynthGen outputs can be used in two ways: 
1) If it is used to simulate a known body, so a gravity model is already available, the synthetic results can be compared to the real measurements, assessing the validity of the evaluated interior model and measuring its performance through different metrics. In this case, SynthGen performed an automated parameter-space exploration (controlled by the user). By randomly sampling model parameters within physically plausible bounds that it is user-configurable (constrained by the satisfaction of the conservation of total mass and moment of inertia, together with external shape constraints), it iteratively evaluates a wide range of configurations. The optimal internal structure is determined by identifying the parameter set that minimises discrepancies between simulated and observed gravitational data. This is performed through a suite of statistical metrics (e.g. RMSE, MAE, R2, SSIM, NCC, PSNR, etc.), finally combined into one.

2) In addition to this procedure, SynthGen can be used predictively in case of an “unmeasured” body. It enables forward modelling of gravitational signals expected from future targets (for example Ganymede, for ESA’s JUICE mission). It can thus serve as a valuable tool for testing theoretical interior structures and simulating their measurable gravitational signatures.

By combining analytical modelling, numerical efficiency, and flexibility across planetary scenarios, SynthGen offers a useful platform for planetary interior investigations from the gravitational point of view. It can handle various planetary shapes, datasets, and scientific objectives, and it is user configurable, together with already implemented configuration files for Mercury, Venus, Earth and Moon, together with a model of Ganymede.


![Comparison on Mercury between Synthetic generated data and MESSENGER-derived model](https://github.com/user-attachments/assets/7fa9d8eb-02e0-483a-9b52-42c1022995bc)


Please cite this if you use it in your research



# Environment Setup


## Linux/Windows WSL
```bash
# Create a virtual environment
python -m venv myenv

# Activate the environment
source myenv/bin/activate

# Install packages
pip install -r requirements.txt
```


# Datasets:
see Planets_ConfigFiles.py; In this module, configuration classes for the main terrestrial planets and Ganymede, including their physical parameters, gravity and topography data sources, and interior structure models, are provided.  
Each planet class contains methods to retrieve bulk parameters (.bulk()), data file information (.data()), and interior structure parameters for different numbers of layers (.interiors()).

## Mercury 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 2439.4               | km           | Reference radius                            |
| GM_const          | 2.2031863566e+13     | m³/s²        | Gravitational constant                      |
| ref_mass          | 3.301e+23            | kg           | Reference mass                              |
| ref_rho           | 5427                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 8.264e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.34597              | (I/MR²)      | Moment of inertia factor (Margot et al 2018)|
| r_e_fact          | 1.0005               | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9995               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                                      | Format   | Header | Reference                        |
|-------------|------------------------------------------------|----------|--------|------------------------------|
| Gravity     | HgM009.sha                 | shtools  | True   | A. Genova et al., ‘Regional variations of Mercury’s crustal density and porosity from MESSENGER gravity data’, Icarus, 2023, doi: 10.1016/j.icarus.2022.115332.          |
| Topography  | gtmes_150v05_sha_nohead.txt | shtools  | False  | pds-geosciences.wustl.edu - /messenger/mess-h-rss_mla-5-sdp-v1/messrs_1001/                 |

- **Bouguer density:** 2900 kg/m³  
- **Crustal thickness filter (n_half):** 40

### Implemented Interior Models

#### 3 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Core      | 6992            | 2039            | polar flattened sphere|
| Mantle      | 3200            | 2404            | downwarded Bouguer anomalies|
| Crust      | 2900            | 2439.4            | surface|

#### 4 Layers 
J.-L. Margot, S. A. H. II, E. Mazarico, S. Padovan, and S. J. Peale, ‘Mercury’s Internal Structure’, 2018, pp. 85–113. doi: 10.1017/9781316650684.005
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 8652.52            | 666.577            | sphere|
| Outer Core      | 6909.98            | 2023.66            | polar flattened sphere|
| Mantle      | 3343.35            |  2402.61            | downwarded Bouguer anomalies|
| Crust      | 2903.03            | 2439.4            | surface|



---

## Venus 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 6051.8               | km           | Reference radius                            |
| GM_const          | 3.248585920790000e+14| m³/s²        | Gravitational constant                      |
| ref_mass          | 4.8673e+24           | kg           | Reference mass                              |
| ref_rho           | 5243                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 3.232e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.337                | (I/MR²)      | Moment of inertia factor (Margot et al 2021)|
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 1.0                  | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | References                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | shgj180u_noheader.a01            | shtools  | True   | A. S. Konopliv, W. B. Banerdt, and W. L. Sjogren, ‘Venus Gravity: 180th Degree and Order Model’, Icarus, 1999, doi: 10.1006/icar.1999.6086.                             |
| Topography  | VenusTopo719.shape               | shtools  | False  | Wieczorek, M. A. (2015). Spherical harmonic model of the planet Venus: VenusTopo719 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3870926 |

- **Bouguer density:** 2800 kg/m³  
- **Crustal thickness filter (n_half):** 80

### Implemented Interior Models
#### 3 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Core      | 13000            | 3200            | polar flattened sphere|
| Mantle      | 3300            | 6020            | downwarded Bouguer anomalies|
| Crust      | 2800            | 6051.8          | surface|

---

## Earth 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 6378.137             | km           | Reference radius                            |
| GM_const          | 3.986004418e+14      | m³/s²        | Gravitational constant                      |
| ref_mass          | 5.9722e+24           | kg           | Reference mass                              |
| ref_rho           | 5514                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 1.992e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3308               | (I/MR²)      | Moment of inertia factor (Williams and James, 1994)|
| r_e_fact          | 0.9999               | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9970               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | EGM2008_to2190_TideFree          | shtools  | False  | N. K. Pavlis, S. A. Holmes, S. C. Kenyon, and J. K. Factor, ‘The development and evaluation of the Earth Gravitational Model 2008 (EGM2008)’, Journal of Geophysical Research: Solid Earth, 2012, doi: 10.1029/2011JB008916. |
| Topography  | Earth2014.BED2014.degree10800.bshc| bshc    | False  |                              |

- **Bouguer density:** 1800 kg/m³  
- **Crustal thickness filter (n_half):** 80

### Implemented Interior Models
#### 5 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 13088.5            | 1221.5            | sphere|
| Outer Core      | 12581.5            | 3480.0            | polar flattened sphere|
| Lower Mantle    | 7956.5             | 5701.0            | polar flattened sphere|
| Upper Mantle    | 7090.9             | 6151.0            | polar flattened sphere|
| Crust           | 2800               | 6371.0            | surface|

#### 8 Layers 
| Layer           | Density (kg/m³) | Radius (km) | Interface Type              |
|-----------------|-----------------|-------------|-----------------------------|
| Inner Core      | 13088.5         | 1221.5      | sphere                      |
| Outer Core      | 12581.5         | 3480.0      | polar flattened sphere      |
| Lower Mantle    | 7956.5          | 5701.0      | polar flattened sphere      |
| Upper Mantle    | 7090.9          | 6151.0      | polar flattened sphere      |
| Transition Zone | 2691.0          | 6346.0      | downwarded Bouguer anomalies|
| Lower Crust     | 2900            | 6356.0      | polar flattened sphere      |
| Upper Crust     | 2600            | 6368.0      | surface                     |
| Sediments       | 1020            | 6371.0      | surface                     |
---

## Moon 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 1738.1               | km           | Reference radius                            |
| GM_const          | 4.9028001218467998e+12| m³/s²       | Gravitational constant                      |
| ref_mass          | 0.07346e+24          | kg           | Reference mass                              |
| ref_rho           | 3344                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 2.7e-06              | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3929               | (I/MR²)      | Moment of inertia factor (Williams and James, 1996)|
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9988               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | GRGM1200l_data.txt               | shtools  | True   | Lemoine, F. G., et al. (2014), GRGM900C: A degree 900 lunar gravity model from GRAIL primary and extended mission data, Geophys. Res. Lett., doi:10.1002/2014GL060027, Goossens, S., et al. (2016), A Global Degree and Order 1200 Model of the Lunar Gravity Field using GRAIL Mission Data, Lunar and Planetary Science Conference, Houston, TX, Abstract #1484.|
| Topography  | MoonTopo2600p.shape              | shtools  | False  |                              |

- **Bouguer density:** 2900 kg/m³  
- **Crustal thickness filter (n_half):** 40

### Implemented Interior Models
#### 4 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 0            | 0            | sphere|
| Outer Core      | 0            | 0            | sphere|
| Mantle    | 0            | 0           | sphere|
| Crust    | 0            | 0           | sphere|
---

## Ganymede 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 2631.2               | km           | Reference radius                            |
| GM_const          | 9.8780e+12           | m³/s²        | Gravitational constant                      |
| ref_mass          | 1.48e+23             | kg           | Reference mass                              |
| ref_rho           | 1942                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 8.264e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3115               | (I/MR²)      | Moment of inertia factor                    |
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 1.0                  | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | None                             | shtools  | True   |                              |
| Topography  | None                             | shtools  | False  |                              |

- **Bouguer density:** 920 kg/m³  
- **Crustal thickness filter (n_half):** 25

### Implemented Interior Models
#### 7 Layers
| Layer           | Density (kg/m³) | Radius (km) | Interface Type |
|-----------------|-----------------|-------------|---------------|
| Core            | 8000            | 570         | sphere        |
| Mantle         | 3400            | 1820        | sphere        |
| Crust         | 3100            | 1870        | rng           |
| Ice VI         | 1320            | 2000        | sphere        |
| Ice V         | 1235            | 2280        | sphere        |
| Ocean         | 1100            | 2460        | sphere        |
| Ice I       | 920             | 2631.2      | surface(rng)       |





# Code Description
- **unet_custom.py**: contains all classes definitions for custom U-Nets developed in our work. It also contains functions to compress, transfer weights and rank adapters
- **model_library.py**: contains all main functions needed to run training, evaluation and default plotting. It also contains classes definition for the datasets.
- **model_library_classic.py**: contains functions of model_library customized for classic algorithms (e.g. otsu, canny and hybrid)
- **model_train.py**: contains functions to train deep learning models using synthetic-moon-dataset. It can be run with wandb (use sweep_config.yaml) or locally.
- **model_test.py**: contains functions to test and plot results of trained deep learning models 
- **model_test_classic.py**: contains functions to test classic algorithms
- **model_test_extdevice.py**: evaluates performance of the deep learning model using external devices (e.g. jetson nano and raspberry pi)
- **adapters_flops_params.py**: evaluates flops and number of trained params for fine-tuning methods and adapters
- **baseline_flops_params.py**: evaluates flops and number of trained params for baseline methods
- **fusemethod_flops_params.py**: evaluates flops and number of trained params for adapter-fusing methods
- **adapters_pareto.py**: plots pareto curves for adapters and traditional fine-tuning methods
- **model_storage.py**: evaluates storage memory of deep learning models
- **plot_layer_ablation.py**: plots results for layer-by-layer ablation study
- **plotpreds.py**: plots predictions for different datasets (MarsDatasetv3 and Real-Moon)
- pvalue.py: evaluates p-value for Shapiro-Wilk and Wilcoxon signed-rank test
