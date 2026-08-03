# Reproducible beta_i Method

## Definition

For individual `i` and nutrient `k`, `beta_i,k` is the finite-difference change in a fitted gut microbiome health index after applying a nutrient-linked microbiome perturbation:

```text
beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose
```

`M_i` is the individual's CLR-transformed genus vector. `H` is a GMWI2-style sparse gut microbiome health index trained to distinguish cMD healthy versus non-healthy samples. `P_k` is the nutrient-to-genus perturbation vector for nutrient `k`, derived from the L7 nutrient-genus bridge and constrained by the expert-reviewed MAC/LIPID channel design.

## Source Inspirations

- GMWI2: sparse microbiome health-index construction from taxonomic profiles.
- DI-GM: literature/evidence-oriented beneficial versus unfavorable gut microbiota directionality, adapted here from diet-level components to nutrient-level perturbations.
- GMNPS expert masks: MAC and LIPID channel membership defines which nutrients receive primary channel weight.

## Reproducible Inputs

- `data/project_data/predict_multi/L5_gmnps_pipeline_v2/M_clr.parquet`
- `data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet`
- `data/project_data/predict_multi/L7_nutrient_bridge/B_nutrient_genus.parquet`
- `data/project_data/predict_multi/L8_inference_bundle_v2/nutrient_index.json`

## Reproducible Command

```bash
PYTHONPATH=code/src .venv/bin/python code/src/scripts/build_beta_i_weights.py \
  --root /Users/fengxi.25/Desktop/GMNPS \
  --output-dir /Users/fengxi.25/Desktop/GMNPS/data/project_data/predict_multi/L7_nutrient_bridge_beta_i
```

## Claim Boundary

`beta_i,k` is a microbiome-derived calibration weight that estimates how a nutrient-linked microbiome perturbation changes a fitted microbiome health index. It is not a causal estimate of nutrient intake, not a clinical treatment effect, and not a validated postprandial response coefficient.
