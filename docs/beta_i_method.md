# Reproducible beta_i Method

## Definition

For individual `i` and nutrient `k`, `beta_i,k` is the central finite-difference change in a fitted gut microbiome health index along a nutrient-linked microbiome perturbation direction:

```text
beta_i,k = (H(M_i + dose * P_k) - H(M_i - dose * P_k)) / (2 * dose)
```

`M_i` is the individual's CLR-transformed genus vector. `H` is a GMWI2-style sparse gut microbiome health index trained to distinguish cMD healthy versus non-healthy samples. `P_k` is the nutrient-to-genus perturbation vector for nutrient `k`, derived from the L7 nutrient-genus bridge and constrained by the literature-defined, expert-reviewed MAC/LIPID channel content. Its direction combines bridge evidence, the fitted health-coefficient sign, and nutrient-level evidence direction (`+1` for MAC, `-1` for LIPID, and a configurable conservative direction for OTHER). Each direction has a locked L2 norm of 2.0 before its channel weight is applied. The locked dose is 0.25, and the finite-difference result is clipped to an absolute numerical limit of 25 only after differencing.

## Source Inspirations

- GMWI2: sparse microbiome health-index construction from taxonomic profiles.
- DI-GM: literature/evidence-oriented beneficial versus unfavorable gut microbiota directionality, adapted here from diet-level components to nutrient-level perturbations.
- GMMAD/L7 nutrient-genus bridge: curated microbe-metabolite-disease associations are used as bridge evidence for nutrient-linked genus perturbations.
- GMNPS expert masks: MAC and LIPID channel membership defines which nutrients receive primary channel weight.

## Reproducible Inputs

- `data/project_data/predict_multi/L5_gmnps_pipeline_v2/M_clr.parquet`
- `data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet`
- `data/project_data/predict_multi/L7_nutrient_bridge/B_nutrient_genus.parquet`
- `data/project_data/predict_multi/L8_inference_bundle_v2/nutrient_index.json`

## Reproducible Command

```bash
PYTHONPATH=code/src .venv/bin/python code/src/scripts/build_beta_i_weights.py \
  --root "$PWD" \
  --output-dir "$PWD/data/project_data/predict_multi/L7_nutrient_bridge_beta_i"
```

## Claim Boundary

`beta_i,k` is a microbiome-derived calibration input that records how a fitted microbiome health index changes along an author-specified nutrient-linked direction. It is not a causal estimate of nutrient intake, not a clinical treatment effect, and not a validated postprandial response coefficient.
