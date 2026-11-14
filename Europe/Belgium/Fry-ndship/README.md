# Fry-ndship

We estimated potato-field edge lengths with linear regression, deriving the short side as `10000 * area_ha / long_side_m` and using both sides plus declared ditch length as inputs.
An 80/20 split with `random_state=42` gave a test R² of 0.999984 and MAE of 0.786 metres.
Plotting measured minus predicted edge length against predicted edge length revealed **SUPERVISION**, supporting the candidate flag `SIGMOID_SUPERVISION`; the lettering also appeared in held-out rows and disappeared after shuffling residuals.
Run `python Europe/Belgium/Fry-ndship/main.py` from the repository root to reproduce the fitting results and figures in `figures/`.
The flag remains unverified on the platform, and the comparison figure documents alternative residual words from different feature sets, including an on-site hedge measurement unsuitable for predicting unsurveyed fields.
