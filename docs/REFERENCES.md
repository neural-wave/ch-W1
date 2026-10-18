# References

## Methods

- Sax, C. and Steiner, P. (2013). Temporal Disaggregation of Time Series. *The R Journal* 5(2).
  https://journal.r-project.org/articles/RJ-2013-028/ — aggregate-preserving disaggregation;
  basis for the mean-preserving cumulative-interpolation baseline.
- Fritsch, F. N. and Carlson, R. E. (1980). Monotone Piecewise Cubic Interpolation. *SIAM J. Numer. Anal.* 17(2).
  Used through SciPy `PchipInterpolator` on the cumulative hourly curve.
- Troncoso, A. et al. (2007). Electricity Market Price Forecasting Based on Weighted Nearest
  Neighbors Techniques. *IEEE Trans. Power Systems*. — analogue-memory experiment (not promoted).
- Li, Y. et al. (2025). Multi-task Graph Adaptive Learning for Multivariate Electricity Price
  Short-Term Forecasting in Australia's NEM. *IEEE Trans. Power Systems*. — cross-zone shrinkage
  experiment (not promoted).
- Ho-Nguyen, N., Alipour, H., Panagiotelis, A. and Athanasopoulos, G. (2026). Optimal Forecast
  Reconciliation for Quantiles. Monash EBS working paper 04/26. — quantile decision experiment (not promoted).

## Libraries

- NumPy, pandas, SciPy (`PchipInterpolator`), LightGBM (`regression_l1`), PyYAML, pytest.
- Optional: Streamlit and Plotly (dashboard), Matplotlib and Jupyter (analysis).

## Challenge material

- AET, *Neural Wave 2026 challenge brief* (v02) and participant dataset README
  ([data/README_participants.txt](../data/README_participants.txt)).
