# Data download and preparation

The numerical experiments use publicly available drilling records from the U.S. Department of Energy Utah FORGE project, distributed through the Geothermal Data Repository (GDR). The repository does not redistribute the full datasets.

## Well 16A(78)-32

- Dataset page: https://gdr.openei.org/submissions/1283
- Persistent DOI: https://doi.org/10.15121/1776602
- Dataset title: *Utah FORGE: Well 16A(78)-32 Drilling Data*

Download the following resources from the dataset page:

- `Standard 16A78-32 Drilling Data 10s Intervals.csv` for the standardized 10-second numerical records used as the principal numerical source;
- `16A78-32 Survey Data Higher Depth Resolution.xlsx` for the survey-consistent trajectory;
- `16A78-32 Daily Reports Complete Set.zip`, `16A78-32 Summary of Daily Operations.pdf`, and the mud-log resources for geological and operational context.

The GDR page also provides a “Download All Resources” option. The full submission is approximately 2.57 GB.

## Well 16B(78)-32

- Dataset page: https://gdr.openei.org/submissions/1516
- Persistent DOI: https://doi.org/10.15121/1998591
- Dataset title: *Utah FORGE: Well 16B(78)-32 Drilling Data*

Download the following resources from the dataset page:

- `Pason Data.zip`; use its 10-second Pason records as the principal numerical source;
- `Well Survey.zip` for the survey-consistent trajectory;
- `Daily Reports.zip`, `Mud Logs.zip`, and `End of Well Report.pdf` for geological and operational context.

The GDR page also provides a “Download All Resources” option. The full submission is approximately 393 MB.

## Expected local layout

After downloading and cleaning the public resources, prepare:

```text
data_cleaned_csv/
  second_stage/
    16A_common_model_features.csv
    16B_common_model_features.csv
  depth_level/
    16A_depth_level_bin_1p0ft_clean_features.csv
    16B_depth_level_bin_1p0ft_clean_features.csv
  data_qwen/
    16A_qwen_depth_context.csv
    16B_qwen_depth_context.csv
```

Run `src/build_depth_level_dataset.py` to convert the common numerical feature tables to the uniform 1-ft depth grid. The 100-ft geological context files must follow the schema shown in `examples/data/deidentified_context_sample.csv` and must not contain target angles or future survey information.

## Dataset citations

McLennan, J., Nash, G., Moore, J., Skowron, G., & Woolsey, S. (2021). *Utah FORGE: Well 16A(78)-32 Drilling Data* [Data set]. Geothermal Data Repository. https://doi.org/10.15121/1776602

McLennan, J., Mock, B., Swearingen, L., Baldwin, R., Hodder, M., Vetsak, A., Kuhns, A. T., Breland, J., & England, K. (2023). *Utah FORGE: Well 16B(78)-32 Drilling Data* [Data set]. Geothermal Data Repository. https://doi.org/10.15121/1998591

Unless otherwise noted on an individual resource, GDR content is made available under Creative Commons Attribution 4.0.

