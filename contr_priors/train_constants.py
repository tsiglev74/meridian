"""Constant variables used throughout the distribute package."""

# Configurable Parameter Strings
N_DRAWS = 'n_draws'
N_CHAINS = 'n_chains'
N_ADAPT = 'n_adapt'
N_BURNIN = 'n_burnin'
N_KEEP = 'n_keep'
CONTROL_POPULATION_SCALING_ID = 'control_population_scaling_id'
MAX_LAG = 'max_lag'
PAID_MEDIA_PRIOR_TYPE = 'paid_media_prior_type'
SIGMA_POPULATION_SCALING = 'sigma_population_scaling'
MEDIA_EFFECTS_DIST = 'media_effects_dist'
PRIOR_CONTAINER = 'prior_container'
PRIOR_CONTAINER_PATH = 'prior_container_path'
PATH_TYPE = 'path_type'
N_DMA_TO_KEEP = 'n_dma_to_keep'
N_LAG_HISTORY_PERIODS = 'n_lag_history_periods'
CUSTOM_PRIORS = 'custom_priors'
USE_POSTERIOR = 'use_posterior'
SPEND_CONSTRAINT_LOWER = 'spend_constraint_lower'
SPEND_CONSTRAINT_UPPER = 'spend_constraint_upper'
FIXED_BUDGET = 'fixed_budget'
TARGET_ROI = 'target_roi'
TARGET_MROI = 'target_mroi'
USE_OPTIMAL_FREQUENCY = 'use_optimal_frequency'
USE_KPI = 'use_kpi'
SESSION_NAME = 'session_name'
DATA_PATH = 'data_path'
LDAP = 'ldap'
CONFIG_PATH = 'config_path'
KNOTS = 'knots'
KNOTS_TYPE = 'knots_type'
TEST_PCT = 'test_pct'
N_SEQUENTIAL = 'n_sequential'
KPI_TYPE = 'kpi_type'
KPI_COLUMN = 'kpi_column'
REVENUE_PER_KPI_COLUMN = 'revenue_per_kpi_column'
GEO_COLUMN = 'geo_column'
TIME_COLUMN = 'time_column'
IMPRESSIONS = 'impressions'
FREQ = 'freq'
CONTROL = 'control'
GQV = 'gqv'
INCLUDE_GQV = 'include_gqv'
SPEND = 'spend'
PRIOR_PATH = 'prior_path'
PRIOR = 'prior'
UNIQUE_SIGMA_FOR_EACH_GEO = 'unique_sigma_for_each_geo'
BASELINE_GEO = 'baseline_geo'
DISTRIBUTION = 'distribution'
DISTRIBUTION_TYPE = 'distribution_type'
BIJECTOR = 'bijector'
BIJECTOR_TYPE = 'bijector_type'
NAME = 'name'
OPTIMIZER_DATA_FILE_NAME = 'optimizer_data.pkl'
MODEL_FILE_NAME = 'model.pkl'
N_OPTIMIZATION_PERIODS = 'n_optimization_periods'
NON_MEDIA_PREFIX = 'non_media'
PAID_MEDIA_PRIOR_TYPE = 'paid_media_prior_type'
ORGANIC_MEDIA_PRIOR_TYPE = 'organic_media_prior_type'
NON_MEDIA_TREATMENTS_PRIOR_TYPE = 'non_media_treatments_prior_type'
NON_MEDIA_TREATMENTS_BASELINE_VALUES = 'non_media_treatments_baseline_values'

# Media Effect Distributions
NORMAL = 'normal'
LOG_NORMAL = 'log_normal'

# Adstock function names
ADSTOCK = 'adstock'
HILL = 'hill'
HILL_BEFORE_ADSTOCK = 'hill_before_adstock'

INT = 'int'
LIST = 'list'
CSV = 'csv'
XARRAY = 'xarray'
KNOTS_TYPES = {INT: int, LIST: list}
PATH_TYPES = (CSV, XARRAY)

PARAM_DEFAULTS = {
    N_DRAWS: 2000,
    N_CHAINS: 10,
    N_ADAPT: 50,
    N_BURNIN: 50,
    N_KEEP: 100,
    CONTROL_POPULATION_SCALING_ID: None,
    MAX_LAG: 10,
    PAID_MEDIA_PRIOR_TYPE: 'roi',
    SIGMA_POPULATION_SCALING: False,
    MEDIA_EFFECTS_DIST: NORMAL,
    PRIOR_CONTAINER_PATH: None,
    # Will default to number of geos
    N_DMA_TO_KEEP: 0,
    N_LAG_HISTORY_PERIODS: 10,
    USE_POSTERIOR: True,
    SPEND_CONSTRAINT_LOWER: None,
    SPEND_CONSTRAINT_UPPER: None,
    FIXED_BUDGET: True,
    TARGET_ROI: None,
    TARGET_MROI: None,
    USE_OPTIMAL_FREQUENCY: True,
    USE_KPI: False,
    N_OPTIMIZATION_PERIODS: None,
    KNOTS: None,
    KNOTS_TYPE: None,
    N_SEQUENTIAL: 1,
    KPI_TYPE: 'revenue',
    GEO_COLUMN: 'dma_no',
    TIME_COLUMN: 'date_week',
    KPI_COLUMN: 'sales_usd',
    REVENUE_PER_KPI_COLUMN: None,
    SESSION_NAME: 'Train_Meridian',
    INCLUDE_GQV: True,
}

MODEL_SPEC_DEFAULTS = {
    CUSTOM_PRIORS: {},
    PRIOR_PATH: None,
    MEDIA_EFFECTS_DIST: LOG_NORMAL,
    HILL_BEFORE_ADSTOCK: False,
    MAX_LAG: 8,
    UNIQUE_SIGMA_FOR_EACH_GEO: False,
    PAID_MEDIA_PRIOR_TYPE: 'roi',
    KNOTS: None,
    BASELINE_GEO: None,
    CONTROL_POPULATION_SCALING_ID: None,
    TEST_PCT: 0.2,
    PAID_MEDIA_PRIOR_TYPE: 'roi',
    ORGANIC_MEDIA_PRIOR_TYPE: 'contribution',
    NON_MEDIA_TREATMENTS_PRIOR_TYPE: 'contribution',
    NON_MEDIA_TREATMENTS_BASELINE_VALUES: None,
}

SINGLE_VAL_PARAMS = [
    PAID_MEDIA_PRIOR_TYPE,
    ORGANIC_MEDIA_PRIOR_TYPE,
    NON_MEDIA_TREATMENTS_PRIOR_TYPE,
    SIGMA_POPULATION_SCALING,
    PRIOR_CONTAINER_PATH,
    FIXED_BUDGET,
    TARGET_ROI,
    TARGET_MROI,
    USE_OPTIMAL_FREQUENCY,
    SPEND_CONSTRAINT_LOWER,
    SPEND_CONSTRAINT_UPPER,
    N_DRAWS,
    N_ADAPT,
    N_BURNIN,
    N_KEEP,
    MAX_LAG,
    N_LAG_HISTORY_PERIODS,
    N_DMA_TO_KEEP,
    MEDIA_EFFECTS_DIST,
    N_SEQUENTIAL,
    UNIQUE_SIGMA_FOR_EACH_GEO,
    KPI_TYPE,
    HILL_BEFORE_ADSTOCK,
    CUSTOM_PRIORS,
]

# Runtime stats strings
USE_GPU = 'USE_GPU'
N_GPUS = 'N_GPUS'
N_CPUS = 'N_CPUS'
RAM_GIB = 'RAM_GIB'
JOB_PRIORITY = 'JOB_PRIORITY'
SAMPLING_WALL_ELAPSED_MS = 'sampling_wall_elapsed_ms'
SAMPLING_PROCESS_ELAPSED_MS = 'sampling_process_elapsed_ms'
SAMPLING_PEAK_GPU_MEM_GB = 'sampling_peak_gpu_mem_gb'
OPTIMIZATION_WALL_ELAPSED_TIME = 'optimization_wall_elapsed_time'
OPTIMIZATION_PROCESS_ELAPSED_TIME = 'optimization_process_elapsed_time'
OPTIMIZATION_PEAK_GPU_MEM_GB = 'optimization_peak_gpu_mem_gb'
