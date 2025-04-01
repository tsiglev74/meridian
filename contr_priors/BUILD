load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//ads/lift/mmm/modeling/research:__subpackages__"],
    deps = [
        "//third_party/py/immutabledict",
    ],
)

pytype_strict_library(
    name = "train_constants",
    srcs = ["train_constants.py"],
    visibility = ["//ads/lift/mmm/modeling/research:__subpackages__"],
)

pytype_strict_library(
    name = "train_meridian_lib",
    srcs = ["train_meridian.py"],
    deps = [
        ":constants",
        ":train_constants",
        "//ads/lift/mmm/modeling/distribute:utils",
        "//ads/lift/mmm/modeling/research/contr_priors/analysis:optimizer",
        "//ads/lift/mmm/modeling/research/contr_priors/data:input_data",
        "//ads/lift/mmm/modeling/research/contr_priors/data:load",
        "//ads/lift/mmm/modeling/research/contr_priors/model",
        "//ads/lift/mmm/modeling/research/contr_priors/model:prior_distribution",
        "//ads/lift/mmm/modeling/research/contr_priors/model:spec",
        "//base/python:pywrapbase",
        "//file/colossus/public:cns",
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/joblib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/xarray",
    ],
)

pytype_strict_binary(
    name = "train_meridian",
    srcs = ["train_meridian.py"],
    deps = [
        ":train_meridian_lib",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)
