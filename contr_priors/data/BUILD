load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
    default_visibility = [
        "//third_party/py/meridian:internal",
        "//third_party/py/meridian:meridian_users",
    ],
)

pytype_strict_library(
    name = "input_data",
    srcs = ["input_data.py"],
    deps = [
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//third_party/py/meridian/data:arg_builder",
        "//third_party/py/meridian/data:time_coordinates",
        "//third_party/py/numpy",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "load",
    srcs = ["load.py"],
    deps = [
        ":input_data",
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//third_party/py/immutabledict",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)
