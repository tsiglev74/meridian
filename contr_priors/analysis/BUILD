load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
    default_visibility = [
        "//third_party/py/meridian:internal",
        "//third_party/py/meridian:meridian_users",
    ],
)

pytype_strict_library(
    name = "analyzer",
    srcs = ["analyzer.py"],
    deps = [
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//ads/lift/mmm/modeling/research/contr_priors/model",
        "//third_party/py/meridian/model:adstock_hill",
        "//third_party/py/meridian/model:transformers",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    data = [
        "//third_party/py/meridian/analysis/templates",
        "//third_party/py/meridian/analysis/templates:style.css",
    ],
    deps = [
        ":analyzer",
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//ads/lift/mmm/modeling/research/contr_priors/model",
        "//third_party/py/altair",
        "//third_party/py/jinja2",
        "//third_party/py/meridian/analysis:formatter",
        "//third_party/py/meridian/analysis:summary_text",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/xarray",
    ],
)
