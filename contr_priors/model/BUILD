load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
    default_visibility = [
        "//third_party/py/meridian:internal",
        "//third_party/py/meridian:meridian_users",
    ],
)

pytype_strict_library(
    name = "media",
    srcs = ["media.py"],
    deps = [
        ":spec",
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//ads/lift/mmm/modeling/research/contr_priors/data:input_data",
        "//third_party/py/meridian/model:transformers",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "model",
    srcs = [
        "model.py",
        "posterior_sampler.py",
        "prior_sampler.py",
    ],
    deps = [
        ":media",
        ":prior_distribution",
        ":spec",
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//ads/lift/mmm/modeling/research/contr_priors/data:input_data",
        "//third_party/py/arviz",
        "//third_party/py/joblib",
        "//third_party/py/meridian/data:time_coordinates",
        "//third_party/py/meridian/model:adstock_hill",
        "//third_party/py/meridian/model:knots",
        "//third_party/py/meridian/model:transformers",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_library(
    name = "prior_distribution",
    srcs = ["prior_distribution.py"],
    deps = [
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_library(
    name = "spec",
    srcs = ["spec.py"],
    deps = [
        ":prior_distribution",
        "//ads/lift/mmm/modeling/research/contr_priors:constants",
        "//third_party/py/numpy",
    ],
)
