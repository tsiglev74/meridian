# About Meridian

Marketing mix modeling (MMM) is a statistical analysis technique that measures
the impact of marketing campaigns and activities to guide budget planning
decisions and improve overall media effectiveness. MMM uses aggregated data to
measure impact across marketing channels and account for non-marketing factors
that impact sales and other key performance indicators (KPIs). MMM is
privacy-safe and does not use any cookie or user-level information.

Meridian is an MMM framework that enables advertisers to set up and run their
own in-house models. Meridian helps you answer key questions such as:

*   How did the marketing channels drive my sales (or other KPI)?
*   What was my marketing return on investment (ROI)?
*   How do I optimize my marketing budget allocation for the future?

Meridian is a highly customizable modeling framework that is based on
[Bayesian causal inference](https://developers.google.com/meridian/docs/basics/bayesian-inference).
It is capable of handling large scale geo-level data, which is encouraged if
available, but it can also be used for national-level modeling. Meridian
provides clear insights and visualizations to inform business decisions around
marketing budget and planning. Additionally, Meridian provides methodologies to
support calibration of MMM with experiments and other prior information, and to
optimize target ad frequency by utilizing reach and frequency data.

If you are using LightweightMMM, see the
[migration guide](https://developers.google.com/meridian/docs/migrate) to help
you understand the differences between these MMM projects.

# Install Meridian

Run the following command to automatically install the most recent version from
GitHub. Note that Python 3.10 or higher is required.

```sh
$ git clone https://github.com/google/meridian.git
$ pip install .
```

## How to use the Meridian library

To get started with Meridian, you can run the code programmatically using sample
data with the [Getting Started Colab][3].

The Meridian model uses a holistic MCMC sampling approach called
[No U Turn Sampler (NUTS)](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/NoUTurnSampler)
which can be compute intensive. To help with this, GPU support has been
developed across the library (out-of-the-box) using tensors. We recommend
running your Meridian model on GPUs to get real time optimization results and
significantly reduce training time.

# Meridian Documentation & Tutorials

The following documentation, colab, and video resources will help you get
started quickly with using Meridian:

| Resource                    | Description                                    |
| --------------------------- | ---------------------------------------------- |
| [Meridian documentation][1] | Main landing page for Meridian documentation.  |
| [Meridian basics][2]        | Learn about Meridian features, methodologies, and the model math. |
| [Getting started colab][3]  | Install and quickly learn how to use Meridian with this colab tutorial using sample data. |
| [User guide][4]             | A detailed walk-through of how to use Meridian and generating visualizations using your own data. |
| [Advanced modeling considerations][5]    | Advanced modeling guidance for model refinement and edge cases. |
| [Model debugging and troubleshooting][6] | Guidance for coding errors, model debugging, and troubleshooting. |
| [Migrate from LMMM][7]      | Learn about the differences between Meridian and LightweightMMM as you consider migrating. |
| [API Reference][8]          | API reference documentation for the Meridian package. |
| [Reference list][9]         | White papers and other referenced material.    |

[1]: https://developers.google.com/meridian
[2]: https://developers.google.com/meridian/docs/basics/about-the-project
[3]: https://developers.google.com/meridian/notebook/meridian-getting-started
[4]: https://developers.google.com/meridian/docs/user-guide/overview
[5]: https://developers.google.com/meridian/docs/advanced-modeling/model-fit
[6]: https://developers.google.com/meridian/docs/model-debugging/model-debugging
[7]: https://developers.google.com/meridian/docs/migrate
[8]: https://developers.google.com/meridian/docs/api
[9]: https://developers.google.com/meridian/docs/reference-list

## Support

**Questions about methodology**: Please see the [Modeling](https://developers.google.com/meridian/docs/basics/about-the-project) tab and [Advanced considerations](https://developers.google.com/meridian/docs/advanced-modeling/control-variables) section in the technical documentation.

**Issues installing or using Meridian**: Feel free to post questions in the
[Discussions](https://github.com/google/meridian/discussions) or [Issues](https://github.com/google/meridian/issues) tabs of the Meridian GitHub repository. The Meridian team responds to
these questions weekly in batches, so please be patient and don't reach out
directly to your Google Account teams.

**Bug reports**: Please post bug reports to the [Issues](https://github.com/google/meridian/issues)
tab of the Meridian GitHub repository. We also encourage the community to share
tips and advice with each other on the [Issues](https://github.com/google/meridian/issues)
tab. When our team addresses or resolves a new bug, we will notify you through
the comments on the issue.

**Feature requests**: Please post these to the [Discussions](https://github.com/google/meridian/discussions)
tab of the Meridian GitHub repository. We have an internal roadmap for Meridian
development, but would love your inputs for new feature requests so that we can
prioritize them based on the roadmap.

**Pull requests**: These are appreciated but are very difficult for us to merge
because the code in this repository is linked to Google internal systems and has
to pass internal review. If you submit a pull request and we believe that we can
incorporate a change in the base code, we will reach out to you directly about
this.

## Citing Meridian

To cite this repository:

```
@software{meridian_github,
  author = {Google Meridian Marketing Mix Modeling Team},
  title = {Meridian: Marketing Mix Modeling},
  url = {https://github.com/google/meridian},
  version = {0.1.0},
  year = {2024},
}
```
