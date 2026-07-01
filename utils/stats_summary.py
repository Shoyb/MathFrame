"""
utils/stats_summary.py — Shared descriptive-statistics summary helper.

Factored out of ``cogs/statistics.py`` rather than left inline in the
``/stat summary`` command, because the Phase 3 probability cog's
``/prob sample`` (draw n samples from a distribution, then summarize them)
needs the exact same computation and must not maintain a second copy of it.

The single entry point is :func:`compute_summary`.
"""

from __future__ import annotations

import statistics as _stats
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class StatsSummary:
    """All the descriptive statistics for one data set, computed once."""

    n: int
    total: float
    sum_sq: float
    mean: float
    median: float
    modes: list[float]
    sample_variance: float
    population_variance: float
    sample_stdev: float
    population_stdev: float
    minimum: float
    maximum: float
    data_range: float
    q1: float
    q3: float
    iqr: float
    skewness: float | None  # None when undefined (zero variance)
    kurtosis: float | None  # None when undefined (zero variance)


def compute_summary(nums: list[float]) -> StatsSummary:
    """
    Compute a full descriptive-statistics summary for *nums* in one pass.

    Raises
    ------
    ValueError
        If *nums* has fewer than 2 points — sample variance/stdev (and by
        extension this whole summary) are undefined for n < 2, matching the
        existing ``/stat stdev``/``/stat variance`` commands' requirement.
    """
    n = len(nums)
    if n < 2:
        raise ValueError(
            "A statistics summary requires at least 2 data points "
            "(sample variance/stdev are undefined for n < 2)."
        )

    arr = np.array(nums, dtype=float)

    total = float(np.sum(arr))
    sum_sq = float(np.sum(arr**2))
    mean = float(np.mean(arr))
    median = float(_stats.median(nums))
    modes = _stats.multimode(nums)

    sample_variance = float(_stats.variance(nums))
    population_variance = float(_stats.pvariance(nums))
    sample_stdev = float(_stats.stdev(nums))
    population_stdev = float(_stats.pstdev(nums))

    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    data_range = maximum - minimum

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    # skew/kurtosis are 0/0 (NaN, with a noisy RuntimeWarning) for
    # zero-variance data — report as explicitly undefined instead of
    # letting NaN leak into a Discord embed.
    if population_variance == 0:
        skewness = None
        kurtosis = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            skewness = float(scipy_stats.skew(arr))
            kurtosis = float(scipy_stats.kurtosis(arr))  # excess kurtosis (normal = 0)

    return StatsSummary(
        n=n,
        total=total,
        sum_sq=sum_sq,
        mean=mean,
        median=median,
        modes=modes,
        sample_variance=sample_variance,
        population_variance=population_variance,
        sample_stdev=sample_stdev,
        population_stdev=population_stdev,
        minimum=minimum,
        maximum=maximum,
        data_range=data_range,
        q1=q1,
        q3=q3,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
    )
