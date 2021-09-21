import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import geom
from itertools import product
from sklearn.neighbors import KernelDensity

from .distances import distances


def get_gap_lengths(flux_data):
    """Get numpy array of lengths of consecutive gaps."""
    flux_data = pd.Series(flux_data)
    gaps = flux_data.isnull().astype(int)
    observed = flux_data.notnull().astype(int)
    gap_lengths_with_zero = gaps.groupby(observed.cumsum()).sum()
    gap_lengths = gap_lengths_with_zero[gap_lengths_with_zero != 0].to_numpy()

    return gap_lengths


def geom_pmf(p, support):
    """ Return truncated geometric pmf with parameter p and length support """
    pmf = np.array([geom.pmf(x, p) for x in range(support)])
    return pmf


def convex_combine_geom(pmf1, p, alpha):
    """ Convex combination of input pmf with a geometric pmf.
    Args:
        pmf1 (np.array): Input pmf. i.e. pmf1[0] = P(X = 0)
        p (float): parameter for geometric pmf
        alpha (float): weight for pmf1
    Return:
        pmf = alpha * pmf1 + (1 - alpha) * Geom(p)
    """
    g_pmf = geom_pmf(p, len(pmf1))
    cc = alpha * pmf1 + (1 - alpha) * g_pmf
    return cc / np.sum(cc)


def compile_empirical_gap_dist(gap_lengths,
                               outlier_quant=0.99,
                               smooth_tail_start=0.95,
                               verbose=False):
    """ Derive empirical gap length distribution for a flux site.
    Compiles all gap lengths from a time series, discarding outliers.
    This is turned into a rough empirical distribution. Because of few
    observations at the tail, the distribution can be smoothed out using a
    kernel density estimator.
    Args:
        gap_lengths (np.array): empirical gap lengths
        outlier_quant (float): gap length quantile threshold to set as outlier
        smooth_tail_start (float): tail quantile above which gets smoothed
        verbose (bool): print intermediate quantities
    """
    gap_list = gap_lengths[gap_lengths < np.quantile(gap_lengths, outlier_quant)]

    # convert list to normalized distribution
    gap_list = np.array(gap_list)
    gap_hist = np.bincount(gap_list)
    gap_pmf = gap_hist / np.sum(gap_hist)

    # smooth tail starting from specified tail probability threshold
    smooth_start = np.where(np.cumsum(gap_pmf) > smooth_tail_start)[0][1]

    # kernel density estimator
    kd = KernelDensity(bandwidth=5.0, kernel="epanechnikov")
    kd.fit(gap_list[:, None])
    support = np.arange(len(gap_pmf))[:, None]
    log_dens = kd.score_samples(support)    # returns log probs
    gap_pmf_smooth = np.exp(log_dens)

    true_tail_pmf = gap_pmf[smooth_start:]
    smooth_tail_pmf = gap_pmf_smooth[smooth_start:]

    # re-normalize smoothed tail to have same density as true tail
    smooth_tail_pmf *= np.sum(true_tail_pmf) / np.sum(smooth_tail_pmf)

    # replace true tail with smooth tail and renormalize for numerical error
    gap_pmf_final = np.copy(gap_pmf)
    gap_pmf_final[smooth_start:] = smooth_tail_pmf
    gap_pmf_final /= np.sum(gap_pmf_final)

    return gap_pmf_final


def sample_artificial_gaps(flux_data,
                           sampling_pmf,
                           eval_frac=0.1,
                           overlap_retries=20,
                           seed=1000):
    """ Randomly introduce gaps in a time series where the length of gaps
    are i.i.d. samples from a specified probability mass function.
    In order to keep the distribution of gap lengths equal to the sampling
    density, gaps are sampled in a non-overlapping way. This is first done by
    randomly selecting a starting location and a gap length. If an overlap
    occurs, a new starting location is selected. If a valid starting location
    cannot be found within a certain number of trials, the gap length is
    re-selected. This adds a small potential bias toward smaller gaps in return
    for avoiding infinite failure.
    Args:
        flux_data (np.array): time series to mask. Ignores existing gaps.
        sampling_pmf (np.array): Mass function/histogram with support [0, n]
        eval_frac (float): Percentage of non-NaN entries in the time series
            to mask out. This is done by introducing gaps until the threshold
            is reached.
        overlap_retries (int): Number of trials of adding a gap before
            re-selecting gap length.
        seed (int): numpy random seed to fix sampling
    Return:
        masked_series (np.array): time series with gaps represented as NaNs
    """
    np.random.seed(seed)
    observed = np.isfinite(flux_data)
    gap_mask = np.ones(len(flux_data))

    prop_masked = 0.
    while prop_masked < eval_frac:
        # pick random index and gap length
        trials = 0
        rand_idx = np.random.choice(np.where(gap_mask == 1)[0])
        rand_gap = np.random.choice(len(sampling_pmf), p=sampling_pmf)

        # retry if overlaps with previously chosen gaps
        while np.any(gap_mask[rand_idx:rand_idx + rand_gap] == 0):
            rand_idx = np.random.choice(np.where(gap_mask == 1)[0])
            trials += 1
            if trials > overlap_retries:
                rand_gap = np.random.choice(len(sampling_pmf), p=sampling_pmf)

        # gap successfully added
        gap_mask[rand_idx:rand_idx + rand_gap] = 0

        # recompute the total observed percentage masked
        prop_masked = np.sum(observed * (1 - gap_mask)) / np.sum(observed)

    masked_series = np.copy(flux_data)
    masked_series[gap_mask == 0] = np.nan
    return masked_series


def simulate_artificial_gap_samples(dist_fn, flux_data, gap_lengths,
                                    sampling_pmf, n_mc=50, p_add=0.3):
    """
    Monte Carlo approximation of distance statistic comparing gap
    distribution pre- and post- artificial gap sampling.
    
    Args:
        dist_fn (object): Distance function to use
        flux_data (np.array): Flux data from FCH4 column
        gap_lengths (list): list of gap lengths
        sampling_pmf (np.array): distribution to sample new gaps from
        n_mc (int): number of monte carlo samples
        p_add (float): Percent of non-NaN rows to artificially mask. The test
            sets use 0.1 but this is raised to increase the signal
    
    Return:
        list of distance scores
    """
    dist_scores = []

    for i in range(n_mc):
        # Vary the seed to sample different gap distributions
        r_seed = np.random.randint(1, 100000)

        # compute p (+) q
        union_flux_data = sample_artificial_gaps(
            flux_data, sampling_pmf, eval_frac=p_add, seed=r_seed
        )

        # compute distance statistics
        union_gap_lengths = get_gap_lengths(union_flux_data)
        union_gap_lengths = union_gap_lengths[
            union_gap_lengths < np.quantile(union_gap_lengths, 0.95)
        ]

        dist_score = dist_fn(gap_lengths, union_gap_lengths).score()
        dist_scores.append(dist_score)

    return dist_scores 


def learn_gap_dist(
        flux_data,
        dist,
        n_grid,
        n_mc,
        seed
):
    """Approximate the empirical gap length distribution.

    Args:
        flux_data (np.array): Flux data from FCH4 column
        dist (str): Distance measure to use for evaluating the similarity 
                    between the empirical and approximated gap distribution.
                    Options: ['CramerVonMises', 'KolmogorovSmirnoff',
                              'ChiSquare', 'HistogramIntersection',
                              'Hellinger']
        n_grid (int): Width of the grid search per hyperparameter.
                      Must be a positive integer.
        n_mc (int): Number of Monte Carlo interations for estimating
                    the artificial gap distribution.
                    Must be a positive integer.
        seed (int): numpy random seed to fix sampling

    Return a numpy array representing the empirical gap length distribution
    """
    print(" - Estimating artificial gap distribution...")
    np.random.seed(seed)
    gap_lengths = get_gap_lengths(flux_data)

    gap_pmf = compile_empirical_gap_dist(
        gap_lengths,
        outlier_quant=0.99,
        smooth_tail_start=0.95
    )

    gap_lengths = gap_lengths[gap_lengths < np.quantile(gap_lengths, 0.95)]

    alpha_search_space = np.linspace(0.01, 0.5, n_grid)
    p_search_space = np.linspace(
        gap_pmf[1],
        min(0.9, 2.0 * gap_pmf[1]),
        n_grid
    )

    # Get distance function from name
    dist_fn = distances[dist]

    # store best results
    best_pmf = None
    best_dist_score = np.inf

    for (alpha, p) in tqdm(list(product(alpha_search_space, p_search_space))):
        # propose sampling distribution
        sampling_pmf = convex_combine_geom(gap_pmf, p=p, alpha=alpha)

        # score on monte carlo samples
        dist_scores = simulate_artificial_gap_samples(
            dist_fn, flux_data, gap_lengths, sampling_pmf, n_mc=n_mc
        )

        score = np.mean(dist_scores)

        # update best results
        if score < best_dist_score:
            best_dist_score = score
            best_pmf = sampling_pmf

    print(" - Done estimating artificial gap distribution.")
    return best_pmf
