import numpy as np
import scipy.stats


class StatDistance:
    """ Base class for computing distances between two empirical samples. """
    def __init__(self, gap_list_1, gap_list_2):
        self.gaps_1 = gap_list_1
        self.gaps_2 = gap_list_2

    def _get_pmfs(self):
        """ Convert two lists of samples to empirical pmf """
        gap_arr_1 = np.array(self.gaps_1)
        gap_arr_2 = np.array(self.gaps_2)

        gap_hist_1 = np.bincount(gap_arr_1)
        gap_hist_2 = np.bincount(gap_arr_2)

        gap_pmf1 = gap_hist_1 / np.sum(gap_hist_1)
        gap_pmf2 = gap_hist_2 / np.sum(gap_hist_2)

        if len(gap_pmf1) > len(gap_pmf2):
            gap_pmf2 = np.append(gap_pmf2,
                                 np.zeros(len(gap_pmf1) - len(gap_pmf2)))
        else:
            gap_pmf1 = np.append(gap_pmf1,
                                 np.zeros(len(gap_pmf2) - len(gap_pmf1)))
        return gap_pmf1, gap_pmf2

    def _get_cmfs(self):
        """ Distances computed on two empirical CDFs compute an integral
        over the combined measure of the two samples. This code was adapted
        from the scipy.stats repository.
        Ref: https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py#L3809
        """
        n1 = len(self.gaps_1)
        n2 = len(self.gaps_2)
        data1 = np.sort(self.gaps_1)
        data2 = np.sort(self.gaps_2)
        data_all = np.concatenate([data1, data2])

        gap_cmf1 = np.searchsorted(data1, data_all, side='right') / (1. * n1)
        gap_cmf2 = np.searchsorted(data2, data_all, side='right') / (1. * n2)
        return gap_cmf1, gap_cmf2

    def score(self):
        raise NotImplementedError


class KolmogorovSmirnoff(StatDistance):
    """ Kolmogorov-Smirnoff distance (sup-norm over empirical CDFs) """
    def __init__(self, gap_list_1, gap_list_2):
        super().__init__(gap_list_1, gap_list_2)

    def score(self):
        return scipy.stats.ks_2samp(self.gaps_1, self.gaps_2)[0]


class ChiSquare(StatDistance):
    """ Chi-square distance variant pooled denominator variant """
    def __init__(self, gap_list_1, gap_list_2):
        super().__init__(gap_list_1, gap_list_2)

    def score(self):
        pmf1, pmf2 = self._get_pmfs()
        return np.sum(np.power(pmf1 - pmf2, 2) / (pmf1 + pmf2))


class HistogramIntersection(StatDistance):
    """ Histogram intersection kernel (1 - sum min(p_1, p_2)) """
    def __init__(self, gap_list_1, gap_list_2):
        super().__init__(gap_list_1, gap_list_2)

    def score(self):
        pmf1, pmf2 = self._get_pmfs()
        return 1 - np.sum(np.minimum(pmf1, pmf2))


class CramerVonMises(StatDistance):
    """ Cramer-von-Mises distance (sum (f_1 - f_2)^2 ) """
    def __init__(self, gap_list_1, gap_list_2):
        super().__init__(gap_list_1, gap_list_2)

    def score(self):
        cmf1, cmf2 = self._get_cmfs()
        return np.sum(np.power(cmf1 - cmf2, 2))


class Hellinger(StatDistance):
    """ Hellinger distance (based on sum of element-wise geometric means) """
    def __init__(self, gap_list_1, gap_list_2):
        super().__init__(gap_list_1, gap_list_2)

    def score(self):
        pmf1, pmf2 = self._get_pmfs()
        bhatt_coef = np.sum(np.sqrt(pmf1 * pmf2))
        return np.sqrt(1 - bhatt_coef)

distances = locals()
