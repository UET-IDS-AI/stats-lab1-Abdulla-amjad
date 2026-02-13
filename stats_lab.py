import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")
    plt.show()
    
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()
    
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()
    
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.array(data)
    return np.sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.array(data)
    n = len(data)
    mean = sample_mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - Q1 (25th percentile)
    - Q3 (75th percentile)

    Quartiles computed using median of lower/upper halves.
    For [5,1,3,2,4] → Q1=2, Q3=4.
    """
    data = sorted(data)
    n = len(data)
    
    minimum = data[0]
    maximum = data[-1]
    median = np.median(data)
    
    # Exclude median when n is odd
    if n % 2 == 1:
        lower_half = data[:n//2]
        upper_half = data[n//2+1:]
    else:
        lower_half = data[:n//2]
        upper_half = data[n//2:]
    
    q1 = np.median(lower_half)
    q3 = np.median(upper_half)
    
    return (minimum, maximum, median, q1, q3)


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    
    n = len(x)
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)
    
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])
