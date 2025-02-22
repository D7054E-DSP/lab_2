import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd


def generate_random_data(size=50, decimal_places=4):
    """Generate random numbers between 0 and 1, rounded to a specified number of decimal places."""
    data = np.random.uniform(0, 1, size)
    return np.round(data, decimal_places)


def calculate_statistics(data):
    """Calculate and return mean, standard deviation, first quartile, third quartile, and median."""
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    first_quartile = np.percentile(data, 25)
    third_quartile = np.percentile(data, 75)
    median = np.median(data)
    
    return mean, std_dev, first_quartile, third_quartile, median

def plot_histogram(data, bins, title):
    """Plot a histogram for the Emperical Data."""
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7, density=True)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Fit and plot normal distribution curve
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='red', linewidth=2)
    plt.show()

def plot_boxplot(data):
    """Construct a box plot of the data."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data, color='lightblue')
    plt.title("Box Plot of Data")
    plt.xlabel("Values")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def detect_outliers(data):
    """Detect potential outliers using the IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers, q1, q3, iqr

# Generate data
data = generate_random_data()

# Calculate statistics
mean, std_dev, q1, q3, median = calculate_statistics(data)

# Detect outliers
outliers, q1, q3, iqr = detect_outliers(data)

# Print results
print("Computed Values for Random Empirical Data:")
print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"First Quartile: {q1:.4f}")
print(f"Third Quartile: {q3:.4f}")
print(f"Median: {median:.4f}\n\n")
print(f"Outliers: {outliers}")
print(f"IQR Width: {iqr:.4f}")

# Plot histograms
plot_histogram(data, bins=8, title="Histogram for Empirical Data with 8 Bins")
plot_histogram(data, bins=5, title="Histogram for Empirical Data with 5 Bins")
# Plot boxplot
plot_boxplot(data)

# Race Lap Times Data
race_lap_times = np.array([
    [135, 130, 131, 132, 130, 131, 133],
    [134, 131, 131, 129, 128, 128, 129],
    [129, 127, 127, 127, 127, 127, 129],
    [125, 125, 126, 126, 124, 125, 125],
    [133, 130, 132, 132, 131, 130, 130],
    [130, 130, 130, 130, 129, 129, 129],
    [132, 131, 133, 133, 134, 134, 129],
    [127, 128, 127, 127, 126, 126, 126],
    [130, 130, 127, 127, 132, 132, 132],
    [131, 131, 131, 131, 132, 132, 129],
    [132, 131, 130, 130, 131, 131, 129],
    [134, 130, 130, 130, 131, 130, 130],
    [133, 127, 128, 128, 128, 128, 128],
    [132, 131, 127, 128, 128, 128, 128],
    [136, 129, 129, 129, 129, 129, 127],
    [129, 129, 129, 129, 129, 129, 129],
    [132, 129, 129, 131, 133, 133, 132],
    [129, 129, 129, 132, 133, 133, 132],
    [130, 129, 129, 129, 129, 129, 129],
    [131, 128, 128, 128, 129, 130, 130]
])

# Stratified sampling: Select 6 lap times from each race
np.random.seed(42)
stratified_sample_laps = np.array([np.random.choice(laps[1:], 6, replace=False) for laps in race_lap_times])

# Flatten the sampled data
sampled_laps = stratified_sample_laps.flatten()

# Compute statistics for the sampled data
mean_sample, std_sample, q1_sample, q3_sample, median_sample = calculate_statistics(sampled_laps)

# Compute IQR
iqr_sample = q3_sample - q1_sample

# Compute percentiles
p15_sample = np.percentile(sampled_laps, 15)
p85_sample = np.percentile(sampled_laps, 85)

# Compute empirical probability of lap time > 130
prob_gt_130 = np.mean(sampled_laps > 130)

# Print computed values
print("Computed Values for Race Lap Times:")
print(f"Sample Mean: {mean_sample:.2f}")
print(f"Sample Standard Deviation: {std_sample:.2f}")
print(f"IQR: {iqr_sample:.2f} (from {q1_sample:.2f} to {q3_sample:.2f})")
print(f"15th Percentile: {p15_sample:.2f}")
print(f"85th Percentile: {p85_sample:.2f}")
print(f"Median: {median_sample:.2f}")
print(f"Empirical Probability (Lap Time > 130): {prob_gt_130:.4f}\n\n")

# Theoretical normal distribution
mu, sigma = mean_sample, std_sample
p15_theory = stats.norm.ppf(0.15, mu, sigma)
p85_theory = stats.norm.ppf(0.85, mu, sigma)
prob_gt_130_theory = 1 - stats.norm.cdf(130, mu, sigma)

# Print theoretical values
print("Theoretical Deistribution for Race Lap Times:")
print(f"Theoretical IQR: {q1_sample:.2f} to {q3_sample:.2f}")
print(f"Theoretical 15th Percentile: {p15_theory:.2f}")
print(f"Theoretical 85th Percentile: {p85_theory:.2f}")
print(f"Theoretical Probability (Lap Time > 130): {prob_gt_130_theory:.4f} \n")

# Plot histogram for stratified sample data
plot_histogram(sampled_laps, bins=6, title="Histogram of Sampled Lap Times")



# Central Limit Theorem Demonstration
cookie_data = np.array([
    1, 5, 2, 5, 6, 1, 2, 6, 5, 2, 5, 1, 1, 3, 2, 2, 2, 4, 6, 1, 6, 5, 2, 5, 1, 6, 4, 1, 6, 2,
    3, 4, 5, 6, 6, 1, 1, 2, 1, 6, 1, 6, 2, 6, 2, 2, 2, 11, 5, 5, 4, 6, 5, 1, 1, 2, 4, 3, 6, 5
])

# Calculate population mean and standard deviation
pop_mean = np.mean(cookie_data)
pop_std = np.std(cookie_data, ddof=1)
print(f"Population Mean (μ): {pop_mean:.2f}")
print(f"Population Standard Deviation (σ): {pop_std:.2f}")

# Sampling function
def sample_means(data, sample_size, num_samples):
    sample_means = [np.mean(np.random.choice(data, sample_size, replace=False)) for _ in range(num_samples)]
    return sample_means

# Generate samples and compute means
sample_means_n5 = sample_means(cookie_data, 5, 5)
sample_means_n10 = sample_means(cookie_data, 10, 5)

# Compute statistics for sample means
mean_n5 = np.mean(sample_means_n5)
std_n5 = np.std(sample_means_n5, ddof=1)
mean_n10 = np.mean(sample_means_n10)
std_n10 = np.std(sample_means_n10, ddof=1)

print(f"\nSample Means (n=5): {sample_means_n5}")
print(f"Mean of Sample Means (N=5): {mean_n5:.2f}")
print(f"Standard Deviation of Sample Means (N=5): {std_n5:.2f}")

print(f"\nSample Means (n=10): {sample_means_n10}")
print(f"Mean of Sample Means (N=10): {mean_n10:.2f}")
print(f"Standard Deviation of Sample Means (N=10): {std_n10:.2f}")

# Plot histogram of the original population
def plot_cookie_histogram_kde(data):
    plt.hist(data, bins=range(int(min(data)), int(max(data)) + 2), edgecolor='black', alpha=0.7, density=True, label="Histogram")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cookie Data with KDE Curve")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # KDE (Kernel Density Estimate) for better fitting curve
    sns.kdeplot(data, color='red', linewidth=2, label="KDE Curve")
    
    plt.legend()
    plt.show()

plot_cookie_histogram_kde(cookie_data)

# 4. Exploring the Melbourne Real-Estate Dataset
# Load dataset
melbourne_data = pd.read_csv("melb_data.csv")

# Analyzing correlation between distance from CBD and price
plt.figure(figsize=(8, 4))
sns.scatterplot(x=melbourne_data['Distance'], y=melbourne_data['Price'], alpha=0.5)
plt.title("Distance from CBD vs Price")
plt.xlabel("Distance from CBD (km)")
plt.ylabel("Price ($AUD)")
plt.show()

# Distribution of property sizes
plt.figure(figsize=(8, 4))
sns.histplot(melbourne_data['BuildingArea'].dropna(), bins=30, kde=True, color='red')
plt.title("Distribution of Property Sizes")
plt.xlabel("Building Area (sqm)")
plt.ylabel("Frequency")
plt.show()

# Properties by suburb count
suburb_counts = melbourne_data['Suburb'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=suburb_counts.index, y=suburb_counts.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Top 10 Suburbs with Most Properties Listed")
plt.xlabel("Suburb")
plt.ylabel("Number of Listings")
plt.show()



################ PART-B 


# Load dataset
melbourne_data = pd.read_csv("melb_data.csv")

# Filter for a specific suburb: Caulfield North
melbourne_data = melbourne_data[melbourne_data['Suburb'] == 'Caulfield North']
melbourne_data = melbourne_data.dropna(subset=['Bedroom2', 'Price'])

# Selecting features
X = melbourne_data[['Bedroom2']].values  # Feature: Number of bedrooms
y = melbourne_data['Price'].values  # Target: Price

# Normalize data
X = (X - np.mean(X)) / np.std(X)
X = X.flatten()  # Convert from (n,1) to (n,)
y = (y - np.mean(y)) / np.std(y)

# Initialize parameters
m = 0  # Slope
c = 0  # Intercept
learning_rates = [0.001, 0.01, 0.1]  # Different learning rates
iterations = 1000

# Function to compute cost
def compute_cost(X, y, m, c):
    error = (m * X + c) - y
    return np.mean(error ** 2) / 2

# Gradient Descent Algorithm
for lr in learning_rates:
    m_current, c_current = 0, 0
    cost_values = []
    for _ in range(iterations):
        y_pred = m_current * X + c_current
        error = y_pred - y
        m_current -= lr * (2 / len(X)) * np.sum(error * X)
        c_current -= lr * (2 / len(X)) * np.sum(error)
        cost_values.append(compute_cost(X, y, m_current, c_current))
    
    # Plot cost function trend
    plt.plot(range(iterations), cost_values, label=f'LR={lr}')
    if lr == 0.01:
        m, c = m_current, c_current  # Save best parameters

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.title("Cost Function Convergence for Different Learning Rates")
plt.show()

# Plot regression line
plt.figure(figsize=(8, 4))
plt.scatter(X, y, alpha=0.5, label='Data Points')
plt.plot(X, m * X + c, color='red', label='Regression Line')
plt.xlabel("Number of Bedrooms (Normalized)")
plt.ylabel("Price (Normalized)")
plt.legend()
plt.title("Gradient Descent Regression for Caulfield North")
plt.show()
