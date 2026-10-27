"""
PROJECT 1: DEMAND PREDICTION

Real-world project: Predict coffee shop demand!
 
Step 1: Generate Dataset

Problem:
  Coffee shop wants to predict daily demand
  Based on: day of week, temperature, marketing spend, etc.
 
Solution:
  - Create synthetic dataset
  - Train neural network
  - Predict future demand

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Real-world scenario:
  A coffee shop wants to optimize inventory
  They need to predict daily customer demand
  
Factors affecting demand:
  - Day of week (weekdays vs weekends)
  - Temperature (hot coffee ↑ in cold, ↓ in heat)
  - Marketing spend (more ads = more customers)
  - Whether it's holiday or not
  - Average customer spending
  
Input features: 6
  1. Day of week (0-6, 0=Monday)
  2. Temperature (Celsius, 0-40)
  3. Marketing spend (dollars, 0-1000)
  4. Is holiday? (0/1)
  5. Competitor nearby? (0/1)
  6. Store experience (1-5 stars)
 
Output target: 1
  - Daily revenue (dollars, 500-5000)
"""

# genearating Data 

np.random.seed(42)

n_samples = 500
print(f" Generating {n_samples} days of data...")

# Features
day_of_week = np.random.randint(0, 7, n_samples)  # 0-6
temperature = np.random.uniform(5, 35, n_samples)  # 5-35°C
marketing_spend = np.random.uniform(0, 1000, n_samples)  # $0-1000
is_holiday = np.random.randint(0, 2, n_samples)  # 0 or 1
competitor = np.random.randint(0, 2, n_samples)  # 0 or 1
experience = np.random.uniform(1, 5, n_samples)  # 1-5 stars


# Target (revenue ) with realistic 

revenue = 2000 

# day of week effect (weekends better)
revenue = revenue + np.where(day_of_week >= 5, 500, 0)

# Temperature effect 
revenue = revenue - np.abs(temperature - 15) * 30

# Market effect 
revenue = revenue + marketing_spend * 0.8

# holiday boost 

evenue = revenue + is_holiday * 300
 
# Competitor effect (bad if nearby)
revenue = revenue - competitor * 400
 
# Experience rating effect
revenue = revenue + experience * 200
 
# Add random noise
revenue = revenue + np.random.normal(0, 150, n_samples)
 
# Ensure revenue is reasonable
revenue = np.clip(revenue, 500, 5000)

print("Feature ranges:")
print(f"  Day of week: {day_of_week.min()}-{day_of_week.max()}")
print(f"  Temperature: {temperature.min():.1f}-{temperature.max():.1f}°C")
print(f"  Marketing: ${marketing_spend.min():.0f}-${marketing_spend.max():.0f}")
print(f"  Is holiday: {is_holiday.min()}-{is_holiday.max()}")
print(f"  Competitor: {competitor.min()}-{competitor.max()}")
print(f"  Experience: {experience.min():.1f}-{experience.max():.1f}")
print(f"\nTarget range:")
print(f"  Revenue: ${revenue.min():.0f}-${revenue.max():.0f}")

data = pd.DataFrame({
    'day_of_week': day_of_week,
    'temperature': temperature,
    'marketing_spend': marketing_spend,
    'is_holiday': is_holiday,
    'competitor': competitor,
    'experience': experience,
    'revenue': revenue
})

print(f"\nDataset shape: {data.shape}")
print("\nFirst 5 samples:")
print(data.head())
 
print("\nDataset statistics:")
print(data.describe())

print("\n3. CORRELATION WITH REVENUE")
print("-"*70)
 
correlations = data.corr()['revenue'].drop('revenue').sort_values(ascending=False)
print("\nFeature correlations with revenue:")
for feature, corr in correlations.items():
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    direction = "positive" if corr > 0 else "negative"
    print(f"  {feature:18s}: {corr:+.3f} ({strength} {direction})")
 
 
print("\n4. VISUALIZING DATA RELATIONSHIPS")
print("-"*70)
 
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
 
# Temperature vs Revenue
ax = axes[0, 0]
ax.scatter(data['temperature'], data['revenue'], alpha=0.5, s=30)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Revenue ($)')
ax.set_title('Temperature vs Revenue', fontweight='bold')
ax.grid(True, alpha=0.3)
 
# Marketing vs Revenue
ax = axes[0, 1]
ax.scatter(data['marketing_spend'], data['revenue'], alpha=0.5, s=30, color='orange')
ax.set_xlabel('Marketing Spend ($)')
ax.set_ylabel('Revenue ($)')
ax.set_title('Marketing vs Revenue', fontweight='bold')
ax.grid(True, alpha=0.3)
 
# Day of week vs Revenue
ax = axes[0, 2]
for day in range(7):
    day_data = data[data['day_of_week'] == day]['revenue']
    ax.boxplot([day_data], positions=[day])
ax.set_xlabel('Day of Week (0=Mon, 6=Sun)')
ax.set_ylabel('Revenue ($)')
ax.set_title('Day of Week vs Revenue', fontweight='bold')
ax.grid(True, alpha=0.3)
 
# Holiday effect
ax = axes[1, 0]
holiday_revenue = data[data['is_holiday'] == 1]['revenue']
no_holiday_revenue = data[data['is_holiday'] == 0]['revenue']
ax.boxplot([no_holiday_revenue, holiday_revenue], tick_labels=['No Holiday', 'Holiday'])
ax.set_ylabel('Revenue ($)')
ax.set_title('Holiday Effect', fontweight='bold')
ax.grid(True, alpha=0.3)
 
# Competitor effect
ax = axes[1, 1]
no_comp = data[data['competitor'] == 0]['revenue']
comp = data[data['competitor'] == 1]['revenue']
ax.boxplot([no_comp, comp], tick_labels=['No Competitor', 'Competitor'])
ax.set_ylabel('Revenue ($)')
ax.set_title('Competitor Effect', fontweight='bold')
ax.grid(True, alpha=0.3)
 
# Revenue distribution
ax = axes[1, 2]
ax.hist(data['revenue'], bins=30, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Revenue ($)')
ax.set_ylabel('Frequency')
ax.set_title('Revenue Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('demand_data_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: demand_data_analysis.png")
plt.show()
 
 
print("\n5. SAVING DATASET")
print("-"*70)
 
# Save to CSV
data.to_csv('coffee_demand_data.csv', index=False)
print("✓ Saved: coffee_demand_data.csv")
 
# Also save as numpy for later use
np.save('coffee_demand_X.npy', data.drop('revenue', axis=1).values)
np.save('coffee_demand_y.npy', data['revenue'].values)
print("✓ Saved: coffee_demand_X.npy, coffee_demand_y.npy")
 
 
print("\n6. DATA SUMMARY")
print("-"*70)
 
print(f"""
Dataset created!
 
Total samples: {n_samples}
Total features: 6
 
Feature columns:
  1. day_of_week: 0-6 (Monday-Sunday)
  2. temperature: 5-35°C
  3. marketing_spend: $0-1000
  4. is_holiday: 0 (no) or 1 (yes)
  5. competitor: 0 (no) or 1 (yes)
  6. experience: 1-5 stars
 
Target column:
  revenue: ${revenue.min():.0f}-${revenue.max():.0f}
 
Next step: Prepare and normalize data!
""")
 
