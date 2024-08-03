# Property Data Analysis and Visualization in Dubai

This repository contains code for analyzing and visualizing property data in Dubai using Python. It includes exploratory data analysis (EDA), descriptive statistics, visualizations, and geographic mapping using libraries such as Pandas, NumPy, Seaborn, Matplotlib, Folium, and Geopy.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project explores a dataset of cleaned property data in Dubai. It aims to provide insights into property prices, size, number of bedrooms and bathrooms, neighborhood impact, and geographic distribution using various statistical and visual techniques.

## Installation

To run the code in this repository locally, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/property-analysis-dubai.git
    cd property-analysis-dubai
    ```

2. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

   Make sure to replace `requirements.txt` with the actual file containing necessary dependencies.

## Usage

1. **Load and Explore Data:**
   - The cleaned property data is loaded from `cleaned_property_data.csv`.
   - Various analyses are performed to understand property prices, sizes, and other attributes.

2. **Visualization:**
   - Visualize property price distributions using histograms and descriptive statistics.
   - Explore relationships between property attributes such as price vs. size, price vs. number of bedrooms, and more.
   - Generate geographic maps of Dubai using Folium to visualize property prices by location.

## Features

- **Data Loading and Preprocessing:** Utilizes Pandas for loading and preprocessing the property dataset.
- **Descriptive Statistics:** Computes descriptive statistics such as skewness, kurtosis, and correlation matrices.
- **Visualization:** Uses Seaborn and Matplotlib for visualizing distributions, scatter plots, box plots, and heatmaps.
- **Geographic Mapping:** Integrates Folium for creating interactive maps with property price heatmaps and markers.

## Visualization

### Examples of Visualizations:

- **Distribution of Property Prices:**
  ![Distribution of Property Prices](images/property_price_distribution.png)

- **Scatter Plot: Property Size vs. Price**
  ![Scatter Plot: Property Size vs. Price](images/property_size_vs_price.png)

- **Geographic Heatmap of Property Prices in Dubai**
  ![Geographic Heatmap of Property Prices](images/property_price_heatmap.png)

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or additional analyses, please fork this repository, make your changes, and submit a pull request. Ensure your code adheres to the existing style and includes relevant documentation.
