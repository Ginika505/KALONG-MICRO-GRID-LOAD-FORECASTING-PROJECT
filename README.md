# KALONG-MICRO-GRID-LOAD-FORECASTING-PROJECT
Accurate short-term load forecasting is critical for efficient operation of standalone solar microgrids. This paper proposes a hybrid deep learning framework integrating Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN), Particle Swarm Optimization, and Gated Recurrent Units for one-step-ahead hourly load forecasting.

Methodology


1. Project Overview

This repository presents MATLAB implementations of deep learning and hybrid optimisation-based models for electricity load forecasting in the Kalong standalone solar microgrid.

The research investigates the application of recurrent neural networks and decomposition-based hybrid learning techniques to improve short-term electricity demand forecasting. The developed models are intended to support demand estimation, energy management, and the reliable planning and operation of renewable energy-based mini-grids.

The study evaluates four forecasting architectures: Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Bidirectional Long Short-Term Memory (BiLSTM), and a hybrid Particle Swarm Optimisation–Complete Ensemble Empirical Mode Decomposition with Adaptive Noise–GRU (PSO-CEEMDAN-GRU) model.

2. Research Objectives

The main objectives of this research are to:

Develop deep learning models for hourly electricity load forecasting.

Investigate the ability of recurrent neural networks to capture temporal dependencies in electricity demand.

Evaluate the predictive performance of LSTM, GRU, and BiLSTM architectures.

Develop a hybrid PSO-CEEMDAN-GRU forecasting framework integrating signal decomposition, metaheuristic optimisation, and recurrent neural networks.

Compare the forecasting accuracy and generalisation performance of the developed models.

Demonstrate the relevance of AI-enhanced load forecasting for standalone renewable energy systems and rural electricity access.

3. Study Area and Dataset

Study: Kalong standalone solar microgrid.

The dataset contains hourly electricity load observations and associated meteorological and temporal predictors.

Dataset characteristic

Description

Study application

Standalone solar microgrid

Temporal resolution

Hourly

Historical data period

1 January 2023 – 30 June 2025

Forecasting period

July – December 2025

Historical observations

21,720 hourly records

Meteorological inputs

NASA POWER-derived variables

Temporal features

Lagged demand and time-dependent predictors

The modelling workflow incorporates historical electricity demand, meteorological variables, lagged load observations, and additional engineered predictors.

Data availability: Dataset access and any applicable restrictions should be documented in the data/ directory. Raw data are not redistributed unless permission and licensing conditions allow it.

4. Forecasting Models

Four forecasting architectures are implemented in this repository.

4.1 Long Short-Term Memory (LSTM)

An LSTM recurrent neural network designed to learn long-term temporal dependencies in electricity demand time series.

4.2 Gated Recurrent Unit (GRU)

A gated recurrent architecture that captures sequential demand patterns with a comparatively simplified recurrent structure.

4.3 Bidirectional Long Short-Term Memory (BiLSTM)

A bidirectional recurrent architecture that learns temporal representations through forward and backward processing of input sequences.

4.4 PSO-CEEMDAN-GRU

A hybrid forecasting framework integrating:

CEEMDAN: Decomposes the original electricity demand signal into intrinsic mode functions to capture different temporal characteristics.

PSO: Particle Swarm Optimisation for model hyperparameter optimisation.

GRU: Recurrent neural network for learning temporal patterns and forecasting the decomposed demand components.

The hybrid architecture combines signal decomposition, metaheuristic optimisation, and deep learning to improve forecasting performance.

5. Modelling Workflow

The general research workflow comprises:

Electricity load data acquisition and preparation.

Meteorological data integration and feature engineering.

Time-series preprocessing and lag feature generation.

Chronological training and testing data partitioning.

Development and training of LSTM, GRU, and BiLSTM models.

CEEMDAN signal decomposition.

PSO-based hyperparameter optimisation.

Development and evaluation of the hybrid PSO-CEEMDAN-GRU model.

Comparative forecasting performance assessment.

All forecasting models should follow consistent chronological data partitioning to minimise information leakage and ensure a fair comparison.

6. Model Performance Evaluation

The forecasting models are evaluated using statistical performance indicators, including:

Metric

Purpose

R²

Measures the proportion of variance explained by the model

RMSE

Quantifies the magnitude of squared prediction errors

MAE

Measures average absolute prediction errors

MAPE

Expresses average absolute percentage prediction errors

These metrics provide complementary assessments of forecasting accuracy and model generalisation.

Cite this publication as:

G. P. Okoroafor, O. O. Oladapo, M. D. Ganggayah, P. W. C. Ho, and C. R. Sarimuthu, “ A Hybrid PSO-CEEMDAN-GRU Framework for Short-Term Load Forecasting in Standalone Solar Microgrids: A Case Study in Nigeria.” IET Renewable Power Generation 20, no. 1 (2026): e70377. https://doi.org/10.1049/rpg2.70377
