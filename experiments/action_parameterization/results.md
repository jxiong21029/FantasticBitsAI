# The Action Parameterization Experiment

The goal of this experiment was to determine the optimal way to parameterize the action
space, specifically regarding the directional target head. Three options were explored:
1. Euclidean (baseline): Two independent normal distributions for x and y, respectively.
The neural network outputs four values: two means and two standard deviations.
2. Normalized Euclidean: Similar to Euclidean, but the mean is normalized to a unit
   vector while the standard deviation is kept as before.
3. Von Mises: The neural network outputs three values. The first two, x and y, determine
the location parameter of the distribution as atan2(y, x), while the last determines the
distribution's concentration.

The three approaches were compared using evaluation performance after behavioral
cloning as a proxy objective.

It was found that the Von Mises distribution has numerical stability issues with small
concentrations (less than 10^-3.5). However, if this issue is addressed (either with
clipping or by adding a small constant e.g. 1e-3 to the concentration), then von Mises
and Euclidean seem to have similar performance, while normalized Euclidean lags behind.

Written on: October 8th, 2022.
