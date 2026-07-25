After building the NFL spread forecasting model, I got curious about applying similar Bayesian thinking to individual player evaluation. When I had the opportunity to complete a data science assessment with an NFL team, I built a QB WAR Bayesian forecasting model using their internal data. Replicating that framework with publicly available nflverse data and switching the target to total passing EPA became the foundation for this project.

The core is a hierarchical Bayesian model: non-centered parameterization, partial pooling via player intercepts, and player-specific forecast variance estimated from each QB's own year-over-year volatility history. Log(attempts) controls for volume, and the strongest predictors turned out to be epa_per_att (r=0.91) and dakota (r=0.86).

With more time after the assessment, I pushed the project further: an XGBoost baseline with time-series cross-validation for comparison, a Gaussian Process replacing the polynomial aging curve so the model learns the career arc from data, and Student-t likelihoods after finding residual excess kurtosis of 1.93. Both the Bayesian and GP versions independently estimated degrees of freedom near 3, confirming fat tails.

I also extended the framework to RBs using RYOE (Rush Yards Over Expected) as the latent talent estimator. One structural insight from this: total rushing EPA = efficiency × carries, which is multiplicative. A linear model predicting total EPA directly fails here; the correct approach is to model EPA/carry using RYOE, then multiply projected efficiency by projected carries.

Validating against 2024 actuals: the QB model produced r=0.56, 54% coverage at the 50% interval level (nearly ideal), and 71% at 90%. The RB model was nearly unbiased (mean error: -0.2 EPA) with well-calibrated intervals, but near-zero rank correlation. The two largest misses were Saquon Barkley and Derrick Henry, both of whom moved to significantly better situations in 2024. That signal does not exist in public data, and the intervals honestly reflect it.

What I find most interesting about this framework is how transferable it is. The hierarchical structure, the aging curve, the player-specific volatility estimates, none of that is football-specific. What the model actually needs is a reliable statistical estimator of latent talent, something that separates signal from context and luck. In football that is EPA and dakota. In baseball it is FIP or wRC+. In basketball it is adjusted plus-minus or EPM. In soccer it is xG and xA built into a possession-adjusted framework. Once you have that estimator, the forecasting machinery is largely the same: pool across players, learn the aging curve, propagate uncertainty forward.

Interactive report: https://dave-zack3.github.io/football_player_forecasting/
Full code and notebook: https://lnkd.in/eVqBjKNq

#SportsAnalytics #NFL #BayesianStatistics #DataScience #FootballAnalytics
