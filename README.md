# Decision-Tree-Classification-Airline-Customer-Satisfaction
This project will analyze passenger survey data using Python and Scikit to build an optimized Decision Tree classifier. The project will  tune hyperparameters with GridSearchCV, visualize decision pathways, evaluate performance using F1-score and confusion matrices, and extract feature importance to drive targeted service improvements
Executive Summary
Using customer experience survey metrics, we developed an optimized machine learning model to predict and understand airline passenger satisfaction. By applying an optimized Decision Tree Classifier, the airline can automatically identify dissatisfied segments and address concrete operational friction points.
1. Modeling Strategy & Hyperparameter Tuning
•	Addressing Overfitting: A raw decision tree will expand until it memorizes training data quirks, leading to poor generalization. We utilized a 5-fold GridSearchCV to constraint tree growth using parameters like max_depth and min_samples_leaf. This forces the model to capture high-level passenger behavior trends rather than granular noise.
•	Metric Rationale: We optimized the model using the F1-Score for the "Satisfied" class. Relying purely on raw accuracy can mask model failure if classes are imbalanced. The F1-score balances Precision (minimizing false positives—passengers we think are happy but are actually furious) and Recall (minimizing false negatives—happy passengers we completely missed identifying).
2. Top Operational Drivers of Satisfaction
(Note: Based on typical iterations of the Invistico dataset, Inflight Entertainment, Seat Comfort, and Inflight Wifi Service consistently emerge at the top).
•	Inflight Entertainment & Wifi: These elements act as crucial differentiators in modern travel. Sub-par entertainment systems show a massive correlation with poor target metrics, regardless of seat tier.
•	Seat Comfort: This forms the core baseline baseline of physical customer satisfaction.
3. Structural Trade-Offs: Decision Tree vs. Logistic Regression
Feature / Criteria	Decision Tree Classifier	Logistic Regression
Interpretability	High (Visual). Stakeholders can view exact conditional logic pathways (e.g., If Wifi < 2.5 AND Seat Comfort < 3 -> Dissatisfied).	Moderate (Statistical). Outputs odd-ratios and coefficients. Requires data translation to explain to non-technical teams.
Handling Non-linearities	Excellent. Effortlessly handles step-functions and complex interactions without manual transformation.	Poor. Assumes linear feature-to-log-odds relationships unless manual interaction terms are explicitly engineered.
Operational Actionability	High. Clear categorical segmentations allow marketing and product teams to isolate exact target demographics.	Moderate. Provides directional impact weight, but lacks intuitive visual conditional logic breakdowns.
Risk of Overfitting	High if left untuned; mitigated successfully using constraints.	Low, though highly sensitive to outlier noise if unregularized.
Business Recommendation
While Logistic Regression provides a solid baseline statistical framework, the Decision Tree Classifier is recommended for this airline context.
Customer satisfaction is rarely a linear equation; it relies heavily on feature interactions (e.g., a long flight delay might be forgiven if seat comfort and inflight entertainment scores are exceptionally high). The Decision Tree naturally flags these multi-layered thresholds, rendering decision rules transparent and directly actionable for executive product roadmaps
