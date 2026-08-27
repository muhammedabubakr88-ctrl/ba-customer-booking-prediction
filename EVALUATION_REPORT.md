# Model Evaluation Report — Customer Booking Prediction

## Objective

Predict whether a customer will complete a flight booking, using pre-booking search and preference data, so British Airways can proactively target high-likelihood customers before they reach the airport.

## Model

**Algorithm:** Random Forest Classifier
**Why:** Handles non-linear relationships well, requires little preprocessing, and produces feature importance scores — directly answering the business need to understand *which* variables drive booking behaviour, not just predict it.

**Key settings:**
- `n_estimators=300`
- `max_depth=12`
- `min_samples_leaf=5`
- `class_weight='balanced'` — to counteract the class imbalance (only ~15% of records are completed bookings)

## Validation Method

- 80/20 train/test split, stratified by target class
- 5-fold stratified cross-validation on the training set, to confirm results aren't dependent on one lucky split
- Final metrics reported on a **held-out test set** the model never saw during training

## Cross-Validation Results (Training Set)

| Metric | Mean | Std. Dev |
|---|---|---|
| Accuracy | 0.6968 | ± 0.0046 |
| Precision | 0.2878 | ± 0.0049 |
| Recall | 0.6966 | ± 0.0103 |
| F1 Score | 0.4073 | ± 0.0063 |
| ROC-AUC | 0.7600 | ± 0.0052 |

Low standard deviation across folds indicates the model's performance is stable and not the result of overfitting to one particular data split.

## Held-Out Test Set Results

| Metric | Score |
|---|---|
| Accuracy | 0.6956 |
| Precision | 0.2908 |
| Recall | 0.7193 |
| F1 Score | 0.4142 |
| ROC-AUC | 0.7673 |

Test set results closely match cross-validation results, confirming the model generalizes well to unseen data.

### Confusion Matrix

| | Predicted: Not Booked | Predicted: Booked |
|---|---|---|
| **Actual: Not Booked** | 5,880 | 2,624 |
| **Actual: Booked** | 420 | 1,076 |

## Interpreting the Numbers

- **ROC-AUC of 0.77** — well above the 0.50 baseline of random guessing, confirming the features genuinely carry predictive signal.
- **Recall of 72%** — the model successfully identifies nearly 3 out of 4 customers who go on to complete a booking. For a proactive targeting use case, this is the metric that matters most: missing a likely booker is more costly than wasting some outreach on a false positive.
- **Precision of 29%** — the trade-off for high recall. Because only 15% of customers book overall (a heavily imbalanced dataset), any model that casts a wide net to catch bookers will also flag many non-bookers. This is expected and acceptable for a *targeting* use case (better to reach out to some non-bookers than miss real ones), but would need revisiting if the business goal shifts toward minimizing wasted outreach cost.
- **Accuracy (70%) is a misleading headline metric here** — a naive model that always predicts "no booking" would score 85% accuracy while being completely useless for the business goal. This is why precision, recall, and ROC-AUC are reported instead of leading with accuracy.

## Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | booking_origin_frequency | 35.1% |
| 2 | length_of_stay | 12.1% |
| 3 | route_frequency | 12.0% |
| 4 | flight_duration | 10.0% |
| 5 | purchase_lead | 8.5% |
| 6 | flight_hour | 5.7% |
| 7 | total_extras_wanted | 3.1% |
| 8 | wants_extra_baggage | 2.4% |
| 9 | num_passengers | 2.0% |
| 10 | sales_channel_Mobile | 1.8% |

Day-of-week flags and trip type each contributed under 1% individually — negligible predictive value.

## Recommendation

The model demonstrates genuine, stable predictive power (ROC-AUC 0.77) and is well suited to a **proactive customer targeting** use case, where the priority is catching as many likely bookers as possible rather than minimizing false positives.

**Suggested next steps:**
1. Deploy as a scoring tool to rank customers by booking likelihood, feeding a marketing/outreach pipeline
2. Investigate *why* `booking_origin` and `route` are such strong predictors — this may reflect regional demand patterns worth exploring separately
3. If outreach cost becomes a concern, revisit the precision/recall trade-off by adjusting the classification threshold, rather than retraining from scratch
4. Consider enriching the dataset with customer-level historical data (past bookings, loyalty tier) if available, since the current model only uses single-search snapshot data