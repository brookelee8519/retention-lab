# Retention Lab

A churn prediction model plus a simulated retention-offer experiment, built
on the IBM Telco Customer Churn dataset. Basically pretending to be a data
scientist at a subscription company trying to figure out who's about to
leave and whether it's worth paying to keep them.

   ![Retention Lab app screenshot](screenshot_retention-lab.png)

## What it does

### Churn prediction
Trained two models on customer demographics, contract type, and usage data:
a Logistic Regression baseline and a Random Forest. Random Forest edges it
out slightly (0.845 ROC-AUC vs 0.841), but the more useful changes weren't
about picking a better model.

First, calibration. I used `class_weight="balanced"` to help the model catch
churners, but that also inflates its probabilities: it predicted 41% of test
customers would churn when only 27% actually did. Since the A/B simulation
and segment summary use those probabilities directly, I calibrated the model
(isotonic calibration). Now it predicts 27% churn vs. 27% actual, with the
same ROC-AUC (0.844).

Second, the threshold. Missing an actual churner costs more than flagging a
loyal customer by mistake: a missed churner is lost revenue, a false alarm
just means someone gets an offer they didn't need. With calibrated
probabilities, the cutoff has a direct cost meaning: flag a customer if their
churn probability is above cost of a false alarm ÷ (cost of a false alarm +
cost of a missed churner). I set it to 0.12, which corresponds to a missed
churner costing about 7x a wasted offer. That catches 90% of actual churners
(43% precision), vs. 80% at the default 0.5 cutoff.

### Simulated A/B test
The dataset doesn't include a real experiment, so I'm upfront that this
part is simulated, same as the simulated ratings in my Movie Match project. I'm
not claiming I ran a real test here, just showing I know how to set one up
and analyze it correctly.

Took the riskiest 25% of customers by predicted churn probability, randomly
split them into treatment (gets a retention discount) and control (doesn't),
and simulated outcomes with a 25% relative reduction in churn probability
for the treatment group: a believable effect size, not something rigged to
obviously "prove" the offer works. Then ran a two-proportion z-test
(cross-checked with chi-square, they agree) on retention rate between the
two groups.

Result: treatment group retained at 51.5%, control at 37.2%, a 14.3-point
lift, p < 0.0001, 95% CI [9.6pt, 18.9pt]. Statistically significant and a
big enough effect that it'd actually be worth running for real.

### Segmentation
K-means on tenure, monthly charges, and total charges, not on churn risk
itself (clustering on the thing you're about to make decisions with just
reproduces the model), to split customers into four segments: Loyal
high-value, Loyal low-spend, New low-spend, and New high-value (at-risk).

That last one is the interesting one: 2,276 customers, 50% average churn risk (48% actually churned), $81/month average revenue. High risk of leaving *and* real revenue to
protect, which is where a retention offer actually has good ROI. A low-spend
customer who's about to churn isn't worth discounting; this segment is.

### The app
Streamlit app with three tabs: look up a customer and see their risk score,
segment, and whether they should get the offer; the A/B test results; and
the model's feature importance and segment breakdown.

## Built with
Python, pandas, scikit-learn, scipy, Streamlit

## Running it

```bash
git clone https://github.com/brookelee8519/retention-lab.git
cd retention-lab
pip install -r requirements.txt

# run these once, in order, to generate the model + results the app reads
python src/train_churn_model.py
python src/ab_test.py
python src/segmentation.py

streamlit run app/app.py
```

Needs the Telco Customer Churn CSV in `data/telco_churn.csv`, grab it from
Kaggle (search "Telco Customer Churn") or IBM's sample datasets.

## Stuff that's not perfect / what I'd fix with more time
- the A/B test is simulated, so the effect size is something I chose to be
  realistic, not something that happened; a real version of this needs an
  actual experiment
- only used tenure/charges for segmentation; a real version would probably
  add usage features (streaming, add-on services) to split segments further
- the 7x cost ratio behind the 0.12 threshold is an assumption; a real
  version would plug in the actual offer cost and customer lifetime value.
  The threshold was also tuned on the test set, so a cleaner version would
  pick it on a separate validation set
- Random Forest beat Logistic Regression by less than half a point of AUC,
  which says the relationship here is mostly linear. A more complex model
  (gradient boosting, some interaction terms) might do better, but wasn't
  worth it for how small the gain looked early on
