-- ==================================================
-- Sample OLAP KPI Queries (PostgreSQL) — Stripe-like DW
-- Author : Loïc Valentini
-- Date   : 2025-10-20
-- Context: Works with schema `dw` built by your ETL script
-- Notes  : Amounts are stored in cents; queries divide by 100.0 for currency units.
-- ==================================================

/* ==================================================
   1) Monthly revenue by merchant (succeeded only)
      - What it shows: revenue trend per merchant by month.
================================================== */
WITH tx AS (
  SELECT
    DATE_TRUNC('month', f.transaction_date) AS month,
    f.merchant_id,
    SUM(CASE WHEN f.status = 'succeeded' THEN f.amount ELSE 0 END) AS revenue_cents
  FROM dw.fact_transactions f
  GROUP BY 1, 2
)
SELECT
  to_char(month, 'YYYY-MM') AS year_month,
  m.merchant_id,
  COALESCE(m.country_code, 'N/A') AS country_code,
  (revenue_cents / 100.0)        AS revenue
FROM tx
JOIN dw.dim_merchant m USING(merchant_id)
ORDER BY month, revenue DESC;


/* ==================================================
   2) Payment success rate by merchant (last 90 days)
      - What it shows: reliability of payment processing per merchant.
================================================== */
WITH window AS (
  SELECT NOW() - INTERVAL '90 days' AS start_dt
),
agg AS (
  SELECT
    f.merchant_id,
    COUNT(*)                                       AS total_tx,
    COUNT(*) FILTER (WHERE f.status = 'succeeded') AS succeeded_tx
  FROM dw.fact_transactions f, window w
  WHERE f.transaction_date >= w.start_dt
  GROUP BY f.merchant_id
)
SELECT
  a.merchant_id,
  ROUND((a.succeeded_tx::numeric / NULLIF(a.total_tx,0)) * 100.0, 2) AS success_rate_pct,
  a.succeeded_tx,
  a.total_tx
FROM agg a
ORDER BY success_rate_pct DESC NULLS LAST;


/* ==================================================
   3) Revenue by plan type (succeeded only)
      - What it shows: which subscription offers generate most revenue.
================================================== */
SELECT
  p.type AS plan_type,
  SUM(f.amount) FILTER (WHERE f.status = 'succeeded') / 100.0 AS revenue
FROM dw.fact_transactions f
JOIN dw.dim_plan p ON p.plan_id = f.plan_id
GROUP BY p.type
ORDER BY revenue DESC;


/* ==================================================
   4) Payment method mix (volume & revenue share)
      - What it shows: usage and business impact of payment methods.
================================================== */
WITH base AS (
  SELECT
    pm.type,
    pm.support,
    COUNT(*)                                       AS tx_count,
    SUM(CASE WHEN f.status = 'succeeded' THEN f.amount ELSE 0 END) AS revenue_cents
  FROM dw.fact_transactions f
  LEFT JOIN dw.dim_payment_method pm ON pm.payment_method_id = f.payment_method_id
  GROUP BY pm.type, pm.support
),
tot AS (
  SELECT
    SUM(tx_count)      AS total_tx,
    SUM(revenue_cents) AS total_rev
  FROM base
)
SELECT
  b.type,
  b.support,
  b.tx_count,
  ROUND((b.tx_count::numeric / NULLIF(t.total_tx,0)) * 100.0, 2) AS volume_share_pct,
  b.revenue_cents / 100.0 AS revenue,
  ROUND((b.revenue_cents::numeric / NULLIF(t.total_rev,0)) * 100.0, 2) AS revenue_share_pct
FROM base b, tot t
ORDER BY revenue DESC NULLS LAST;


/* ==================================================
   5) Average Days Sales Outstanding (DSO) by merchant
      - What it shows: average delay between invoice issue and first payment.
================================================== */
WITH first_pay AS (
  SELECT
    f.invoice_id,
    MIN(f.transaction_date) AS first_paid_at
  FROM dw.fact_transactions f
  WHERE f.status = 'succeeded' AND f.invoice_id IS NOT NULL
  GROUP BY f.invoice_id
),
joined AS (
  SELECT
    i.invoice_id,
    i.issue_date,
    fp.first_paid_at,
    EXTRACT(EPOCH FROM (fp.first_paid_at - i.issue_date)) / 86400.0 AS days_to_pay,
    ft.merchant_id
  FROM dw.dim_invoice i
  JOIN first_pay fp ON fp.invoice_id = i.invoice_id
  JOIN dw.fact_transactions ft ON ft.invoice_id = i.invoice_id
)
SELECT
  merchant_id,
  ROUND(AVG(days_to_pay)::numeric, 2) AS avg_dso_days,
  COUNT(*) AS paid_invoices
FROM joined
GROUP BY merchant_id
ORDER BY avg_dso_days;


/* ==================================================
   6) Top merchants by revenue (last 30 days)
      - What it shows: leaders in recent revenue.
================================================== */
WITH w AS (SELECT NOW() - INTERVAL '30 days' AS start_dt)
SELECT
  f.merchant_id,
  SUM(CASE WHEN f.status='succeeded' AND f.transaction_date >= w.start_dt THEN f.amount ELSE 0 END)/100.0 AS revenue_30d
FROM dw.fact_transactions f, w
GROUP BY f.merchant_id
ORDER BY revenue_30d DESC NULLS LAST
LIMIT 20;


/* ==================================================
   7) Month-over-month revenue growth (%)
      - What it shows: growth dynamics of total revenue.
================================================== */
WITH mth AS (
  SELECT DATE_TRUNC('month', transaction_date) AS month,
         SUM(CASE WHEN status='succeeded' THEN amount ELSE 0 END) AS rev_cents
  FROM dw.fact_transactions
  GROUP BY 1
),
lagged AS (
  SELECT month,
         rev_cents,
         LAG(rev_cents) OVER (ORDER BY month) AS prev_rev_cents
  FROM mth
)
SELECT
  to_char(month,'YYYY-MM') AS year_month,
  rev_cents/100.0          AS revenue,
  ROUND(((rev_cents - prev_rev_cents)::numeric / NULLIF(prev_rev_cents,0)) * 100.0, 2) AS mom_growth_pct
FROM lagged
ORDER BY month;


/* ==================================================
   8) Average ticket size by merchant (succeeded only)
      - What it shows: average order value to compare merchants.
================================================== */
SELECT
  merchant_id,
  ROUND(AVG(CASE WHEN status='succeeded' THEN amount END)::numeric / 100.0, 2) AS avg_ticket,
  COUNT(*) FILTER (WHERE status='succeeded') AS succeeded_tx
FROM dw.fact_transactions
GROUP BY merchant_id
HAVING COUNT(*) FILTER (WHERE status='succeeded') > 0
ORDER BY avg_ticket DESC;


/* ==================================================
   9) Active customers per month (>=1 succeeded transaction)
      - What it shows: engagement of the customer base over time.
================================================== */
WITH base AS (
  SELECT DATE_TRUNC('month', transaction_date) AS month,
         customer_id
  FROM dw.fact_transactions
  WHERE status='succeeded'
  GROUP BY 1, 2
)
SELECT to_char(month,'YYYY-MM') AS year_month,
       COUNT(DISTINCT customer_id) AS active_customers
FROM base
GROUP BY month
ORDER BY month;


/* ==================================================
   10) Failure rate by payment method (last 60 days)
       - What it shows: reliability differences across methods.
================================================== */
WITH w AS (SELECT NOW() - INTERVAL '60 days' AS start_dt),
agg AS (
  SELECT
    pm.type,
    pm.support,
    COUNT(*) FILTER (WHERE f.transaction_date >= w.start_dt) AS total_tx,
    COUNT(*) FILTER (WHERE f.status='failed' AND f.transaction_date >= w.start_dt) AS failed_tx
  FROM dw.fact_transactions f
  LEFT JOIN dw.dim_payment_method pm ON pm.payment_method_id = f.payment_method_id
  CROSS JOIN w
  GROUP BY pm.type, pm.support
)
SELECT
  type,
  support,
  failed_tx,
  total_tx,
  ROUND((failed_tx::numeric / NULLIF(total_tx,0)) * 100.0, 2) AS failed_rate_pct
FROM agg
ORDER BY failed_rate_pct DESC NULLS LAST;
