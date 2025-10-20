-- ==================================================
-- OLAP ETL/ELT Script (PostgreSQL) — Stripe
-- Author : Loïc Valentini
-- Date   : 2025-10-20
-- Purpose: Build + (re)load a minimal star schema from the OLTP model
-- Notes  : Script idempotent (safe to run multiple times)
-- ==================================================

-- ==================================================
-- 0) Création du schéma cible
-- Le schéma "dw" contiendra toutes les tables OLAP (dimensions et faits).
-- ==================================================
CREATE SCHEMA IF NOT EXISTS dw;

-- ==================================================
-- 1) Création des tables du modèle en étoile
-- Chaque table "dim_*" est une dimension, et "fact_transactions" est la table de faits centrale.
-- ==================================================

-- Table calendrier (dimension temps)
CREATE TABLE IF NOT EXISTS dw.dim_date (
  date_id        BIGINT PRIMARY KEY,   -- Identifiant unique YYYYMMDD
  date           DATE NOT NULL,        -- Date complète
  day_of_week    TEXT,                 -- Jour de la semaine (texte)
  day            INT,                  -- Jour du mois
  month          INT,                  -- Mois (1-12)
  quarter        INT,                  -- Trimestre (1-4)
  year           INT,                  -- Année
  is_weekend     BOOLEAN,              -- Vrai si samedi ou dimanche
  is_holiday     BOOLEAN               -- Vrai si jour férié (placeholder)
);

-- Dimension des marchands (anonymisée)
CREATE TABLE IF NOT EXISTS dw.dim_merchant (
  merchant_id    BIGINT PRIMARY KEY,   -- Identifiant du marchand
  country_code   VARCHAR(5),           -- Code pays ISO
  status         TEXT,                 -- Statut (active, suspended, etc.)
  creation_date  TIMESTAMP             -- Date d'inscription
);

-- Dimension des clients (anonymisée)
CREATE TABLE IF NOT EXISTS dw.dim_customer (
  customer_id    BIGINT PRIMARY KEY,   -- Identifiant client
  country_code   VARCHAR(5),           -- Code pays ISO
  creation_date  TIMESTAMP             -- Date d'inscription
);

-- Dimension des plans d'abonnement
CREATE TABLE IF NOT EXISTS dw.dim_plan (
  plan_id        BIGINT PRIMARY KEY,   -- Identifiant du plan
  type           VARCHAR(255)          -- Type d'offre (ex: "Basic", "Pro", etc.)
);

-- Dimension des moyens de paiement
CREATE TABLE IF NOT EXISTS dw.dim_payment_method (
  payment_method_id BIGINT PRIMARY KEY, -- Identifiant moyen de paiement
  type           VARCHAR(100),          -- Type (carte, virement, etc.)
  support        VARCHAR(100)           -- Réseau / fournisseur (Visa, SEPA, etc.)
);

-- Dimension des factures
CREATE TABLE IF NOT EXISTS dw.dim_invoice (
  invoice_id     BIGINT PRIMARY KEY,    -- Identifiant facture
  issue_date     TIMESTAMP,             -- Date d'émission
  due_date       TIMESTAMP,             -- Date d'échéance
  status         TEXT                   -- Statut (open, paid, etc.)
);

-- Table de faits principale (transactions financières)
CREATE TABLE IF NOT EXISTS dw.fact_transactions (
  transaction_id     BIGINT PRIMARY KEY,   -- Identifiant transaction
  invoice_id         BIGINT,               -- FK vers facture
  customer_id        BIGINT NOT NULL,      -- FK vers client
  merchant_id        BIGINT NOT NULL,      -- FK vers marchand
  plan_id            BIGINT,               -- FK vers plan (nullable)
  payment_method_id  BIGINT,               -- FK vers moyen de paiement
  amount             INT NOT NULL,         -- Montant en centimes
  currency           VARCHAR(3) NOT NULL,  -- Devise (EUR, USD...)
  status             TEXT NOT NULL,        -- Statut de la transaction
  transaction_date   TIMESTAMP NOT NULL,   -- Date/heure transaction
  -- Contraintes de clé étrangère pour l’intégrité référentielle
  CONSTRAINT fk_ft_inv   FOREIGN KEY (invoice_id)        REFERENCES dw.dim_invoice(invoice_id),
  CONSTRAINT fk_ft_cust  FOREIGN KEY (customer_id)       REFERENCES dw.dim_customer(customer_id),
  CONSTRAINT fk_ft_merch FOREIGN KEY (merchant_id)       REFERENCES dw.dim_merchant(merchant_id),
  CONSTRAINT fk_ft_plan  FOREIGN KEY (plan_id)           REFERENCES dw.dim_plan(plan_id),
  CONSTRAINT fk_ft_pm    FOREIGN KEY (payment_method_id) REFERENCES dw.dim_payment_method(payment_method_id)
);

-- Index utiles pour les requêtes analytiques (par date, marchand, statut)
CREATE INDEX IF NOT EXISTS idx_ft_date     ON dw.fact_transactions (transaction_date);
CREATE INDEX IF NOT EXISTS idx_ft_merchant ON dw.fact_transactions (merchant_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_ft_status   ON dw.fact_transactions (status);

-- ==================================================
-- 2) Chargement des dimensions (UPSERT)
-- Chaque insertion met à jour la table cible sans créer de doublon.
-- ==================================================

-- Dimension des marchands
INSERT INTO dw.dim_merchant (merchant_id, country_code, status, creation_date)
SELECT  m.merchant_id,
        m.merchant_country_code,        -- Ajouté si dispo dans OLTP
        m.merchant_status::TEXT,
        m.merchant_creation_date
FROM merchants m
ON CONFLICT (merchant_id) DO UPDATE
SET name = EXCLUDED.name,
    status = EXCLUDED.status,
    creation_date = EXCLUDED.creation_date;

-- Dimension des clients
INSERT INTO dw.dim_customer (customer_id, country_code, creation_date)
SELECT  c.customer_id,
        c.customer_country_code,
        c.customer_creation_date
FROM customers c
ON CONFLICT (customer_id) DO UPDATE
SET country_code = EXCLUDED.country_code,
    creation_date = EXCLUDED.creation_date;

-- Dimension des plans
INSERT INTO dw.dim_plan (plan_id, type)
SELECT p.plan_id, p.plan_type
FROM plans p
ON CONFLICT (plan_id) DO UPDATE
SET type = EXCLUDED.type;

-- Dimension des moyens de paiement
INSERT INTO dw.dim_payment_method (payment_method_id, type, support)
SELECT pm.payment_method_id, pm.payment_method_type, pm.payment_method_support
FROM payment_methods pm
ON CONFLICT (payment_method_id) DO UPDATE
SET type = EXCLUDED.type,
    support = EXCLUDED.support;

-- Dimension des factures
INSERT INTO dw.dim_invoice (invoice_id, issue_date, due_date, status)
SELECT i.invoice_id, i.invoice_issue_date, i.invoice_due_date, i.invoice_status
FROM invoices i
ON CONFLICT (invoice_id) DO UPDATE
SET issue_date = EXCLUDED.issue_date,
    due_date   = EXCLUDED.due_date;

-- Dimension temporelle (dim_date)
-- Génère toutes les dates comprises entre la première et la dernière date du système OLTP.
WITH bounds AS (
  SELECT
    DATE_TRUNC('day', LEAST(
      (SELECT MIN(transaction_date) FROM transactions),
      (SELECT MIN(invoice_issue_date) FROM invoices)
    ))::date AS d_min,
    DATE_TRUNC('day', GREATEST(
      (SELECT COALESCE(MAX(transaction_date), NOW()) FROM transactions),
      (SELECT COALESCE(MAX(invoice_issue_date), NOW()) FROM invoices)
    ))::date AS d_max
),
series AS (
  SELECT generate_series(d_min, d_max, INTERVAL '1 day')::date AS d
  FROM bounds
)
INSERT INTO dw.dim_date (
  date_id, date, day_of_week, day, month, quarter, year, is_weekend, is_holiday
)
SELECT
  (TO_CHAR(d, 'YYYYMMDD'))::BIGINT      AS date_id,
  d                                     AS date,
  TO_CHAR(d, 'Day')                     AS day_of_week,
  EXTRACT(DAY FROM d)::INT              AS day,
  EXTRACT(MONTH FROM d)::INT            AS month,
  EXTRACT(QUARTER FROM d)::INT          AS quarter,
  EXTRACT(YEAR FROM d)::INT             AS year,
  (EXTRACT(ISODOW FROM d) IN (6,7))     AS is_weekend,
  FALSE                                 AS is_holiday
FROM series s
ON CONFLICT (date_id) DO NOTHING;

-- ==================================================
-- 3) Chargement de la table de faits (incrémental)
-- Seules les transactions plus récentes que la dernière déjà chargée sont insérées.
-- ==================================================

WITH last_loaded AS (
  SELECT COALESCE(MAX(transaction_date), TIMESTAMP '1900-01-01') AS max_ts
  FROM dw.fact_transactions
),
src AS (
  SELECT
      t.transaction_id,
      t.invoice_id,
      t.customer_id,
      t.merchant_id,
      s.plan_id,                       -- Plan actif au moment de la transaction (via souscription)
      t.payment_method_id,
      t.transaction_amount AS amount,
      t.transaction_currency AS currency,
      t.transaction_status  AS status,
      t.transaction_date
  FROM transactions t
  JOIN last_loaded ll ON t.transaction_date > ll.max_ts
  LEFT JOIN subscriptions s
    ON s.customer_id = t.customer_id
   AND t.transaction_date BETWEEN s.subscription_starting
                              AND COALESCE(s.subscription_ending, NOW())
)
INSERT INTO dw.fact_transactions (
  transaction_id, invoice_id, customer_id, merchant_id, plan_id, payment_method_id,
  amount, currency, status, transaction_date
)
SELECT *
FROM src
ON CONFLICT (transaction_id) DO UPDATE
SET invoice_id        = EXCLUDED.invoice_id,
    customer_id       = EXCLUDED.customer_id,
    merchant_id       = EXCLUDED.merchant_id,
    plan_id           = EXCLUDED.plan_id,
    payment_method_id = EXCLUDED.payment_method_id,
    amount            = EXCLUDED.amount,
    currency          = EXCLUDED.currency,
    status            = EXCLUDED.status,
    transaction_date  = EXCLUDED.transaction_date;

-- Fin du script
