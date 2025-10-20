-- ==================================================
-- Projet Stripe - Modèle OLTP
-- Auteur : Loïc Valentini
-- Date : 15/10/2025
-- Description : Création des tables, contraintes et index
-- ==================================================

CREATE TABLE merchants (
    merchant_id BIGSERIAL PRIMARY KEY,
    merchant_name VARCHAR(255) NOT NULL,
    merchant_creation_date TIMESTAMP NOT NULL DEFAULT NOW(),
    merchant_status VARCHAR(20) CHECK (merchant_status IN ('active','suspended','closed'))
);

CREATE TABLE merchants_private (
    merchant_id BIGINT PRIMARY KEY REFERENCES merchants(merchant_id),
    merchant_legal_name VARCHAR(255),
    merchant_address VARCHAR(255),
    merchant_contact_name VARCHAR(255),
    merchant_email VARCHAR(255),
    merchant_iban VARCHAR(34)
);

CREATE TABLE customers (
    customer_id BIGSERIAL PRIMARY KEY,
    merchant_id BIGINT REFERENCES merchants(merchant_id),
    customer_country_code CHAR(2),
    customer_creation_date TIMESTAMP DEFAULT NOW()
);

CREATE TABLE customers_private (
    customer_id BIGINT PRIMARY KEY REFERENCES customers(customer_id),
    customer_first_name VARCHAR(255),
    customer_last_name VARCHAR(255),
    customer_birthdate DATE,
    customer_address VARCHAR(255),
    customer_email VARCHAR(255)
);

CREATE TABLE plans (
    plan_id BIGSERIAL PRIMARY KEY,
    merchant_id BIGINT REFERENCES merchants(merchant_id),
    plan_type VARCHAR(50),
    plan_name VARCHAR(255),
    amount INT,
    currency CHAR(3),
    interval VARCHAR(20) CHECK (interval IN ('month','year')),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE subscriptions (
    subscription_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT REFERENCES customers(customer_id),
    plan_id BIGINT REFERENCES plans(plan_id),
    subscription_status VARCHAR(20) CHECK (subscription_status IN ('active','paused','canceled')),
    subscription_starting TIMESTAMP NOT NULL,
    subscription_ending TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE invoices (
    invoice_id BIGSERIAL PRIMARY KEY,
    merchant_id BIGINT REFERENCES merchants(merchant_id),
    customer_id BIGINT REFERENCES customers(customer_id),
    subscription_id BIGINT REFERENCES subscriptions(subscription_id),
    invoice_issue_date TIMESTAMP NOT NULL,
    invoice_due_date TIMESTAMP,
    amount_subtotal INT,
    amount_tax INT,
    amount_total INT,
    amount_due INT,
    invoice_status VARCHAR(20) CHECK (status IN ('draft','open','paid','void','uncollectible'))
);

CREATE TABLE payment_methods (
    payment_method_id BIGSERIAL PRIMARY KEY,
    payment_method_type VARCHAR(50),
    payment_method_support VARCHAR(50)
);

CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    invoice_id BIGINT REFERENCES invoices(invoice_id),
    payment_method_id BIGINT REFERENCES payment_methods(payment_method_id),
    customer_id BIGINT REFERENCES customers(customer_id),
    merchant_id BIGINT REFERENCES merchants(merchant_id),
    transaction_date TIMESTAMP NOT NULL DEFAULT NOW(),
    transaction_amount INT NOT NULL,
    transaction_currency CHAR(3) DEFAULT 'EUR',
    transaction_status VARCHAR(20) CHECK (transaction_status IN ('pending','succeeded','failed'))
);

CREATE TABLE refunds (
    refund_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT REFERENCES transactions(transaction_id),
    refund_amount INT,
    refund_reason VARCHAR(255),
    refund_date TIMESTAMP DEFAULT NOW(),
    refund_status VARCHAR(20)
);

CREATE TABLE chargebacks (
    chargeback_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT REFERENCES transactions(transaction_id),
    chargeback_amount INT,
    chargeback_reason VARCHAR(255),
    chargeback_opening TIMESTAMP DEFAULT NOW(),
    chargeback_status VARCHAR(20) CHECK (chargeback_status IN ('open','won','lost'))
);

-- ========================================
-- Index
-- ========================================

CREATE INDEX idx_tx_merchant_date ON transactions(merchant_id, transaction_date);
CREATE INDEX idx_inv_merchant_issue ON invoices(merchant_id, invoice_issue_date);
CREATE INDEX idx_subs_customer ON subscriptions(customer_id);
CREATE INDEX idx_refunds_tx ON refunds(transaction_id);
CREATE INDEX idx_cb_tx ON chargebacks(transaction_id);
