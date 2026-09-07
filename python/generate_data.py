"""
Synthetic European banking transaction data generator.

Generates customers, accounts, merchants, and card/wire transactions across
8 European markets, with realistic messiness (duplicates, missing values,
mixed formats) and deliberately injected fraud patterns so the downstream
cleaning / feature engineering / detection layers have real work to do.

No real customer or transaction data is used anywhere in this project.
"""
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = "data/raw"

COUNTRIES = {
    "DE": {"name": "Germany", "locale": "de_DE", "weight": 0.22, "currency": "EUR"},
    "FR": {"name": "France", "locale": "fr_FR", "weight": 0.18, "currency": "EUR"},
    "ES": {"name": "Spain", "locale": "es_ES", "weight": 0.13, "currency": "EUR"},
    "IT": {"name": "Italy", "locale": "it_IT", "weight": 0.13, "currency": "EUR"},
    "NL": {"name": "Netherlands", "locale": "nl_NL", "weight": 0.09, "currency": "EUR"},
    "SE": {"name": "Sweden", "locale": "sv_SE", "weight": 0.08, "currency": "SEK"},
    "PL": {"name": "Poland", "locale": "pl_PL", "weight": 0.10, "currency": "PLN"},
    "IE": {"name": "Ireland", "locale": "en_IE", "weight": 0.07, "currency": "EUR"},
}
COUNTRY_CODES = list(COUNTRIES.keys())
COUNTRY_WEIGHTS = [COUNTRIES[c]["weight"] for c in COUNTRY_CODES]

# Rough centroid distances (km) used to derive a "geo-risk" signal between
# transaction country and the customer's home country.
CENTROIDS = {
    "DE": (51.16, 10.45), "FR": (46.60, 2.35), "ES": (40.30, -3.70),
    "IT": (42.50, 12.50), "NL": (52.30, 5.75), "SE": (60.10, 18.65),
    "PL": (52.00, 19.15), "IE": (53.35, -8.25),
}

MCC_CATEGORIES = [
    ("5411", "Grocery Stores", 0.9),
    ("5812", "Restaurants", 1.1),
    ("5651", "Clothing Stores", 1.0),
    ("4900", "Utilities", 0.3),
    ("5541", "Fuel Stations", 0.8),
    ("5732", "Electronics", 1.6),
    ("7011", "Hotels", 2.0),
    ("4511", "Airlines", 2.4),
    ("6051", "Crypto / Money Transfer", 3.5),
    ("5944", "Jewelry", 3.0),
    ("5999", "Misc Retail", 1.0),
    ("5967", "Direct Marketing / Subscriptions", 1.4),
]

CHANNELS = ["card_present", "card_not_present", "online", "atm_withdrawal", "wire_transfer"]
CHANNEL_WEIGHTS = [0.32, 0.28, 0.22, 0.10, 0.08]

N_CUSTOMERS = 6000
N_MERCHANTS = 450
N_MONTHS = 18
START_DATE = datetime(2024, 4, 1)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def gen_customers(n=N_CUSTOMERS):
    fakers = {c: Faker(COUNTRIES[c]["locale"]) for c in COUNTRY_CODES}
    rows = []
    for i in range(n):
        country = np.random.choice(COUNTRY_CODES, p=COUNTRY_WEIGHTS)
        fk = fakers[country]
        age = int(np.clip(np.random.normal(42, 14), 18, 90))
        signup = START_DATE - timedelta(days=int(np.random.exponential(400)))
        segment = np.random.choice(
            ["retail", "premium", "student", "business"], p=[0.55, 0.15, 0.15, 0.15]
        )
        rows.append({
            "customer_id": f"CUST{i:06d}",
            "full_name": fk.name(),
            "email": fk.email(),
            "country": country,
            "age": age,
            "segment": segment,
            "signup_date": signup.strftime("%Y-%m-%d"),
            "kyc_verified": np.random.choice([True, False], p=[0.96, 0.04]),
        })
    return pd.DataFrame(rows)


def gen_accounts(customers):
    rows = []
    for _, c in customers.iterrows():
        n_accounts = np.random.choice([1, 2], p=[0.8, 0.2])
        for j in range(n_accounts):
            acc_type = np.random.choice(["checking", "savings", "credit_card"], p=[0.55, 0.25, 0.20])
            opened = pd.to_datetime(c["signup_date"]) + timedelta(days=int(np.random.exponential(20)))
            rows.append({
                "account_id": f"ACC{len(rows):07d}",
                "customer_id": c["customer_id"],
                "account_type": acc_type,
                "opened_date": opened.strftime("%Y-%m-%d"),
                "home_country": c["country"],
                "credit_limit": round(np.random.choice([0, 2000, 5000, 10000, 20000]), 2)
                if acc_type == "credit_card" else np.nan,
            })
    return pd.DataFrame(rows)


def gen_merchants(n=N_MERCHANTS):
    rows = []
    for i in range(n):
        mcc, category, risk_mult = MCC_CATEGORIES[np.random.choice(len(MCC_CATEGORIES))]
        country = np.random.choice(COUNTRY_CODES, p=COUNTRY_WEIGHTS)
        rows.append({
            "merchant_id": f"MERCH{i:05d}",
            "merchant_name": f"{category.split(' / ')[0].split(' ')[0]} {fake_company_suffix(i)}",
            "mcc": mcc,
            "mcc_category": category,
            "country": country,
            "base_risk_multiplier": risk_mult,
            "onboarded_date": (START_DATE - timedelta(days=int(np.random.exponential(500)))).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def fake_company_suffix(i):
    suffixes = ["GmbH", "SARL", "SA", "SRL", "BV", "AB", "Sp. z o.o.", "Ltd", "Group", "& Co"]
    return f"{i:04d} {suffixes[i % len(suffixes)]}"


def customer_baseline(customer_row):
    """Each customer gets a stable spending baseline used to simulate normal behaviour."""
    segment_base = {"retail": 55, "premium": 180, "student": 25, "business": 320}
    mean_amt = max(8, np.random.gamma(shape=2.2, scale=segment_base[customer_row["segment"]] / 2.2))
    txn_rate_per_day = np.random.gamma(shape=2.0, scale=0.045)  # avg ~0.09 transactions/day (~50 over 18mo)
    return mean_amt, txn_rate_per_day


def gen_transactions(customers, accounts, merchants):
    accounts_by_cust = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()
    merchants_idx = merchants.set_index("merchant_id")
    merchant_ids = merchants["merchant_id"].values
    merchant_country = merchants.set_index("merchant_id")["country"].to_dict()
    merchant_riskmult = merchants.set_index("merchant_id")["base_risk_multiplier"].to_dict()

    rows = []
    fraud_case_log = []
    end_date = START_DATE + timedelta(days=30 * N_MONTHS)
    n_days = (end_date - START_DATE).days

    for _, cust in customers.iterrows():
        cust_accounts = accounts_by_cust.get(cust["customer_id"], [])
        if not cust_accounts:
            continue
        mean_amt, txn_rate = customer_baseline(cust)
        home_country = cust["country"]
        is_dormant = np.random.random() < 0.05  # some customers barely transact

        active_days = n_days if not is_dormant else int(n_days * 0.15)
        expected_txns = max(1, int(txn_rate * active_days))

        for _ in range(np.random.poisson(expected_txns)):
            day_offset = np.random.randint(0, n_days)
            ts = START_DATE + timedelta(days=day_offset,
                                         hours=int(np.clip(np.random.normal(14, 5), 0, 23)),
                                         minutes=np.random.randint(0, 60))
            account_id = np.random.choice(cust_accounts)
            channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
            merchant_id = np.random.choice(merchant_ids) if channel != "wire_transfer" else None
            risk_mult = merchant_riskmult.get(merchant_id, 1.0) if merchant_id else 1.2
            amount = round(max(1.0, np.random.lognormal(mean=np.log(mean_amt), sigma=0.6) * (risk_mult ** 0.3)), 2)

            txn_country = home_country
            if channel in ("online", "card_not_present") and np.random.random() < 0.08:
                txn_country = np.random.choice(COUNTRY_CODES, p=COUNTRY_WEIGHTS)
            elif merchant_id is not None and np.random.random() < 0.5:
                txn_country = merchant_country.get(merchant_id, home_country)

            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": account_id,
                "customer_id": cust["customer_id"],
                "merchant_id": merchant_id,
                "timestamp": ts,
                "amount": amount,
                "currency": COUNTRIES[home_country]["currency"],
                "channel": channel,
                "txn_country": txn_country,
                "home_country": home_country,
                "device_id": f"DEV-{cust['customer_id'][-4:]}-{np.random.randint(1,3)}",
                "is_fraud": False,
                "fraud_type": None,
            })

    txns = pd.DataFrame(rows)
    txns = inject_fraud_patterns(txns, customers, merchants, fraud_case_log)
    return txns, pd.DataFrame(fraud_case_log)


def inject_fraud_patterns(txns, customers, merchants, fraud_case_log):
    """Injects four distinct, labeled fraud scenarios into the legitimate transaction stream."""
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    merchant_ids = merchants["merchant_id"].values
    all_customers = customers["customer_id"].values

    # Pre-group once so fraud injection doesn't do an O(n) scan per customer.
    txns_by_cust = {cid: grp for cid, grp in txns.groupby("customer_id")}

    def sample_base_txn(cust_id):
        grp = txns_by_cust.get(cust_id)
        if grp is None or grp.empty:
            return None
        return grp.sample(1).iloc[0]

    new_rows = []

    # 1) Card-testing: bursts of small, rapid-fire card_not_present transactions
    #    against random merchants within a few minutes, on a compromised card.
    n_card_testing_victims = 70
    victims = np.random.choice(all_customers, n_card_testing_victims, replace=False)
    for cust_id in victims:
        base = sample_base_txn(cust_id)
        if base is None:
            continue
        burst_start = pd.to_datetime(base["timestamp"]) + timedelta(days=np.random.randint(5, 500))
        burst_len = np.random.randint(6, 18)
        for k in range(burst_len):
            new_rows.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": base["account_id"],
                "customer_id": cust_id,
                "merchant_id": np.random.choice(merchant_ids),
                "timestamp": burst_start + timedelta(seconds=int(np.random.uniform(5, 90) * k)),
                "amount": round(np.random.uniform(0.5, 4.0), 2),
                "currency": base["currency"],
                "channel": "card_not_present",
                "txn_country": np.random.choice(COUNTRY_CODES, p=COUNTRY_WEIGHTS),
                "home_country": base["home_country"],
                "device_id": f"DEV-UNK-{np.random.randint(1000,9999)}",
                "is_fraud": True,
                "fraud_type": "card_testing",
            })
        fraud_case_log.append({"customer_id": cust_id, "fraud_type": "card_testing", "n_txns": burst_len})

    # 2) Account takeover: one large, geographically-distant, new-device
    #    transaction shortly after a burst of failed-login-like small probes.
    n_ato_victims = 55
    victims = np.random.choice(all_customers, n_ato_victims, replace=False)
    for cust_id in victims:
        base = sample_base_txn(cust_id)
        if base is None:
            continue
        ts = pd.to_datetime(base["timestamp"]) + timedelta(days=np.random.randint(5, 500))
        foreign_country = np.random.choice([c for c in COUNTRY_CODES if c != base["home_country"]])
        big_amount = round(np.random.uniform(800, 6500), 2)
        new_rows.append({
            "transaction_id": str(uuid.uuid4()),
            "account_id": base["account_id"],
            "customer_id": cust_id,
            "merchant_id": np.random.choice(merchant_ids),
            "timestamp": ts,
            "amount": big_amount,
            "currency": base["currency"],
            "channel": np.random.choice(["online", "wire_transfer"]),
            "txn_country": foreign_country,
            "home_country": base["home_country"],
            "device_id": f"DEV-NEW-{np.random.randint(1000,9999)}",
            "is_fraud": True,
            "fraud_type": "account_takeover",
        })
        fraud_case_log.append({"customer_id": cust_id, "fraud_type": "account_takeover", "n_txns": 1})

    # 3) Merchant collusion: a small set of merchants systematically produce
    #    inflated, round-number transactions for a rotating set of customers.
    colluding_merchants = np.random.choice(merchant_ids, 6, replace=False)
    for m_id in colluding_merchants:
        n_cases = np.random.randint(15, 35)
        colluded_custs = np.random.choice(all_customers, n_cases, replace=False)
        for cust_id in colluded_custs:
            base = sample_base_txn(cust_id)
            if base is None:
                continue
            ts = pd.to_datetime(base["timestamp"]) + timedelta(days=np.random.randint(1, 540))
            new_rows.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": base["account_id"],
                "customer_id": cust_id,
                "merchant_id": m_id,
                "timestamp": ts,
                "amount": float(np.random.choice([200, 250, 300, 500, 750, 1000])),
                "currency": base["currency"],
                "channel": "card_not_present",
                "txn_country": base["home_country"],
                "home_country": base["home_country"],
                "device_id": base["device_id"],
                "is_fraud": True,
                "fraud_type": "merchant_collusion",
            })
        fraud_case_log.append({"customer_id": None, "merchant_id": m_id, "fraud_type": "merchant_collusion", "n_txns": n_cases})

    # 4) Synthetic identity / mule accounts: brand-new accounts that receive
    #    a wire then immediately drain via ATM withdrawals in another country.
    mule_customers = customers[pd.to_datetime(customers["signup_date"]) > (START_DATE - timedelta(days=60))]
    mule_sample = mule_customers.sample(min(40, len(mule_customers))) if len(mule_customers) else pd.DataFrame()
    for _, cust in mule_sample.iterrows():
        cust_id = cust["customer_id"]
        cust_txns = txns_by_cust.get(cust_id)
        base_acc = cust_txns["account_id"].iloc[0] if cust_txns is not None and not cust_txns.empty else f"ACC_MULE_{cust_id}"
        ts0 = pd.to_datetime(cust["signup_date"]) + timedelta(days=np.random.randint(1, 20))
        deposit = round(np.random.uniform(3000, 15000), 2)
        new_rows.append({
            "transaction_id": str(uuid.uuid4()), "account_id": base_acc, "customer_id": cust_id,
            "merchant_id": None, "timestamp": ts0, "amount": deposit, "currency": "EUR",
            "channel": "wire_transfer", "txn_country": cust["country"], "home_country": cust["country"],
            "device_id": f"DEV-{cust_id[-4:]}-1", "is_fraud": True, "fraud_type": "mule_account",
        })
        remaining = deposit
        for w in range(np.random.randint(2, 5)):
            withdraw = round(min(remaining, np.random.uniform(500, 2500)), 2)
            remaining -= withdraw
            new_rows.append({
                "transaction_id": str(uuid.uuid4()), "account_id": base_acc, "customer_id": cust_id,
                "merchant_id": None, "timestamp": ts0 + timedelta(hours=np.random.randint(1, 48) * (w + 1)),
                "amount": withdraw, "currency": "EUR", "channel": "atm_withdrawal",
                "txn_country": np.random.choice([c for c in COUNTRY_CODES if c != cust["country"]]),
                "home_country": cust["country"], "device_id": None,
                "is_fraud": True, "fraud_type": "mule_account",
            })
        fraud_case_log.append({"customer_id": cust_id, "fraud_type": "mule_account", "n_txns": 1 + w + 1})

    fraud_df = pd.DataFrame(new_rows)
    combined = pd.concat([txns, fraud_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return combined


def add_messiness(customers, accounts, merchants, txns):
    """Injects the kind of real-world messiness a source-system export has."""
    txns = txns.copy()

    # Duplicate ~0.6% of rows (double-posted transactions)
    dupes = txns.sample(frac=0.006, random_state=1)
    txns = pd.concat([txns, dupes], ignore_index=True)

    # Missing device_id for some legit online transactions (not just mule rows)
    mask = (txns["channel"] == "online") & (np.random.random(len(txns)) < 0.03)
    txns.loc[mask, "device_id"] = np.nan

    # Mixed timestamp formats: convert a slice to strings in a different format
    txns["timestamp"] = txns["timestamp"].astype(str)
    slice_idx = txns.sample(frac=0.15, random_state=2).index
    txns.loc[slice_idx, "timestamp"] = pd.to_datetime(txns.loc[slice_idx, "timestamp"]).dt.strftime("%d/%m/%Y %H:%M")

    # A handful of negative amounts from a sign-flip bug in one export batch
    neg_idx = txns.sample(frac=0.004, random_state=3).index
    txns.loc[neg_idx, "amount"] = -txns.loc[neg_idx, "amount"].abs()

    # Currency code casing inconsistencies
    case_idx = txns.sample(frac=0.05, random_state=4).index
    txns.loc[case_idx, "currency"] = txns.loc[case_idx, "currency"].str.lower()

    # Whitespace / casing noise in customer names & emails
    customers = customers.copy()
    noisy_idx = customers.sample(frac=0.05, random_state=5).index
    customers.loc[noisy_idx, "full_name"] = "  " + customers.loc[noisy_idx, "full_name"].str.upper() + "  "

    return customers, accounts, merchants, txns


def main():
    print("Generating customers...")
    customers = gen_customers()
    print("Generating accounts...")
    accounts = gen_accounts(customers)
    print("Generating merchants...")
    merchants = gen_merchants()
    print("Generating transactions with injected fraud patterns (this takes a moment)...")
    txns, fraud_log = gen_transactions(customers, accounts, merchants)
    customers, accounts, merchants, txns = add_messiness(customers, accounts, merchants, txns)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    customers.to_csv(f"{OUT_DIR}/customers_raw.csv", index=False)
    accounts.to_csv(f"{OUT_DIR}/accounts_raw.csv", index=False)
    merchants.to_csv(f"{OUT_DIR}/merchants_raw.csv", index=False)
    txns.to_csv(f"{OUT_DIR}/transactions_raw.csv", index=False)
    fraud_log.to_csv(f"{OUT_DIR}/fraud_case_log.csv", index=False)

    print(f"customers:    {len(customers):,}")
    print(f"accounts:     {len(accounts):,}")
    print(f"merchants:    {len(merchants):,}")
    print(f"transactions: {len(txns):,}  (fraud: {int(txns['is_fraud'].sum()):,}, "
          f"{txns['is_fraud'].mean()*100:.3f}%)")
    print("Done. Raw files written to data/raw/")


if __name__ == "__main__":
    main()
