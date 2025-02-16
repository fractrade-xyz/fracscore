import os
import json
import time
import requests
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from hyperliquid.info import Info
from hyperliquid.utils import constants
import argparse

# Set up logging configuration after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API Endpoints
HISTORICAL_BALANCE_API = "https://api.hypurrscan.io/holdersAtTime/FRAC/{timestamp}"
CURRENT_BALANCE_API = "https://api.hypurrscan.io/holders/FRAC"

# Start Date: 14th Jan 2025, 11:00 AM UTC
START_DATE = datetime(2025, 1, 14, 11, 0)
START_TIMESTAMP = int(START_DATE.timestamp())

# Cache Directories
CACHE_DIR = "historical_balances_cache"
TIMESTAMP_CACHE_FILE = os.path.join(CACHE_DIR, "cached_timestamps.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Add new constants
MIN_HOLDING_PERIOD = 7 * 24 * 60 * 60  # 7 days in seconds
MIN_HOLDING_AMOUNT = 100  # Minimum tokens required

# Add these constants at the top with other constants
EXCLUDED_WALLETS = {
    "0x000000000000000000000000000000000000dead",  # Burn address
    "0xffffffffffffffffffffffffffffffffffffffff"    # Hyperliquid LP
}

# Add to constants at the top
TOTAL_CIRCULATING_SUPPLY = 4_000_000  # 4 million FRAC tokens

def generate_fixed_timestamps():
    """Generate fixed timestamps every hour since the start date with slight randomization."""
    now = int(time.time())  # Current timestamp
    timestamps = []

    # Iterate over each hour since start date
    hours_since_start = (now - START_TIMESTAMP) // (60 * 60)

    for hour in range(hours_since_start + 1):
        base_time = START_TIMESTAMP + (hour * 60 * 60)  # Every hour
        offset = random.randint(-1800, 1800)  # Random shift (-30 min to +30 min)
        timestamps.append(base_time + offset)

    return timestamps

# If timestamp cache doesn't exist, generate and save
if not os.path.exists(TIMESTAMP_CACHE_FILE):
    cached_timestamps = generate_fixed_timestamps()
    with open(TIMESTAMP_CACHE_FILE, "w") as f:
        json.dump(cached_timestamps, f)
else:
    with open(TIMESTAMP_CACHE_FILE, "r") as f:
        cached_timestamps = json.load(f)


def get_historical_balances(timestamp):
    """Fetch historical balances at a given timestamp, using local caching."""
    cache_file = os.path.join(CACHE_DIR, f"{timestamp}.json")

    # Check if data is already cached
    if os.path.exists(cache_file):
        logger.debug(f"Loading cached balances for timestamp {timestamp}")
        with open(cache_file, "r") as f:
            data = json.load(f)
            # Filter out excluded wallets
            return {k: v for k, v in data.items() if k not in EXCLUDED_WALLETS}

    # Fetch from API
    logger.info(f"Fetching balances from API for timestamp {timestamp}")
    url = HISTORICAL_BALANCE_API.format(timestamp=timestamp)
    response = requests.get(url, headers={"accept": "application/json"})
    
    if response.status_code == 200:
        data = response.json()
        # Filter out excluded wallets before caching
        filtered_data = {k: v for k, v in data["holders"].items() if k not in EXCLUDED_WALLETS}
        
        # Cache filtered response
        with open(cache_file, "w") as f:
            json.dump(filtered_data, f)
        
        logger.info(f"Successfully cached balances for timestamp {timestamp}")
        return filtered_data
    else:
        logger.error(f"Error fetching historical balances for {timestamp}: {response.status_code}")
        return {}

def get_wallet_first_seen():
    """Determine when each wallet was first seen holding tokens"""
    logger.info("Starting to determine first appearance of wallets")
    first_seen = {}
    
    for timestamp in sorted(cached_timestamps):
        balances = get_historical_balances(timestamp)
        
        for wallet, amount in balances.items():
            if float(amount) >= MIN_HOLDING_AMOUNT:
                if wallet not in first_seen:
                    first_seen[wallet] = timestamp
                    logger.debug(
                        f"First appearance of wallet {wallet[:8]}... "
                        f"at timestamp {datetime.fromtimestamp(timestamp)} "
                        f"with balance {amount}"
                    )
                    
    logger.info(f"Found first appearance for {len(first_seen)} wallets")
    return first_seen

def calculate_loyalty_scores():
    """Compute loyalty scores for all wallets."""
    logger.info("Starting loyalty score calculation")
    
    # Get first appearance of each wallet
    wallet_first_seen = get_wallet_first_seen()
    current_time = int(time.time())

    logger.info("Loading historical balances")
    historical_balances = {}
    for ts in cached_timestamps:
        historical_balances[ts] = get_historical_balances(ts)

    # Fetch current balances
    current_balances = get_historical_balances(current_time)
    logger.info(f"Found {len(current_balances)} current wallet balances")

    # Store max historical balance per wallet
    wallet_max_balances = {}
    
    for timestamp, balances in historical_balances.items():
        for wallet, balance in balances.items():
            if wallet not in wallet_max_balances:
                wallet_max_balances[wallet] = float(balance)
            else:
                wallet_max_balances[wallet] = max(wallet_max_balances[wallet], float(balance))

    # Create DataFrame directly from records list
    records = []
    for wallet in set(wallet_max_balances.keys()).union(set(current_balances.keys())):
        records.append({
            "wallet": wallet,
            "max_balance": wallet_max_balances.get(wallet, 0),
            "current_balance": float(current_balances.get(wallet, 0)),
            "first_seen": wallet_first_seen.get(wallet, current_time)
        })
    
    df = pd.DataFrame.from_records(records)
    
    # Filter wallets based on minimum requirements
    holding_duration = current_time - df["first_seen"]
    df = df[
        (df["current_balance"] >= MIN_HOLDING_AMOUNT) & 
        (holding_duration >= MIN_HOLDING_PERIOD)
    ]
    
    logger.info(f"Found {len(df)} wallets meeting minimum requirements")
    
    if len(df) == 0:
        logger.warning("No wallets meet the minimum holding period and amount requirements")
        return pd.DataFrame(columns=["wallet", "current_balance", "holding_days", "Loyalty Score"])

    # Add holding duration in days for display
    df["holding_days"] = ((current_time - df["first_seen"]) / (24 * 60 * 60)).round(2)
    
    # Normalize the metrics
    df["balance_factor"] = (
        np.log10(df["current_balance"]) / np.log10(df["current_balance"].max())
    ).clip(lower=0)
    
    # Normalize holding time 
    max_holding_time = current_time - START_TIMESTAMP
    df["holding_factor"] = (current_time - df["first_seen"]) / max_holding_time
    
    # Calculate percentage sold relative to total supply
    df["percent_sold"] = (df["max_balance"] - df["current_balance"]) / TOTAL_CIRCULATING_SUPPLY * 100
    
    # Initialize selling impact
    df["selling_impact"] = 0
    
    # Micro sells (0-0.05% of supply)
    micro_sells = df["percent_sold"] <= 0.05
    df.loc[micro_sells, "selling_impact"] = -0.1 * (df.loc[micro_sells, "percent_sold"] / 0.05)
    
    # Small sells (0.05-0.1% of supply)
    small_sells = (df["percent_sold"] > 0.05) & (df["percent_sold"] <= 0.1)
    df.loc[small_sells, "selling_impact"] = -(
        0.1 +  # Base penalty from micro sells
        0.2 * ((df.loc[small_sells, "percent_sold"] - 0.05) / 0.05)  # Additional penalty
    )
    
    # Medium sells (0.1-0.3% of supply)
    medium_sells = (df["percent_sold"] > 0.1) & (df["percent_sold"] <= 0.3)
    df.loc[medium_sells, "selling_impact"] = -(
        0.3 +  # Base penalty from small sells
        0.7 * ((df.loc[medium_sells, "percent_sold"] - 0.1) / 0.2)  # Additional penalty
    )
    
    # Large sells (>0.3% of supply)
    large_sells = df["percent_sold"] > 0.3
    df.loc[large_sells, "selling_impact"] = -(
        1.0 +  # Base penalty from medium sells
        1.0 * ((df.loc[large_sells, "percent_sold"] - 0.3) / 0.3)  # Additional severe penalty
    )
    
    # Add time consideration - check if the max balance was within last 7 days
    recent_max_timestamps = {}
    min_check_timestamp = current_time - MIN_HOLDING_PERIOD  # 7 days ago
    
    # Find when each wallet had their max balance
    for timestamp in sorted(historical_balances.keys()):
        if timestamp >= min_check_timestamp:
            balances = historical_balances[timestamp]
            for wallet, balance in balances.items():
                if wallet in df.index:
                    if float(balance) == df.loc[wallet, "max_balance"]:
                        recent_max_timestamps[wallet] = timestamp
    
    # Double the selling impact if max balance was within last 7 days
    for wallet in df.index:
        if wallet in recent_max_timestamps:
            df.loc[wallet, "selling_impact"] *= 2
    
    logger.info("\nSelling Impact Examples:")
    sample_sells = pd.concat([
        df.nlargest(3, "percent_sold"),     # Largest sells
        df.nsmallest(3, "percent_sold"),    # Smallest sells
    ]).drop_duplicates()
    
    for _, row in sample_sells.iterrows():
        logger.info(
            f"\nWallet: {row['wallet'][:8]}..."
            f"\n  Tokens Sold: {row['max_balance'] - row['current_balance']:,.2f} FRAC"
            f"\n  Percent of Supply: {row['percent_sold']:.3f}%"
            f"\n  Selling Impact: {row['selling_impact']:.3f}"
            f"\n  Recent Max Balance: {'Yes' if row['wallet'] in recent_max_timestamps else 'No'}"
        )
    
    # Calculate time power for exponential holding bonus
    df["time_power"] = np.power(df["holding_days"] / 30, 1.5)  # Exponential growth per month
    df["holding_bonus"] = df["time_power"] * df["holding_factor"] * 200
    
    # Final score calculation with adjusted weights
    BALANCE_WEIGHT = 100    # Decreased to give more room for holding impact
    HOLDING_WEIGHT = 200    # Increased to make holding time more significant
    SELLING_WEIGHT = 10     # Keep selling impact weight
    BASE_SCORE = 100
    
    df["Loyalty Score"] = (
        BASE_SCORE + 
        (BALANCE_WEIGHT * df["balance_factor"] * 100) +
        (HOLDING_WEIGHT * df["holding_factor"] * 100) +
        (SELLING_WEIGHT * df["selling_impact"] * 100) +
        df["holding_bonus"]  # Dynamic time-based bonus
    ).clip(lower=100)
    
    # Round to integers
    df["Loyalty Score"] = df["Loyalty Score"].astype(int)
    
    # Add debug logging
    logger.info(f"\nScore Distribution:")
    logger.info(f"Min Score: {df['Loyalty Score'].min()}")
    logger.info(f"Max Score: {df['Loyalty Score'].max()}")
    logger.info(f"Mean Score: {df['Loyalty Score'].mean():.2f}")
    logger.info(f"Median Score: {df['Loyalty Score'].median()}")
    
    # Log example score breakdowns
    logger.info("\nExample Score Breakdowns:")
    sample_wallets = pd.concat([
        df.nlargest(3, "current_balance"),  # Top 3 by balance
        df.nlargest(3, "holding_days"),     # Top 3 by holding time
    ]).drop_duplicates()
    
    for _, row in sample_wallets.iterrows():
        logger.info(
            f"\nWallet: {row['wallet'][:8]}..."
            f"\n  Balance: {row['current_balance']:,.2f} FRAC"
            f"\n  Holding Days: {row['holding_days']:.1f}"
            f"\n  Time Power: {row['time_power']:.2f}"
            f"\n  Holding Bonus: {row['holding_bonus']:.2f}"
            f"\n  Final Score: {row['Loyalty Score']:,}"
        )
    
    return df[["wallet", "current_balance", "holding_days", "Loyalty Score"]].sort_values(
        by="Loyalty Score", ascending=False
    )

def distribute_rewards(loyalty_scores: pd.DataFrame, reward_pool_usd: float) -> pd.DataFrame:
    """
    Distribute rewards using Min-Max normalized balance and loyalty scores
    with equal weighting between the two factors
    """
    if len(loyalty_scores) == 0:
        logger.warning("No eligible wallets for rewards")
        return pd.DataFrame(columns=["wallet", "loyalty_score", "reward_usd"])
    
    rewards_df = loyalty_scores.copy()
    
    # Min-Max normalize balance (0-1 scale)
    min_balance = rewards_df["current_balance"].min()
    max_balance = rewards_df["current_balance"].max()
    if max_balance == min_balance:
        balance_normalized = pd.Series(1.0, index=rewards_df.index)
    else:
        balance_normalized = (rewards_df["current_balance"] - min_balance) / (max_balance - min_balance)
    
    # Min-Max normalize loyalty score (0-1 scale)
    min_score = rewards_df["Loyalty Score"].min()
    max_score = rewards_df["Loyalty Score"].max()
    if max_score == min_score:
        score_normalized = pd.Series(1.0, index=rewards_df.index)
    else:
        score_normalized = (rewards_df["Loyalty Score"] - min_score) / (max_score - min_score)
    
    # Combine normalized scores with equal weight (50/50)
    rewards_df["weighted_points"] = (0.5 * balance_normalized) + (0.5 * score_normalized)
    
    # Calculate rewards (handle case where total_points is 0)
    total_points = rewards_df["weighted_points"].sum()
    if total_points == 0:
        # If total_points is 0, distribute equally
        rewards_df["reward_usd"] = reward_pool_usd / len(rewards_df)
    else:
        rewards_df["reward_usd"] = (
            rewards_df["weighted_points"] / total_points * reward_pool_usd
        )
    
    # Round to 2 decimal places
    rewards_df["reward_usd"] = rewards_df["reward_usd"].round(2)
    
    # Calculate percentages (handle case where reward_pool_usd is 0)
    if reward_pool_usd == 0:
        rewards_df["reward_percentage"] = 0.0
    else:
        rewards_df["reward_percentage"] = (
            rewards_df["reward_usd"] / reward_pool_usd * 100
        ).round(2)
    
    # Add detailed logging
    logger.info("\nTop 10 rewards:")
    top_10 = rewards_df.sort_values("reward_usd", ascending=False).head(10)
    for _, row in top_10.iterrows():
        logger.info(
            f"Wallet: {row['wallet'][:8]}... "
            f"Balance: {row['current_balance']:,.2f} FRAC "
            f"(normalized: {balance_normalized.loc[row.name]:.3f}), "
            f"Score: {row['Loyalty Score']:,} "
            f"(normalized: {score_normalized.loc[row.name]:.3f}), "
            f"Reward: ${row['reward_usd']:,.2f} ({row['reward_percentage']:.1f}%)"
        )
    
    return rewards_df[["wallet", "Loyalty Score", "reward_usd", "reward_percentage"]].sort_values(
        by="reward_usd", ascending=False
    )

def save_and_display_results(loyalty_scores: pd.DataFrame, rewards: pd.DataFrame, reward_pool: float):
    """Save results to CSV and display a summary"""
    
    # Create results directory if it doesn't exist
    results_dir = "reward_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reward_distribution_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Combine relevant information
    results_df = pd.DataFrame({
        'wallet': rewards['wallet'],
        'current_balance': loyalty_scores['current_balance'],
        'holding_days': loyalty_scores['holding_days'],
        'loyalty_score': rewards['Loyalty Score'],
        'reward_usd': rewards['reward_usd'],
        'reward_percentage': rewards['reward_percentage']
    })
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    
    # Display summary
    print("\n" + "="*50)
    print(f"FRAC Loyalty Rewards Summary")
    print("="*50)
    print(f"Total Reward Pool: ${reward_pool:,.2f} USD")
    print(f"Number of Eligible Wallets: {len(rewards)}")
    print(f"Average Reward: ${rewards['reward_usd'].mean():.2f} USD")
    print(f"Highest Reward: ${rewards['reward_usd'].max():.2f} USD")
    print(f"Lowest Reward: ${rewards['reward_usd'].min():.2f} USD")
    print("\nTop 5 Rewards:")
    print("-"*50)
    top_5 = results_df.head()
    for _, row in top_5.iterrows():
        print(f"Wallet: {row['wallet'][:8]}...")
        print(f"  Balance: {row['current_balance']:,.2f} FRAC")
        print(f"  Holding Period: {row['holding_days']:.1f} days")
        print(f"  Loyalty Score: {row['loyalty_score']:,}")
        print(f"  Reward: ${row['reward_usd']:,.2f} ({row['reward_percentage']:.1f}%)")
        print("-"*30)
    
    print(f"\nResults saved to: {filepath}")
    return results_df


def get_wallet_score(wallet_address: str, loyalty_scores: pd.DataFrame) -> None:
    """Display detailed score information for a single wallet"""
    if wallet_address not in loyalty_scores.index:
        logger.error(f"Wallet {wallet_address} not found or doesn't meet minimum requirements")
        return
    
    wallet_data = loyalty_scores.loc[wallet_address]
    print("\n" + "="*50)
    print(f"Wallet Score Details: {wallet_address[:8]}...")
    print("="*50)
    print(f"Current Balance: {wallet_data['current_balance']:,.2f} FRAC ({wallet_data['supply_percentage']:.2f}% of supply)")
    print(f"Max Balance: {wallet_data['max_balance']:,.2f} FRAC ({wallet_data['max_supply_percentage']:.2f}% of supply)")
    print(f"Holding Period: {wallet_data['holding_days']:.1f} days")
    print(f"Loyalty Score: {wallet_data['Loyalty Score']:,}")
    print(f"Balance Factor: {wallet_data['balance_factor']:.3f}")
    print(f"Holding Factor: {wallet_data['holding_factor']:.3f}")
    print(f"Selling Impact: {wallet_data['selling_impact']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='FRAC Token Loyalty Score Calculator')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--wallet', type=str,
                      help='Calculate loyalty score for a specific wallet address')
    group.add_argument('--all', action='store_true',
                      help='Calculate scores for all eligible holders')
    
    # Add optional reward pool argument
    parser.add_argument('--reward-pool', type=float,
                       help='Calculate reward distribution with given USD pool amount')
    
    args = parser.parse_args()
    
    if args.wallet:
        print(f"Calculating score for wallet: {args.wallet}")
        loyalty_scores = calculate_loyalty_scores()
        get_wallet_score(args.wallet, loyalty_scores)
        
    elif args.all:
        print("Calculating scores for all eligible holders...")
        loyalty_scores = calculate_loyalty_scores()
        
        # Only calculate rewards if reward pool is specified
        if args.reward_pool:
            print(f"\nCalculating reward distribution for ${args.reward_pool:,.2f} USD pool")
            rewards = distribute_rewards(loyalty_scores, reward_pool_usd=args.reward_pool)
            results = save_and_display_results(loyalty_scores, rewards, reward_pool=args.reward_pool)
        else:
            # Just display the loyalty scores without rewards
            print("\nLoyalty Scores:")
            print(loyalty_scores.to_string())

if __name__ == "__main__":
    main()
