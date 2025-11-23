"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=60, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        m = len(assets)
        # 回到之前表現最好的策略思路
        equal_weight = 1.0 / m
        
        for i in range(len(self.price.index)):
            date = self.price.index[i]
            
            if i < 126:  # 6個月數據累積
                self.portfolio_weights.loc[date, assets] = equal_weight
                continue
                
            try:
                # 使用多時間窗口 (之前0.95夏普比率的配置)
                short_data = self.returns.iloc[max(0, i-63):i][assets]    # 3個月
                medium_data = self.returns.iloc[max(0, i-126):i][assets]  # 6個月
                
                # 計算收益和波動
                returns_3m = (1 + short_data).prod() - 1
                returns_6m = (1 + medium_data).prod() - 1
                volatility_3m = short_data.std()
                volatility_6m = medium_data.std()
                
                # 避免極端值
                volatility_3m = volatility_3m.clip(lower=volatility_3m.quantile(0.15))
                volatility_6m = volatility_6m.clip(lower=volatility_6m.quantile(0.15))
                
                # 計算夏普比率 (70% 3個月 + 30% 6個月)
                sharpe_3m = returns_3m / (volatility_3m + 1e-8)
                sharpe_6m = returns_6m / (volatility_6m + 1e-8)
                combined_sharpe = 0.7 * sharpe_3m + 0.3 * sharpe_6m
                
                # 選擇前3-4個夏普比率最高的資產
                positive_sharpe = combined_sharpe[combined_sharpe > 0]
                
                if len(positive_sharpe) >= 3:
                    top_assets = positive_sharpe.nlargest(4)
                    # 根據夏普比率大小分配權重
                    total_sharpe = top_assets.sum()
                    weights = pd.Series(0.0, index=assets)
                    for asset in top_assets.index:
                        weights[asset] = top_assets[asset] / total_sharpe
                    
                    # 權重約束：單一資產不超過40%
                    weights = weights.clip(upper=0.4)
                    weights = weights / weights.sum()
                    
                else:
                    # 沒有足夠正夏普，使用低波動策略
                    low_vol_assets = volatility_3m.nsmallest(4)
                    weights = pd.Series(0.0, index=assets)
                    for asset in low_vol_assets.index:
                        weights[asset] = 1.0 / len(low_vol_assets)
                
                # 確保權重有效
                weights = weights.fillna(0)
                if weights.sum() == 0:
                    weights = pd.Series(equal_weight, index=assets)
                else:
                    weights = weights / weights.sum()
                    
                self.portfolio_weights.loc[date, assets] = weights.values
                
            except Exception as e:
                self.portfolio_weights.loc[date, assets] = equal_weight
        
        self.portfolio_weights[self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
