# Created on 2026/01/05
# Author: KAI
# mail: kai0615xx@gmail.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

class FundDataProcessor:

    def __init__(self, fund_path, base_mnt_path, base_daily_path, riskfree_path, theme_fund_path):
        self.fund_path = fund_path
        self.base_mnt_path = base_mnt_path
        self.base_daily_path = base_daily_path
        self.riskfree_path = riskfree_path
        self.theme_fund_path = theme_fund_path

        self.data = None
        self.base_mnt = None
        self.base_daily = None
        self.riskfree_mnt = None
        self.themed_fund = None
        self.data_filter = None
        self.data_mnt = None

        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        # 加载基金数据
        self.data = pd.read_csv(self.fund_path)
        del self.data['daily_profit']
        del self.data['weekly_yield']
        self.data['mnttime'] = pd.to_datetime(self.data['datetime']).dt.to_period('M')

        # 加载基准数据
        self.base_mnt = pd.read_excel(self.base_mnt_path)
        self.base_daily = pd.read_excel(self.base_daily_path)
        self.base_mnt['base_ret'] /= 100
        self.base_daily['base_ret'] /= 100
        self.base_mnt['mnttime'] = pd.to_datetime(self.base_mnt['mnttime']).dt.to_period('M')
        self.base_daily['datetime'] = pd.to_datetime(self.base_daily['datetime'])
        self.base_daily.dropna(inplace=True)

        # 计算夏普比率用的无风险利率
        riskfree = pd.read_csv(self.riskfree_path)
        riskfree['mnttime'] = pd.to_datetime(riskfree['datetime']).dt.to_period('M')
        self.riskfree_mnt = riskfree[['mnttime', 'riskfree_rate']].groupby(['mnttime'], as_index=False).mean()

        # 加载主题基金数据
        self.themed_fund = pd.read_csv(self.theme_fund_path)
        self.themed_fund.columns = ['order_book_id'] + [pd.to_datetime(col).strftime('%Y-%m')
                                                        for col in self.themed_fund.columns[1:]]

    def filter_themed_funds(self):
        data_filter = self.data.copy()

        # 定义检查周期
        check_periods = [
            {'target_month': '2023-04', 'themed_col': '2022-12'},
            {'target_month': '2023-08', 'themed_col': '2023-06'},
            {'target_month': '2024-04', 'themed_col': '2023-12'},
            {'target_month': '2024-08', 'themed_col': '2024-06'},
            {'target_month': '2025-04', 'themed_col': '2024-12'},
            {'target_month': '2025-08', 'themed_col': '2025-06'}
        ]

        # 获取每个目标月份的检查点
        check_points = []

        for period in check_periods:
            target_month = period['target_month']
            themed_col = period['themed_col']

            # 获取该月所有交易日
            month_data = data_filter[data_filter['mnttime'] == target_month]

            # 找到该月最后一个交易日
            last_trade_date = month_data['datetime'].max()

            # 找到下一个交易日（下个月的第一天，向后查找第一个交易日）
            next_month = pd.to_datetime(target_month) + pd.DateOffset(months=1)
            next_month_str = next_month.strftime('%Y-%m')

            # 获取下个月的所有交易日
            next_month_data = data_filter[data_filter['mnttime'] == next_month_str]
            first_next_trade_date = next_month_data['datetime'].min()

            check_points.append({
                'check_month': target_month,
                'check_date': last_trade_date,  # 筛选日
                'themed_col': themed_col,  # 检查起始日
                'delete_from': first_next_trade_date
            })

        # 开始筛选+删除 - 每次重新从原始数据中查询
        for i, cp in enumerate(check_points, 1):
            themed_col = cp['themed_col']
            check_date = cp['check_date']
            delete_from = cp['delete_from']

            print(f"\n=== 处理检查点: {cp['check_month']} ===")

            # 从原始数据中重新查询符合条件的基金
            # 每次都要基于原始 self.themed_fund 数据重新计算
            high_ratio_funds_all = self.themed_fund.loc[self.themed_fund[themed_col] >= 50, 'order_book_id'].tolist()

            # 创建当前检查时的数据副本
            current_data = self.data.copy()

            # 过滤出检查当天存在的基金
            check_day_funds = set(current_data[current_data['datetime'] == check_date]['order_book_id'])
            high_ratio_funds = list(set(high_ratio_funds_all) & set(check_day_funds))

            print(f"有 {len(high_ratio_funds)} 只基金单一行业占比 > 50%")

            # 应用删除 - 只删除从该检查点开始往后的数据
            mask = (data_filter['order_book_id'].isin(high_ratio_funds)) & (data_filter['datetime'] >= delete_from)
            data_filter = data_filter[~mask]

            # 可选：显示删除的数量
            deleted_count = mask.sum()
            print(f"删除 {deleted_count} 条记录（从 {delete_from} 开始）")

            # 清理内存
            del current_data

        self.data_filter = data_filter
        return self.data_filter
    
    def build_factors(self):
        # 提取月末、月初数据
        mnt_last = self.data_filter.groupby(['order_book_id', 'mnttime']).last()
        mnt_first = self.data_filter.groupby(['order_book_id', 'mnttime']).first()
        data_mnt = pd.DataFrame(index=mnt_last.index)
        data_mnt['last'] = mnt_last['acc_net_value']
        data_mnt['first'] = mnt_first['acc_net_value']

        # 计算月度收益率
        data_mnt['prev_last'] = data_mnt.groupby('order_book_id')['last'].shift(1)
        data_mnt['ret'] = data_mnt['last']/data_mnt['prev_last']-1
        data_mnt['ret'] = data_mnt['ret'].fillna(data_mnt['last']/data_mnt['first']-1)

        # 删除多余数据
        del data_mnt['prev_last']
        del data_mnt['last']
        del data_mnt['first']

        # 计算胜率
        data_mnt['rank'] = data_mnt.groupby('mnttime')['ret'].rank(ascending=False)

        # 当期是否获胜
        data_mnt['win_bool'] = (data_mnt['rank'] < data_mnt.groupby('mnttime')['rank'].transform('median')).astype(int)
        # 这里的胜率因子变成了一个负向因子，胜率越高的基金未来收益表现越差，需要拉长回测期间进一步观察。
        # rolling 12期胜率
        data_mnt['win_rate'] = data_mnt.groupby('order_book_id')['win_bool'].rolling(window=12, min_periods=12).mean().reset_index(level=0, drop=True)

        # 计算稳定性
        # rolling 12期排名标准差 - 稳定性
        data_mnt['rank_z'] = data_mnt.groupby('mnttime')['rank'].transform(lambda x: (x - x.mean())/x.std())
        data_mnt['stability'] = data_mnt.groupby('order_book_id')['rank_z'].rolling(window=12, min_periods=12).std().reset_index(level=0, drop=True)

        # 整理因子数据
        data_mnt.reset_index(inplace=True)
        self.data_filter.reset_index(inplace=True)

        # 只保留因子数据齐全的行
        data_mnt.dropna(inplace=True)
        self.data_mnt = data_mnt
        
        # 计算基准净值变化
        self.base_mnt['base_nv'] = (1 + self.base_mnt['base_ret']).cumprod().shift(1)
        self.base_mnt.loc[0,'base_nv'] = 1
        
        return self.data_mnt


class FactorSelector:
    
    @staticmethod
    def select_stocks(month_data, factors):
        # 初始化一个DataFrame来存储每个股票的因子分数
        stock_scores = pd.DataFrame()
        stock_scores['order_book_id'] = month_data['order_book_id']
        
        # 为每个因子计算十分组分数
        for factor in factors:
            factor_data = month_data[factor].dropna()
            
            # 对因子值进行十分组（0-9分），值越大分数越高
            factor_quantiles = factor_data.quantile([i/10 for i in range(1, 10)])
             # 为每个股票分配分数
            scores = []
            for fund in month_data['order_book_id']:
                factor_value = month_data.loc[month_data['order_book_id'] == fund, factor].values
                if  not pd.isna(factor_value[0]):
                    value = factor_value[0]
                    # 确定分数（0-9分）
                    score = 0
                    for i in range(1, 10):
                        if value > factor_quantiles.iloc[i-1]:
                            score = i
                        else: 
                            break
                    scores.append(score)
                else:
                    scores.append(np.nan)
            
            # 存储该因子的分数
            stock_scores[f'{factor}_score'] = scores
            
        # 因子值缺失则丢弃
        stock_scores.dropna(inplace=True)
            
        # 计算等权平均分数
        score_columns = [f'{factor}_score' for factor in factors]
        if len(score_columns) == 1:
            stock_scores['score'] = (stock_scores[score_columns]).mean(axis=1) # 单因子
        else:
            stock_scores['score'] = (stock_scores[score_columns] * [1/3, 1/3, 1/3]).sum(axis=1) #多因子

        # 获取高分股票（前10%）
        long_stkcd = stock_scores[stock_scores['score'] > stock_scores['score'].quantile(0.9)]['order_book_id'].tolist()
        score = stock_scores[['order_book_id','score']]

        return score, long_stkcd


class FactorBacktester:
    
    def __init__(self, data_processor, factor_selector=None):
        self.data_processor = data_processor
        self.factor_selector = factor_selector or FactorSelector()
    
    def backtest(self, factors):
        data_mnt = self.data_processor.data_mnt
        monthly_results = []
        current_net_value = 1
        
        # 按月份分组
        for month, month_data in data_mnt.groupby('mnttime'):

            if len(month_data[factors].dropna()) < 15:  # 确保足够数据
                continue

            # 获取下个月数据用于后续分析
            next_month = month + 1
            next_month_data = data_mnt[data_mnt['mnttime'] == next_month]
            
            # 调用多因子选股函数
            score, long_stkcd = self.factor_selector.select_stocks(month_data, factors)
            
            # 持仓下个月收益率
            portfolio_ret = next_month_data[next_month_data['order_book_id'].isin(long_stkcd)]['ret'].mean()

            # 持仓下个月超额收益率
            try:
                base_ret = self.data_processor.base_mnt.loc[self.data_processor.base_mnt['mnttime'] == next_month, 'base_ret'].iloc[0]
            except IndexError:
                base_ret = np.nan
            portfolio_exret = portfolio_ret - base_ret

            # 计算每月净值
            current_net_value *= 1 + portfolio_ret
            
            # 计算IC
            merged_for_ic = pd.merge(score, next_month_data[['order_book_id','ret']], on='order_book_id', how='inner')

            ic = merged_for_ic['score'].corr(merged_for_ic['ret'], method='spearman')

            # 结果汇总
            monthly_results.append({
                'mnttime': next_month,
                'portfolio_exret': portfolio_exret,
                'portfolio_ret': portfolio_ret,
                'net_value': current_net_value,
                'IC_prev': ic,
                'top_funds': len(long_stkcd),
                'total_funds': len(next_month_data)
            })

        # 调整结果格式
        result = pd.DataFrame(monthly_results)
        
        # 计算IC
        if len(result) > 1:
            ic_mean = result['IC_prev'].mean()
            ic_std = result['IC_prev'].std()
            icir = ic_mean / ic_std if ic_std != 0 else np.nan
            
            result['IC_mean'] = ic_mean
            result['ICIR'] = icir    
            return result

class PerformanceVisualizer:

    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def plot_net_value(self, net_values, factors, daily_data=None, monthly_data=None):
        """因子每日净值变化可视化"""
        if daily_data is None:
            daily_data = self.data_processor.data_filter
        if monthly_data is None:
            monthly_data = self.data_processor.data_mnt
            
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        daily_nv = pd.DataFrame(daily_data['datetime'].unique(), columns=['datetime'])
        daily_nv = daily_nv.sort_values(by='datetime')
        daily_nv['datetime'] = pd.to_datetime(daily_nv['datetime'])
        daily_nv['mnttime'] = daily_nv['datetime'].dt.to_period('M')
        daily_nv['daily_ret'] = np.nan
        
        # 计算每日净值
        # 这里的month就是持有月，而非上个函数的调仓月，选股时需要month-1
        for month in net_values['mnttime']:
            
            month_data = monthly_data[monthly_data['mnttime'] == month-1]

            # 调用多因子选股函数
            score, long_stkcd = FactorSelector.select_stocks(month_data, factors)

            # 筛出当前月份基金在持仓列表内的所有日度净值数据
            portfolio_data = daily_data[(daily_data['mnttime'] == month) & (daily_data['order_book_id'].isin(long_stkcd))]

            # 按日期分组，对每天持仓的ret求均值
            daily_mean_ret = portfolio_data.groupby('datetime', as_index=False)['change_rate'].mean()
            daily_mean_ret['datetime'] = pd.to_datetime(daily_mean_ret['datetime'])
            
            # 只更新当前月份对应的日期范围
            month_mask = daily_nv['mnttime'] == month
            
            if not daily_mean_ret.empty:
                date_to_rate = dict(zip(daily_mean_ret['datetime'], daily_mean_ret['change_rate']))
                daily_nv.loc[month_mask, 'daily_ret'] = daily_nv.loc[month_mask, 'datetime'].map(date_to_rate)

        # 计算累计净值
        daily_nv['daily_nv'] = (1 + daily_nv['daily_ret']).cumprod()
        daily_nv.dropna(inplace=True)

        # 绘制净值曲线
        fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 

        # 确保日期对齐 - 以daily_nv的日期为准
        aligned_dates = daily_nv['datetime']

        # 计算并对齐基准净值数据
        aligned_base_daily = self.data_processor.base_daily[self.data_processor.base_daily['datetime'].isin(aligned_dates)].reset_index(drop=True)
        aligned_base_daily['base_nv'] = (1 + aligned_base_daily['base_ret']).cumprod()

        # 第一条曲线：投资组合净值
        ax.plot(aligned_dates, daily_nv['daily_nv'], 
            linewidth=2, color='#F65314', label='Portfolio Net Value')

        # 第二条曲线：基准净值
        ax.plot(aligned_dates, aligned_base_daily['base_nv'], 
            linewidth=2, color='#8A8A8A', linestyle='--', label='Benchmark Net Value')

        # 填充区域
        ax.fill_between(aligned_dates, daily_nv['daily_nv'], 1, 
                    where=(daily_nv['daily_nv'] >= 1), 
                    color='#F65314', alpha=0.3, label='Positive Area')
        ax.fill_between(aligned_dates, daily_nv['daily_nv'], 1, 
                    where=(daily_nv['daily_nv'] < 1), 
                    color='#8A8A8A', alpha=0.3, label='Negative Area')

        # 基准线
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)

        # 设置图标题和标签
        plt.title(f'backtesting result:{factors}', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Net Value', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.show()
        
        return

class PerformanceAnalyzer:
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def calculate_metrics(self, result, factor):
        result = result.merge(self.data_processor.riskfree_mnt, on='mnttime', how='left')
        result['riskfree_exret'] = result['portfolio_ret'] - ((1 + result['riskfree_rate']/100)**(1/12) - 1)
        
        base_returns = self.data_processor.base_mnt['base_ret'][self.data_processor.base_mnt['mnttime'].isin(result['mnttime'])]
        base_annual_ret = (1 + base_returns).prod() ** (12 / len(result)) if len(base_returns) > 0 else 0
        
        summary_ret = {
            '因子名称': str(factor),
            '回测期间': f"{result['mnttime'].min().strftime('%Y-%m')} - {result['mnttime'].max().strftime('%Y-%m')}",
            '总月数': len(result),
            '年化收益率': (1 + result['portfolio_ret']).prod() ** (12 / len(result)) - 1,
            '年化超额': (1 + result['portfolio_ret']).prod() ** (12 / len(result)) - base_annual_ret, 
            '夏普比率': ((1 + result['riskfree_exret'].mean())**12 - 1) / (result['portfolio_ret'].std() * np.sqrt(12)) 
                        if result['portfolio_ret'].std() != 0 else np.nan,
            'RankIC': result['IC_mean'].iloc[0] if 'IC_mean' in result.columns and not result['IC_mean'].empty else np.nan,
            'ICIR': result['ICIR'].iloc[0] if 'ICIR' in result.columns and not result['ICIR'].empty else np.nan,
            'IC>0比例': (result['IC_prev'] > 0).mean(),
            '胜率': (result['portfolio_exret'] > 0).mean(),
            '最大回撤': (result['net_value'].min() - result['net_value'].max()) / result['net_value'].max() if result['net_value'].max() != 0 else np.nan
        }
        
        summary_ret = pd.DataFrame([summary_ret])
        return summary_ret


class FundFactorAnalysisSystem:

    def __init__(self, fund_path, base_mnt_path, base_daily_path, riskfree_path, theme_fund_path):
        self.data_processor = FundDataProcessor(
            fund_path, base_mnt_path, base_daily_path, riskfree_path, theme_fund_path
        )
        self.backtester = FactorBacktester(self.data_processor)
        self.visualizer = PerformanceVisualizer(self.data_processor)
        self.analyzer = PerformanceAnalyzer(self.data_processor)
    
    def run_analysis(self):
        print("开始数据处理...")
        self.data_processor.filter_themed_funds()
        self.data_processor.build_factors()
        
        print("开始因子回测...")
        
        # 收益率因子
        print("\n回测收益率因子...")
        result_ret = self.backtester.backtest(['ret'])
        self.visualizer.plot_net_value(result_ret[['mnttime','net_value']], ['ret'])
        metrics_ret = self.analyzer.calculate_metrics(result_ret.dropna(), ['ret'])
        print("收益率因子回测结果:")
        print(metrics_ret)
        
        # 稳定性因子
        print("\n回测稳定性因子...")
        result_stability = self.backtester.backtest(['stability'])
        self.visualizer.plot_net_value(result_stability[['mnttime','net_value']], ['stability'])
        metrics_stability = self.analyzer.calculate_metrics(result_stability.dropna(), ['stability'])
        print("稳定性因子回测结果:")
        print(metrics_stability)
        
        # 胜率因子
        print("\n回测胜率因子...")
        result_winrate = self.backtester.backtest(['win_rate'])
        self.visualizer.plot_net_value(result_winrate[['mnttime','net_value']], ['win_rate'])
        metrics_winrate = self.analyzer.calculate_metrics(result_winrate.dropna(), ['win_rate'])
        print("胜率因子回测结果:")
        print(metrics_winrate)
        
        # 多因子回测
        print("\n回测多因子...")
        result_multifactor = self.backtester.backtest(['ret', 'stability', 'win_rate'])
        self.visualizer.plot_net_value(result_multifactor[['mnttime','net_value']], ['ret', 'stability', 'win_rate'])
        metrics_multifactor = self.analyzer.calculate_metrics(result_multifactor.dropna(), ['ret', 'stability', 'win_rate'])
        print("多因子回测结果:")
        print(metrics_multifactor)
        
        # 保存结果到Excel
        result_dict = {
            'ret': result_ret if result_ret is not None else pd.DataFrame(),
            'stability': result_stability if result_stability is not None else pd.DataFrame(),
            'winrate': result_winrate if result_winrate is not None else pd.DataFrame(),
            'multifactor': result_multifactor if result_multifactor is not None else pd.DataFrame()
        }

        with pd.ExcelWriter('../output_multiple_sheets.xlsx', engine='openpyxl') as writer:
            for sheet_name, df in result_dict.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print("\n结果已保存至 output_multiple_sheets.xlsx")

if __name__ == "__main__":
    system = FundFactorAnalysisSystem(
        fund_path='./fund_data.csv',
        base_mnt_path='./930950_mnt.xlsx',
        base_daily_path='./930950_daily.xlsx',
        riskfree_path='./riskfree.csv',
        theme_fund_path='./fundid_theme_wind.csv'
    )
    
    system.run_analysis()
