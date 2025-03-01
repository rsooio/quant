import akshare as ak
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
import os
from datetime import date, timedelta

# 配置参数
START_DATE = date(1990, 12, 19)
ADJUST = 'hfq'
PERIOD = 'daily'
MAX_WORKERS = 200  # 降低并发数避免被封
SAVE_DIR = 'data/bar'
TIMEOUT = 15

def get_last_date(existing: pd.DataFrame) -> date:
    """获取文件中的最后日期"""
    try:
        return existing['日期'].iloc[-1]
    except (KeyError, ValueError):
        return START_DATE

def get_last_trade_date():
    """获取包含今日的有效结束日期"""
    today = date.today()
    df = ak.tool_trade_date_hist_sina()
    return df[df['trade_date'] <= today]['trade_date'].iloc[-1]

def sync_single_stock(symbol: str, last: date, delisted: pd.Series) -> tuple:
    """增量同步单个股票数据"""
    try:
        # 读取已有数据
        file_path = os.path.join(SAVE_DIR, f"{symbol}.parquet")
        if os.path.exists(file_path):
            # XXX: 如果股票在两次同步之间被退市，无法获取到最后日期
            if symbol in delisted.values:
                return True, symbol, "Delisted"
            try:
                existing = pd.read_parquet(file_path)
                last_date = get_last_date(existing)
                if last_date == last:
                    return True, symbol, "Already up-to-date"
                start_date = last_date + timedelta(days=1)
            except (KeyError, ValueError):
                start_date = START_DATE
        else:
            start_date = START_DATE

        # 获取增量数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=PERIOD,
            start_date=start_date,
            end_date=last,
            adjust=ADJUST,
            timeout=TIMEOUT
        )

        if not df.empty:
            # 合并数据
            if os.path.exists(file_path):
                combined = pd.concat([existing, df]).drop_duplicates('日期')
                # TODO: 确认是否需要排序
                combined = combined.sort_values('日期')
                combined.to_parquet(file_path, index=False)
            else:
                df.to_parquet(file_path, index=False)
                
            return True, symbol, f"Added {len(df)} rows"
        return False, symbol, "No new data"
    
    except Exception as e:
        return False, symbol, str(e)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 获取交易日历
    print("获取交易日历...")
    last_trade_date = get_last_trade_date()
    print(f"最后有效交易日: {last_trade_date}")

    # 获取股票列表
    print("获取股票列表...")
    spot_df = ak.stock_zh_a_spot_em()
    symbols = spot_df['代码']
    delisted = spot_df[spot_df['最新价'].isnull()]['代码']
    print(f"需同步股票数量: {len(symbols)}")

    with Progress() as progress:
        task = progress.add_task("[cyan]同步历史数据...", total=len(symbols))
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(sync_single_stock, sym, last_trade_date, delisted): sym for sym in symbols}
            
            success = 0
            errors = []
            
            for future in as_completed(futures):
                done, symbol, msg = future.result()
                
                if done:
                    success += 1
                else:
                    errors.append((symbol, msg))
                
                # 实时更新状态
                progress.update(task, advance=1, description=f"[cyan]同步历史数据... 成功:{success} 失败:{len(errors)}")
                
        # 输出结果
        print(f"\n同步完成！成功: {success}, 失败: {len(errors)}")
        if errors:
            pd.DataFrame(errors, columns=['股票代码', '错误信息']).to_csv('sync_errors.csv', index=False)
            print("失败详情已保存到 sync_errors.csv")

if __name__ == "__main__":
    main()