#!/usr/bin/env python3
"""
优质股票候选池生成器 — 初筛模块

定位: 不负责"买不买"，只负责把明显差的股票筛掉，留下优质候选池。
输出: 所有通过筛选的股票（不排名，只有通过/不通过）

五层过滤:
  1. 基础排除: ST/退市/上市不足2年/北交所/科创板/价格<3元
  2. 流动性: 近20日平均成交额 ≥ 5000万
  3. 基本面质量: ROE≥8% / 净利润>0 / 营收同比>-10% / 净利润同比>-20% /
     资产负债率≤70% / 经营现金流为正或现金流/净利润≥0.5
  4. 估值不过分: 0<PE_TTM≤80 / 0<PB≤10
  5. 趋势不能太差: 收盘价≥MA120 或 MA60斜率未明显向下

用法:
  python screener_quality.py
  python screener_quality.py --pool etf    # ETF持仓池（默认，快）
  python screener_quality.py --pool all    # 全市场（慢）
"""

import json, csv, os, sys, time, re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tushare_provider import get_tushare_api

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'screener')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY = datetime.now().strftime('%Y%m%d')
TODAY_DASH = datetime.now().strftime('%Y-%m-%d')


def _to_float(v):
    try:
        return float(v) if v not in (None, '-', '') else None
    except (ValueError, TypeError):
        return None


def _sina_code(code):
    return f"sh{code}" if code.startswith('6') else f"sz{code}"


# ============================================================
# 第一层：基础排除
# ============================================================

def load_stock_pool(pool_type='etf'):
    """加载股票池，执行基础排除"""
    pro = get_tushare_api()

    if pool_type == 'etf':
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_analysis.db')
        conn = __import__('sqlite3').connect(db_path)
        codes = pd.read_sql("SELECT DISTINCT stock_code FROM etf_holdings", conn)['stock_code'].tolist()
        conn.close()
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,list_date')
        df_basic['code'] = df_basic['ts_code'].str[:6]
        df = df_basic[df_basic['code'].isin(codes)]
    else:
        df = pro.stock_basic(exchange='', list_status='L',
                             fields='ts_code,name,industry,market,list_date')
        if df is None or df.empty:
            return pd.DataFrame()
        df['code'] = df['ts_code'].str[:6]

    # 排除 ST / 退市
    df = df[~df['name'].str.contains('ST|退', na=False)]

    # 上市满2年
    cutoff = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    df = df[df['list_date'] <= cutoff]

    # 排除北交所（8/4开头）和科创板（688开头）
    df = df[~df['code'].str.startswith(('8', '4', '688'))]

    print(f"📊 第一层 基础排除后: {len(df)} 只")
    return df


# ============================================================
# 数据获取
# ============================================================

def fetch_realtime_sina(codes, batch_size=800):
    """批量实时行情（新浪）"""
    all_data = {}
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        sina_codes = ",".join([_sina_code(c) for c in batch])
        try:
            resp = requests.get(f"https://hq.sinajs.cn/list={sina_codes}",
                                headers={"Referer": "https://finance.sina.com.cn"}, timeout=15)
            resp.encoding = 'gbk'
            for line in resp.text.strip().split('\n'):
                if '="' not in line:
                    continue
                m = re.match(r'var hq_str_(s[hz]\d{6})="(.*)"', line)
                if not m:
                    continue
                code = m.group(1)[2:]
                f = m.group(2).split(',')
                if len(f) < 32:
                    continue
                price = _to_float(f[3]) or _to_float(f[2])
                if not price or price <= 0:
                    continue
                prev = _to_float(f[2])
                all_data[code] = {
                    'name': f[0], 'price': price,
                    'volume': _to_float(f[8]),
                    'amount': _to_float(f[9]),
                    'change_pct': (price - prev) / prev * 100 if prev and prev > 0 else 0,
                }
        except Exception:
            pass
    return all_data


def fetch_hist_sina(code, days=130):
    """获取历史K线"""
    try:
        resp = requests.get(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
            params={"symbol": _sina_code(code), "scale": 240, "ma": "no", "datalen": days}, timeout=10)
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['amount'] = df['close'] * df['volume']
        return df
    except Exception:
        return None


def fetch_hist_parallel(codes, max_workers=15):
    """并行获取历史K线"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_hist_sina, c): c for c in codes}
        for fut in as_completed(futs, timeout=600):
            code = futs[fut]
            try:
                df = fut.result()
                if df is not None and len(df) >= 20:
                    results[code] = df
            except Exception:
                pass
    print(f"  ✅ 历史K线: {len(results)}/{len(codes)} 只")
    return results


def fetch_financial_em(codes):
    """
    东财获取财务数据（RPT_F10_FINANCE_MAINFINADATA）

    取4期报告，优先使用年报数据（ROE是单期值，年报才是年化值）。
    """
    results = {}
    total = len(codes)
    print(f"  📥 获取 {total} 只财务数据（东财 MAINFINADATA）...")

    for i, code in enumerate(codes):
        try:
            url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
            params = {
                "reportName": "RPT_F10_FINANCE_MAINFINADATA",
                "columns": "SECURITY_CODE,ROEJQ,PARENTNETPROFIT,TOTALOPERATEREVE,"
                           "TOTALOPERATEREVETZ,PARENTNETPROFITTZ,ZCFZL,"
                           "NETCASH_OPERATE_PK,XSJLL,REPORT_TYPE",
                "filter": f'(SECURITY_CODE="{code}")',
                "pageSize": 4,
                "source": "WEB",
                "client": "WEB",
            }
            resp = requests.get(url, params=params, timeout=8)
            data = resp.json()
            if not data.get('result') or not data['result'].get('data'):
                continue

            rows = data['result']['data']

            # 优先找年报，找不到用最新一期
            row = None
            for r in rows:
                if r.get('REPORT_TYPE') == '年报':
                    row = r
                    break
            if row is None:
                row = rows[0]

            net_profit = _to_float(row.get('PARENTNETPROFIT'))
            cashflow = _to_float(row.get('NETCASH_OPERATE_PK'))

            results[code] = {
                'roe': _to_float(row.get('ROEJQ')),
                'net_profit': net_profit,
                'revenue': _to_float(row.get('TOTALOPERATEREVE')),
                'revenue_yoy': _to_float(row.get('TOTALOPERATEREVETZ')),
                'profit_yoy': _to_float(row.get('PARENTNETPROFITTZ')),
                'debt_to_assets': _to_float(row.get('ZCFZL')),
                'cashflow': cashflow,
                'net_margin': _to_float(row.get('XSJLL')),
                'report_type': row.get('REPORT_TYPE', ''),
            }

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"    进度: {i+1}/{total} ({len(results)} 有效)")
            time.sleep(0.5)

    print(f"  ✅ 财务数据: {len(results)}/{total} 只")
    return results


def fetch_valuation_em(codes):
    """东财获取估值数据（PE/PB/市值）"""
    results = {}
    total = len(codes)
    print(f"  📥 获取 {total} 只估值数据...")
    for i, code in enumerate(codes):
        try:
            url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
            params = {
                "reportName": "RPT_VALUEANALYSIS_DET",
                "columns": "SECURITY_CODE,PE_TTM,PB_MRQ,TOTAL_MARKET_CAP",
                "filter": f'(SECURITY_CODE="{code}")',
                "pageSize": 1, "source": "WEB", "client": "WEB",
            }
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if data.get('result') and data['result'].get('data'):
                row = data['result']['data'][0]
                results[code] = {
                    'pe': _to_float(row.get('PE_TTM')),
                    'pb': _to_float(row.get('PB_MRQ')),
                    'market_cap': _to_float(row.get('TOTAL_MARKET_CAP')),
                }
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"    估值进度: {i+1}/{total} ({len(results)} 有效)")
            time.sleep(0.5)
    print(f"  ✅ 估值数据: {len(results)}/{total} 只")
    return results


# ============================================================
# 五层过滤器
# ============================================================

def filter_liquidity(code, realtime, hist):
    """
    第二层：流动性过滤
    近20日平均成交额 ≥ 5000万
    """
    reasons = []

    if hist is not None and len(hist) >= 20:
        avg_amount = float(hist['amount'].tail(20).mean())
        if avg_amount < 50_000_000:
            reasons.append(f"20日均额{avg_amount/1e6:.0f}M<5000万")
            return False, reasons, {'avg_amount_20d': avg_amount}
        return True, reasons, {'avg_amount_20d': avg_amount}

    rt = realtime.get(code, {})
    amount = rt.get('amount')
    if amount and amount < 10_000_000:
        reasons.append(f"成交额{amount/1e6:.0f}M过低")
        return False, reasons, {'amount_today': amount}

    return True, reasons, {}


def filter_fundamentals(code, fin, industry=''):
    """
    第三层：基本面质量过滤
    - ROE ≥ 8%（年报数据）
    - 净利润 > 0
    - 营收同比 > -10%
    - 净利润同比 > -20%
    - 资产负债率 ≤ 70%（金融行业豁免）
    - 经营现金流为正 或 现金流/净利润 ≥ 0.5
    """
    reasons = []
    passed = True

    # 金融行业（银行/保险/证券）负债率天然高，豁免负债率检查
    is_financial = industry in ('银行', '保险', '证券', '多元金融', '银行Ⅱ', '保险Ⅱ', '证券Ⅱ')

    roe = fin.get('roe')
    net_profit = fin.get('net_profit')
    rev_yoy = fin.get('revenue_yoy')
    profit_yoy = fin.get('profit_yoy')
    debt = fin.get('debt_to_assets')
    cashflow = fin.get('cashflow')

    # ROE（年报数据）
    if roe is None:
        reasons.append("ROE缺失")
        passed = False
    elif roe < 8:
        reasons.append(f"ROE {roe:.1f}%<8%")
        passed = False

    # 净利润
    if net_profit is None:
        reasons.append("净利润缺失")
        passed = False
    elif net_profit <= 0:
        reasons.append("净利润≤0")
        passed = False

    # 营收同比
    if rev_yoy is not None and rev_yoy < -10:
        reasons.append(f"营收同比{rev_yoy:.0f}%<-10%")
        passed = False

    # 净利润同比
    if profit_yoy is not None and profit_yoy < -20:
        reasons.append(f"利润同比{profit_yoy:.0f}%<-20%")
        passed = False

    # 资产负债率（金融行业豁免）
    if not is_financial and debt is not None and debt > 70:
        reasons.append(f"负债率{debt:.0f}%>70%")
        passed = False

    # 现金流：经营现金流为正，或现金流/净利润≥0.5
    if cashflow is not None and net_profit is not None and net_profit > 0:
        cf_ratio = cashflow / net_profit
        if cashflow < 0 and cf_ratio < 0.5:
            reasons.append(f"现金流/利润{cf_ratio:.2f}<0.5")
            passed = False
    elif cashflow is not None and cashflow < 0:
        reasons.append("经营现金流为负")
        passed = False

    return passed, reasons


def filter_valuation(code, val):
    """
    第四层：估值不过分
    - 0 < PE_TTM ≤ 80
    - 0 < PB ≤ 10
    """
    reasons = []
    passed = True

    pe = val.get('pe')
    pb = val.get('pb')

    if pe is None or pe <= 0:
        reasons.append("PE无效(亏损)")
        passed = False
    elif pe > 80:
        reasons.append(f"PE {pe:.0f}>80")
        passed = False

    if pb is None or pb <= 0:
        reasons.append("PB无效")
        passed = False
    elif pb > 10:
        reasons.append(f"PB {pb:.1f}>10")
        passed = False

    return passed, reasons


def filter_trend(code, price, hist):
    """
    第五层：趋势不能太差
    - 收盘价 ≥ MA120
    - 或 MA60 斜率没有明显向下（近20日斜率 > -1%）
    二选一即可
    """
    reasons = []

    if hist is None or len(hist) < 60:
        return True, reasons

    close = hist['close'].astype(float)
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ma120 = float(close.rolling(120).mean().iloc[-1]) if len(hist) >= 120 else None

    above_ma120 = ma120 is not None and price >= ma120

    ma60_slope = 0
    ma60_s = close.rolling(60).mean()
    if len(ma60_s) >= 20:
        ma60_slope = (ma60_s.iloc[-1] - ma60_s.iloc[-20]) / ma60_s.iloc[-20] * 100
    ma60_not_falling = ma60_slope > -1.0

    if above_ma120 or ma60_not_falling:
        return True, reasons

    reasons.append(f"价格<MA120且MA60斜率{ma60_slope:.1f}%")
    return False, reasons


# ============================================================
# 主流程
# ============================================================

def run_screener(pool_type='etf'):
    print(f"{'='*60}")
    print(f"🔍 优质股票候选池生成器 — {TODAY_DASH}")
    print(f"{'='*60}")
    t0 = time.time()

    # 第一层：基础排除
    pool = load_stock_pool(pool_type)
    codes = pool['code'].tolist()

    # 实时行情
    print("\n📥 获取实时行情...")
    realtime = fetch_realtime_sina(codes)
    print(f"  ✅ 实时行情: {len(realtime)} 只")

    # 价格 < 3元 排除
    codes_after_price = [c for c in codes if c in realtime and realtime[c]['price'] >= 3]
    print(f"  价格≥3元: {len(codes_after_price)} 只")

    # 历史K线
    print(f"\n📥 获取历史K线 ({len(codes_after_price)} 只)...")
    hist_data = fetch_hist_parallel(codes_after_price)

    # 财务数据
    codes_with_hist = [c for c in codes_after_price if c in hist_data]
    print(f"\n📥 获取财务数据...")
    financials = fetch_financial_em(codes_with_hist)

    # 估值数据
    fin_codes = list(financials.keys())
    print(f"\n📥 获取估值数据 ({len(fin_codes)} 只)...")
    valuations = fetch_valuation_em(fin_codes)
    print(f"  ✅ 估值数据: {len(valuations)} 只")

    # ========== 五层过滤 ==========
    print(f"\n🔍 执行五层过滤...")
    passed = []
    fail_stats = {'流动性': 0, '基本面': 0, '估值': 0, '趋势': 0}

    for code in codes_after_price:
        rt = realtime.get(code, {})
        price = rt.get('price', 0)
        if price <= 0:
            continue

        fin = financials.get(code, {})
        val = valuations.get(code, {})
        hist = hist_data.get(code)
        row = pool[pool['code'] == code]
        if row.empty:
            continue
        row = row.iloc[0]

        # 第二层：流动性
        ok, reasons, liq_info = filter_liquidity(code, realtime, hist)
        if not ok:
            fail_stats['流动性'] += 1
            continue

        # 第三层：基本面
        ok, reasons = filter_fundamentals(code, fin, row['industry'])
        if not ok:
            fail_stats['基本面'] += 1
            continue

        # 第四层：估值
        ok, reasons = filter_valuation(code, val)
        if not ok:
            fail_stats['估值'] += 1
            continue

        # 第五层：趋势
        ok, reasons = filter_trend(code, price, hist)
        if not ok:
            fail_stats['趋势'] += 1
            continue

        # 通过 — 计算风险标签
        risk_tags = []
        if fin.get('debt_to_assets') and fin['debt_to_assets'] > 60:
            risk_tags.append('高负债')
        if fin.get('profit_yoy') is not None and fin['profit_yoy'] < 0:
            risk_tags.append('利润下滑')
        if val.get('pe') and val['pe'] > 50:
            risk_tags.append('高估值')
        if fin.get('cashflow') and fin.get('net_profit') and fin['net_profit'] > 0:
            if fin['cashflow'] / fin['net_profit'] < 0.8:
                risk_tags.append('现金流偏弱')

        # MA120状态
        ma120_status = '-'
        if hist is not None and len(hist) >= 120:
            ma120 = float(hist['close'].astype(float).rolling(120).mean().iloc[-1])
            ma120_status = '上方' if price >= ma120 else '下方'

        passed.append({
            'code': code,
            'name': row['name'],
            'industry': row['industry'],
            'price': price,
            'market_cap': val.get('market_cap'),
            'avg_amount_20d': liq_info.get('avg_amount_20d'),
            'roe': fin.get('roe'),
            'revenue_yoy': fin.get('revenue_yoy'),
            'profit_yoy': fin.get('profit_yoy'),
            'debt_to_assets': fin.get('debt_to_assets'),
            'cashflow': fin.get('cashflow'),
            'net_profit': fin.get('net_profit'),
            'pe': val.get('pe'),
            'pb': val.get('pb'),
            'ma120_status': ma120_status,
            'risk_tags': '|'.join(risk_tags) if risk_tags else '-',
        })

    # ========== 输出 ==========
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = os.path.join(OUTPUT_DIR, f'quality_pool_{ts}.csv')
    report_path = os.path.join(OUTPUT_DIR, f'quality_pool_report_{ts}.md')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['代码', '名称', '行业', '价格', '市值(亿)', '20日均额(万)',
                     'ROE%', '营收同比%', '利润同比%', '负债率%', '现金流/利润',
                     'PE', 'PB', 'MA120状态', '风险标签'])
        for r in sorted(passed, key=lambda x: x['industry']):
            cf_ratio = '-'
            if r.get('cashflow') and r.get('net_profit') and r['net_profit'] > 0:
                cf_ratio = f"{r['cashflow']/r['net_profit']:.2f}"
            w.writerow([
                r['code'], r['name'], r['industry'],
                f"{r['price']:.2f}",
                f"{r['market_cap']/1e8:.0f}" if r.get('market_cap') else '-',
                f"{r['avg_amount_20d']/1e4:.0f}" if r.get('avg_amount_20d') else '-',
                f"{r['roe']:.1f}" if r.get('roe') else '-',
                f"{r['revenue_yoy']:.1f}" if r.get('revenue_yoy') is not None else '-',
                f"{r['profit_yoy']:.1f}" if r.get('profit_yoy') is not None else '-',
                f"{r['debt_to_assets']:.1f}" if r.get('debt_to_assets') else '-',
                cf_ratio,
                f"{r['pe']:.1f}" if r.get('pe') and r['pe'] > 0 else '-',
                f"{r['pb']:.2f}" if r.get('pb') and r['pb'] > 0 else '-',
                r['ma120_status'],
                r['risk_tags'],
            ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 优质股票候选池 {TODAY_DASH}\n\n")
        f.write(f"## 筛选统计\n")
        f.write(f"- 输入: {len(codes)} 只\n")
        f.write(f"- 价格≥3元: {len(codes_after_price)} 只\n")
        f.write(f"- 流动性淘汰: {fail_stats['流动性']} 只\n")
        f.write(f"- 基本面淘汰: {fail_stats['基本面']} 只\n")
        f.write(f"- 估值淘汰: {fail_stats['估值']} 只\n")
        f.write(f"- 趋势淘汰: {fail_stats['趋势']} 只\n")
        f.write(f"- **通过: {len(passed)} 只**\n\n")
        industry_count = {}
        for r in passed:
            ind = r['industry']
            industry_count[ind] = industry_count.get(ind, 0) + 1
        f.write(f"## 行业分布\n")
        for ind, cnt in sorted(industry_count.items(), key=lambda x: -x[1]):
            f.write(f"- {ind}: {cnt} 只\n")
        f.write(f"\n## 候选池明细（前50）\n\n")
        for i, r in enumerate(passed[:50], 1):
            f.write(f"{i}. **{r['code']} {r['name']}** [{r['industry']}] "
                    f"ROE:{r.get('roe',0):.1f}% PE:{r.get('pe',0):.0f} "
                    f"市值:{r.get('market_cap',0)/1e8:.0f}亿 "
                    f"{'⚠️'+r['risk_tags'] if r['risk_tags'] != '-' else '✅'}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 完成! 耗时 {elapsed:.0f}s")
    print(f"  通过: {len(passed)} / {len(codes_after_price)} 只")
    print(f"  淘汰: 流动性{fail_stats['流动性']} 基本面{fail_stats['基本面']} "
          f"估值{fail_stats['估值']} 趋势{fail_stats['趋势']}")
    print(f"  {csv_path}")
    print(f"{'='*60}")
    return passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='优质股票候选池生成器')
    parser.add_argument('--pool', choices=['etf', 'all'], default='etf',
                        help='股票池: etf=ETF持仓(快) all=全市场(慢)')
    args = parser.parse_args()
    run_screener(args.pool)
