#!/usr/bin/env python3
"""
短期走势健康股票筛选器

定位: 筛出短期资金关注、趋势向上、流动性好、但还没有明显过热的股票。
输出: 通过筛选的股票名单（不排名，只输出通过/不通过）

六层过滤:
  1. 基础排除: ST/退市/停牌/上市<120天/北交所/科创板/价格<3元
  2. 流动性: 20日均成交额≥5000万
  3. 趋势: 收盘价>MA20 + (MA20斜率>0 或 MA20>MA60)
  4. 量价配合: 近5日均量≥近20日均量
  5. 位置不过热: 5日涨幅-5%~15% / 20日涨幅-10%~35% / 距MA20≤12% / RSI 40~75
  6. 强度确认: 至少满足2条（MA5>MA10 / MA10>MA20 / MACD DIF>DEA / MACD柱子改善 /
     距60日高点≤20% / 行业5日涨幅>0 / 个股20日涨幅强于沪深300）

用法:
  python screener_short_term.py
  python screener_short_term.py --pool etf
  python screener_short_term.py --pool all
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
# 数据获取
# ============================================================

def load_stock_pool(pool_type='etf'):
    """加载股票池"""
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

    # 基础排除
    df = df[~df['name'].str.contains('ST|退', na=False)]
    cutoff = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')  # ~120交易日
    df = df[df['list_date'] <= cutoff]
    df = df[~df['code'].str.startswith(('8', '4', '688'))]

    print(f"📊 基础排除后: {len(df)} 只")
    return df


def fetch_realtime_sina(codes, batch_size=800):
    """批量实时行情"""
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


def fetch_industry_perf(codes_with_industry, hist_data):
    """用历史数据计算行业近5日涨幅"""
    industry_changes = {}
    for code, industry in codes_with_industry:
        if code not in hist_data:
            continue
        df = hist_data[code]
        if len(df) < 6:
            continue
        close = df['close'].astype(float)
        chg_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
        industry_changes.setdefault(industry, []).append(chg_5d)

    return {k: sum(v)/len(v) for k, v in industry_changes.items() if len(v) >= 3}


def fetch_index_chg():
    """获取沪深300近20日涨幅（用于强度对比）"""
    try:
        df = fetch_hist_sina('000300', days=25)  # 沪深300
        if df is not None and len(df) >= 21:
            close = df['close'].astype(float)
            return (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100
    except Exception:
        pass
    return 0


# ============================================================
# 六层过滤器
# ============================================================

def filter_liquidity(hist, realtime_today):
    """第二层：流动性过滤"""
    reasons = []

    if hist is not None and len(hist) >= 20:
        avg_amount_20d = float(hist['amount'].tail(20).mean())
        if avg_amount_20d < 50_000_000:
            reasons.append(f"20日均额{avg_amount_20d/1e6:.0f}M<5000万")
            return False, reasons, {'avg_amount_20d': avg_amount_20d}
        return True, reasons, {'avg_amount_20d': avg_amount_20d}

    # 没有历史数据时用当日成交额
    amount = realtime_today.get('amount')
    if amount and amount < 50_000_000:
        reasons.append(f"当日额{amount/1e6:.0f}M过低")
        return False, reasons, {}

    return True, reasons, {}


def filter_trend(price, hist):
    """
    第三层：趋势过滤
    必须满足: 收盘价 > MA20
    加分条件(至少1条): MA20斜率>0 / MA20>MA60
    """
    reasons = []
    if hist is None or len(hist) < 20:
        return False, reasons.append("K线不足"), {}

    close = hist['close'].astype(float)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(hist) >= 60 else np.nan

    # 必须: 价格 > MA20
    if price <= ma20:
        reasons.append(f"价格{price:.2f}<MA20 {ma20:.2f}")
        return False, reasons, {}

    # 加分条件
    ma20_slope = 0
    ma20_s = close.rolling(20).mean()
    if len(ma20_s) >= 5:
        ma20_slope = (ma20_s.iloc[-1] - ma20_s.iloc[-5]) / ma20_s.iloc[-5] * 100

    ma20_above_ma60 = not pd.isna(ma60) and ma20 > ma60
    ma20_up = ma20_slope > 0

    if not (ma20_up or ma20_above_ma60):
        reasons.append(f"MA20斜率{ma20_slope:.1f}%且MA20<MA60")
        return False, reasons, {}

    return True, reasons, {'ma20': ma20, 'ma60': ma60, 'ma20_slope': ma20_slope}


def filter_volume(hist):
    """第四层：量价配合 — 近5日均量≥近20日均量"""
    if hist is None or len(hist) < 20:
        return True, [], {}

    volume = hist['vol'].astype(float) if 'vol' in hist.columns else hist['volume'].astype(float)
    avg5 = float(volume.tail(5).mean())
    avg20 = float(volume.tail(20).mean())

    if avg20 > 0 and avg5 / avg20 < 1.0:
        return False, [f"5日均量/20日均量={avg5/avg20:.2f}<1.0"], {}

    return True, [], {'vol_ratio': avg5/avg20 if avg20 > 0 else 1.0}


def filter_position(price, hist):
    """
    第五层：位置不过热
    - 近5日涨幅: -5% ~ 15%
    - 近20日涨幅: -10% ~ 35%
    - 收盘价距离MA20: ≤ 12%
    - RSI: 40 ~ 75
    """
    reasons = []
    if hist is None or len(hist) < 21:
        return True, reasons, {}

    close = hist['close'].astype(float)
    price_f = float(close.iloc[-1])

    # 涨幅
    gain_5d = (price_f / float(close.iloc[-6]) - 1) * 100 if len(hist) >= 6 else 0
    gain_20d = (price_f / float(close.iloc[-21]) - 1) * 100 if len(hist) >= 21 else 0

    if gain_5d > 15:
        reasons.append(f"5日涨{gain_5d:.0f}%>15%")
    if gain_5d < -5:
        reasons.append(f"5日涨{gain_5d:.0f}%<-5%")
    if gain_20d > 35:
        reasons.append(f"20日涨{gain_20d:.0f}%>35%")
    if gain_20d < -10:
        reasons.append(f"20日涨{gain_20d:.0f}%<-10%")

    # 距MA20
    ma20 = float(close.rolling(20).mean().iloc[-1])
    dist_ma20 = (price_f - ma20) / ma20 * 100
    if abs(dist_ma20) > 12:
        reasons.append(f"距MA20 {dist_ma20:.0f}%>12%")

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    if rsi > 75:
        reasons.append(f"RSI {rsi:.0f}>75")
    if rsi < 40:
        reasons.append(f"RSI {rsi:.0f}<40")

    if reasons:
        return False, reasons, {}

    return True, reasons, {
        'gain_5d': gain_5d, 'gain_20d': gain_20d,
        'dist_ma20': dist_ma20, 'rsi': rsi,
    }


def check_strength(price, hist, industry_chg, index_chg):
    """
    第六层：强度确认
    至少满足以下2条:
    1. MA5 > MA10
    2. MA10 > MA20
    3. MACD DIF > DEA
    4. MACD柱子近3日改善
    5. 收盘价距离60日高点 ≤ 20%
    6. 行业5日涨幅为正
    7. 个股20日涨幅强于沪深300
    """
    if hist is None or len(hist) < 20:
        return True, 0, []

    close = hist['close'].astype(float)
    price_f = float(close.iloc[-1])

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_hist = dif - dea

    # 60日高点
    high_60 = float(close.tail(60).max()) if len(hist) >= 60 else float(close.max())
    dist_high = (price_f - high_60) / high_60 * 100

    # 20日涨幅
    gain_20d = (price_f / float(close.iloc[-21]) - 1) * 100 if len(hist) >= 21 else 0

    # MACD柱子改善
    hist_vals = macd_hist.tail(3).tolist()
    macd_improve = len(hist_vals) >= 3 and hist_vals[-1] > hist_vals[-3]

    checks = {
        'MA5>MA10': ma5 > ma10,
        'MA10>MA20': ma10 > ma20,
        'MACD DIF>DEA': float(dif.iloc[-1]) > float(dea.iloc[-1]),
        'MACD柱子改善': macd_improve,
        '距60日高点≤20%': dist_high >= -20,
        '行业5日涨幅>0': industry_chg is not None and industry_chg > 0,
        '20日涨幅强于沪深300': gain_20d > index_chg,
    }

    met = sum(1 for v in checks.values() if v)
    met_list = [k for k, v in checks.items() if v]

    if met < 2:
        return False, met, met_list
    return True, met, met_list


# ============================================================
# 主流程
# ============================================================

def run_screener(pool_type='etf'):
    print(f"{'='*60}")
    print(f"📈 短期走势健康股票筛选器 — {TODAY_DASH}")
    print(f"{'='*60}")
    t0 = time.time()

    # 1. 加载股票池
    pool = load_stock_pool(pool_type)
    codes = pool['code'].tolist()

    # 2. 实时行情
    print("\n📥 获取实时行情...")
    realtime = fetch_realtime_sina(codes)
    print(f"  ✅ 实时行情: {len(realtime)} 只")

    # 价格过滤
    codes_price = [c for c in codes if c in realtime and realtime[c]['price'] >= 3]
    print(f"  价格≥3元: {len(codes_price)} 只")

    # 3. 历史K线
    print(f"\n📥 获取历史K线 ({len(codes_price)} 只)...")
    hist_data = fetch_hist_parallel(codes_price)

    # 4. 行业涨幅
    codes_with_ind = [(c, pool[pool['code']==c].iloc[0]['industry'])
                      for c in codes_price if c in pool[pool['code']==c].index]
    industry_perf = fetch_industry_perf(
        [(c, pool[pool['code']==c].iloc[0]['industry']) for c in codes_price if c in hist_data],
        hist_data
    )

    # 5. 沪深300基准
    print("📥 获取沪深300基准...")
    index_chg = fetch_index_chg()
    print(f"  沪深300近20日涨幅: {index_chg:.1f}%")

    # ========== 六层过滤 ==========
    print(f"\n🔍 执行六层过滤...")
    passed = []
    fail_stats = {'流动性': 0, '趋势': 0, '量价': 0, '位置': 0, '强度': 0}

    for code in codes_price:
        rt = realtime.get(code, {})
        price = rt.get('price', 0)
        if price <= 0:
            continue

        hist = hist_data.get(code)
        row = pool[pool['code'] == code]
        if row.empty:
            continue
        row = row.iloc[0]
        industry = row['industry']
        ind_chg = industry_perf.get(industry)

        # 第二层：流动性
        ok, reasons, liq = filter_liquidity(hist, rt)
        if not ok:
            fail_stats['流动性'] += 1
            continue

        # 第三层：趋势
        ok, reasons, trend = filter_trend(price, hist)
        if not ok:
            fail_stats['趋势'] += 1
            continue

        # 第四层：量价
        ok, reasons, vol = filter_volume(hist)
        if not ok:
            fail_stats['量价'] += 1
            continue

        # 第五层：位置
        ok, reasons, pos = filter_position(price, hist)
        if not ok:
            fail_stats['位置'] += 1
            continue

        # 第六层：强度
        ok, met_count, met_list = check_strength(price, hist, ind_chg, index_chg)
        if not ok:
            fail_stats['强度'] += 1
            continue

        # 通过 — 生成标签
        tags = []
        if pos.get('gain_5d') is not None and 0 < pos['gain_5d'] < 8:
            tags.append('温和上涨')
        if vol.get('vol_ratio') and vol['vol_ratio'] >= 1.2:
            tags.append('温和放量')
        if pos.get('dist_ma20') is not None and 0 < pos['dist_ma20'] < 5:
            tags.append('回踩MA20')
        if pos.get('rsi') is not None and 50 <= pos['rsi'] <= 65:
            tags.append('RSI健康')
        if trend.get('ma20_slope') and trend['ma20_slope'] > 0.3:
            tags.append('趋势向上')
        if ind_chg is not None and ind_chg > 1:
            tags.append('行业走强')

        risk_tags = []
        if pos.get('gain_5d') and pos['gain_5d'] > 10:
            risk_tags.append('涨幅偏高')
        if pos.get('dist_ma20') and pos['dist_ma20'] > 8:
            risk_tags.append('距MA20偏远')
        if pos.get('rsi') and pos['rsi'] > 70:
            risk_tags.append('RSI偏高')
        if liq.get('avg_amount_20d') and liq['avg_amount_20d'] < 100_000_000:
            risk_tags.append('成交额一般')

        # 距60日高点
        close = hist['close'].astype(float)
        high_60 = float(close.tail(60).max()) if len(hist) >= 60 else float(close.max())
        dist_high = (price - high_60) / high_60 * 100

        # MACD状态
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd_status = '金叉' if float(dif.iloc[-1]) > float(dea.iloc[-1]) else '死叉'

        passed.append({
            'code': code,
            'name': row['name'],
            'industry': industry,
            'price': price,
            'avg_amount_20d': liq.get('avg_amount_20d'),
            'gain_5d': pos.get('gain_5d', 0),
            'gain_20d': pos.get('gain_20d', 0),
            'dist_ma20': pos.get('dist_ma20', 0),
            'rsi': pos.get('rsi', 0),
            'ma20_slope': trend.get('ma20_slope', 0),
            'macd_status': macd_status,
            'dist_high_60': dist_high,
            'tags': '|'.join(tags) if tags else '-',
            'risk_tags': '|'.join(risk_tags) if risk_tags else '-',
        })

    # ========== 输出 ==========
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = os.path.join(OUTPUT_DIR, f'short_term_{ts}.csv')
    report_path = os.path.join(OUTPUT_DIR, f'short_term_report_{ts}.md')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['代码', '名称', '行业', '价格', '20日均额(万)', '5日涨幅%', '20日涨幅%',
                     '距MA20%', 'RSI', 'MA20斜率', 'MACD', '距60日高点%', '短期标签', '风险标签'])
        for r in sorted(passed, key=lambda x: x['industry']):
            w.writerow([
                r['code'], r['name'], r['industry'],
                f"{r['price']:.2f}",
                f"{r['avg_amount_20d']/1e4:.0f}" if r.get('avg_amount_20d') else '-',
                f"{r['gain_5d']:.1f}", f"{r['gain_20d']:.1f}",
                f"{r['dist_ma20']:.1f}", f"{r['rsi']:.0f}",
                f"{r['ma20_slope']:.2f}", r['macd_status'],
                f"{r['dist_high_60']:.1f}",
                r['tags'], r['risk_tags'],
            ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 短期走势健康股票 {TODAY_DASH}\n\n")
        f.write(f"## 筛选统计\n")
        f.write(f"- 输入: {len(codes)} 只\n")
        f.write(f"- 价格≥3元: {len(codes_price)} 只\n")
        f.write(f"- 流动性淘汰: {fail_stats['流动性']}\n")
        f.write(f"- 趋势淘汰: {fail_stats['趋势']}\n")
        f.write(f"- 量价淘汰: {fail_stats['量价']}\n")
        f.write(f"- 位置淘汰: {fail_stats['位置']}\n")
        f.write(f"- 强度淘汰: {fail_stats['强度']}\n")
        f.write(f"- **通过: {len(passed)} 只**\n\n")
        industry_count = {}
        for r in passed:
            industry_count[r['industry']] = industry_count.get(r['industry'], 0) + 1
        f.write(f"## 行业分布\n")
        for ind, cnt in sorted(industry_count.items(), key=lambda x: -x[1]):
            f.write(f"- {ind}: {cnt} 只\n")
        f.write(f"\n## 候选池明细\n\n")
        for i, r in enumerate(passed[:50], 1):
            f.write(f"{i}. **{r['code']} {r['name']}** [{r['industry']}] "
                    f"¥{r['price']:.2f} 5d:{r['gain_5d']:+.1f}% RSI:{r['rsi']:.0f} "
                    f"{r['macd_status']} "
                    f"{'✅'+r['tags'] if r['tags']!='-' else ''} "
                    f"{'⚠️'+r['risk_tags'] if r['risk_tags']!='-' else ''}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 完成! 耗时 {elapsed:.0f}s")
    print(f"  通过: {len(passed)} / {len(codes_price)} 只")
    print(f"  淘汰: 流动性{fail_stats['流动性']} 趋势{fail_stats['趋势']} "
          f"量价{fail_stats['量价']} 位置{fail_stats['位置']} 强度{fail_stats['强度']}")
    print(f"  {csv_path}")
    print(f"{'='*60}")
    return passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='短期走势健康股票筛选器')
    parser.add_argument('--pool', choices=['etf', 'all'], default='etf',
                        help='股票池: etf=ETF持仓(快) all=全市场(慢)')
    args = parser.parse_args()
    run_screener(args.pool)
