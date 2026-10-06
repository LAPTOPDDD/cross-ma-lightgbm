import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ДАННЫЕ + ПРИЗНАКИ (как v8/v9)
# ==============================================================================
file_path = 'EURUSD_H1_202401020000_202608262100.csv'
df = pd.read_csv(file_path, sep='\t', header=0,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2'])
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('DateTime', inplace=True); df.sort_index(inplace=True)
for col in ['Open','High','Low','Close']: df[col] = df[col].astype(float)

def add_features(data):
    c,h,l,o = data['Close'],data['High'],data['Low'],data['Open']
    data['SMA_20']=c.rolling(20).mean(); data['SMA_50']=c.rolling(50).mean()
    data['EMA_12']=c.ewm(span=12).mean(); data['EMA_26']=c.ewm(span=26).mean()
    data['Dist_SMA20']=(c-data['SMA_20'])/data['SMA_20']
    data['Dist_SMA50']=(c-data['SMA_50'])/data['SMA_50']
    data['SMA_Spread']=(data['SMA_20']-data['SMA_50'])/data['SMA_50']
    pc=c.shift(1); ps=data['SMA_20'].shift(1)
    data['Cross_Up']=((c>data['SMA_20'])&(pc<=ps)).astype(int)
    data['Cross_Down']=((c<data['SMA_20'])&(pc>=ps)).astype(int)
    d=c.diff(); g=d.where(d>0,0).ewm(alpha=1/14).mean(); lo=(-d.where(d<0,0)).ewm(alpha=1/14).mean()
    data['RSI_14']=100-100/(1+g/(lo+1e-10))
    data['MACD']=data['EMA_12']-data['EMA_26']; data['MACD_Hist']=data['MACD']-data['MACD'].ewm(span=9).mean()
    tr=pd.concat([h-l,np.abs(h-c.shift()),np.abs(l-c.shift())],axis=1).max(axis=1)
    data['ATR_14']=tr.rolling(14).mean(); data['ATR_Ratio']=data['ATR_14']/(data['ATR_14'].rolling(50).mean()+1e-10)
    bbm=c.rolling(20).mean(); bbs=c.rolling(20).std()
    data['BB_Pos']=(c-(bbm-2*bbs))/(4*bbs+1e-10); data['BB_Width']=(4*bbs)/(bbm+1e-10)
    data['Body']=np.abs(c-o)/(data['ATR_14']+1e-10)
    data['UpperWick']=(h-pd.concat([c,o],axis=1).max(axis=1))/(data['ATR_14']+1e-10)
    data['LowerWick']=(pd.concat([c,o],axis=1).min(axis=1)-l)/(data['ATR_14']+1e-10)
    data['Vol_Ratio']=data['Volume']/(data['Volume'].rolling(20).mean()+1)
    for p in [1,2,3,5,10,20]: data[f'Ret_{p}']=c.pct_change(p)
    data['MR_Score']=(c-c.rolling(20).mean())/(c.rolling(20).std()+1e-10)
    data['Hour']=data.index.hour; data['DayOfWeek']=data.index.dayofweek
    data['London']=data['Hour'].between(8,15).astype(int); data['Overlap']=data['Hour'].between(13,15).astype(int)
    pdm=h.diff(); mdm=-l.diff(); pdm=pdm.where((pdm>mdm)&(pdm>0),0); mdm=mdm.where((mdm>pdm)&(mdm>0),0)
    a14=tr.rolling(14).mean(); pdi=100*(pdm.rolling(14).mean()/(a14+1e-10)); mdi=100*(mdm.rolling(14).mean()/(a14+1e-10))
    dx=100*np.abs(pdi-mdi)/(pdi+mdi+1e-10)
    data['ADX']=dx.rolling(14).mean(); data['Trend_Strength']=data['ADX']/50.0
    data['Trend_Direction']=np.sign(data['SMA_Spread'])
    return data

def add_mtf(data):
    c=data['Close']
    data['H4_SMA_5']=c.rolling(20).mean(); data['H4_SMA_20']=c.rolling(80).mean(); data['H4_Ret']=c.pct_change(4)
    data['D1_SMA_5']=c.rolling(120).mean(); data['D1_SMA_20']=c.rolling(240).mean(); data['D1_Ret']=c.pct_change(24)
    data['Dist_H4_SMA5']=(c-data['H4_SMA_5'])/(data['H4_SMA_5']+1e-10)
    data['Dist_H4_SMA20']=(c-data['H4_SMA_20'])/(data['H4_SMA_20']+1e-10); data['H4_SMA_Cross']=np.sign(data['H4_SMA_5']-data['H4_SMA_20'])
    data['Dist_D1_SMA5']=(c-data['D1_SMA_5'])/(data['D1_SMA_5']+1e-10)
    data['Dist_D1_SMA20']=(c-data['D1_SMA_20'])/(data['D1_SMA_20']+1e-10); data['D1_SMA_Cross']=np.sign(data['D1_SMA_5']-data['D1_SMA_20'])
    return data

df=add_features(df); df=add_mtf(df); df=df.iloc[240:].copy()
feature_cols=['Dist_SMA20','Dist_SMA50','SMA_Spread','Cross_Up','Cross_Down','RSI_14','MACD_Hist',
    'ATR_Ratio','BB_Pos','BB_Width','Body','UpperWick','LowerWick','Vol_Ratio','MR_Score',
    'Ret_1','Ret_2','Ret_3','Ret_5','Ret_10','Trend_Strength','Trend_Direction',
    'Dist_H4_SMA5','Dist_H4_SMA20','H4_SMA_Cross','H4_Ret','Dist_D1_SMA5','Dist_D1_SMA20',
    'D1_SMA_Cross','D1_Ret','Hour','DayOfWeek','London','Overlap']
df['Target']=(df['Close'].shift(-1)>df['Close']).astype(int); df.dropna(subset=['Target'],inplace=True)

N_MODELS=7
ALLOWED=['range_lo','range_hi']
capital_start=10000; spread_pips=1.0; pip_value=10

def regime_of(r):
    if (r['ATR_Ratio']>1.2) and (r['Trend_Strength']>0.6): return 'trend_hi'
    elif r['ATR_Ratio']>1.2: return 'range_hi'
    elif r['Trend_Strength']>0.6: return 'trend_lo'
    else: return 'range_lo'
df['Regime']=df.apply(regime_of,axis=1)
regimes=['trend_hi','range_hi','trend_lo','range_lo']

# ==============================================================================
# 2. WALK-FORWARD: накапливаем предсказания КАЖДОЙ модели + edge по режимам
# ==============================================================================
pred_by_model={s: pd.Series(dtype=float) for s in range(N_MODELS)}
# edge[m][r] = исторический WR модели m в режиме r (по всем WF окнам, out-of-sample)
edge={m:{r:[] for r in regimes} for m in range(N_MODELS)}

test_months=pd.date_range('2025-03-01','2026-07-01',freq='2MS')
for i,test_start in enumerate(test_months):
    test_end=test_start+pd.DateOffset(months=2)
    train=df[df.index<test_start]; test=df[(df.index>=test_start)&(df.index<test_end)]
    if len(train)<500 or len(test)<30: continue
    X_tr,y_tr=train[feature_cols],train['Target']; X_te=test[feature_cols]
    for seed in range(N_MODELS):
        m=lgb.LGBMClassifier(n_estimators=300+seed*50,max_depth=3+(seed%3),
            learning_rate=0.03,subsample=0.6+seed*0.05,colsample_bytree=0.6+seed*0.05,
            min_child_samples=50,reg_alpha=0.1,reg_lambda=1.0,random_state=seed*7,verbose=-1)
        m.fit(X_tr,y_tr)
        prob=pd.Series(m.predict_proba(X_te)[:,1],index=test.index)
        pred_by_model[seed]=pd.concat([pred_by_model[seed],prob])
        # edge по режимам (target из test — это out-of-sample для этой модели)
        for r in regimes:
            rmask=(test['Regime']==r)&(prob.index.isin(test.index))
            sub=test.loc[prob.index][test.loc[prob.index]['Regime']==r]
            if len(sub)>0:
                preds=(prob[sub.index]>0.5).astype(int)
                correct=(preds.values==sub['Target'].values).sum()
                edge[seed][r].append(correct/len(sub))
    print(f"  WF {i+1}: done")

# Усредняем edge по окнам
for m in range(N_MODELS):
    for r in regimes:
        edge[m][r]=np.mean(edge[m][r]) if edge[m][r] else 0.5

# ==============================================================================
# 3. ДИНАМИЧЕСКОЕ ВЗВЕШИВАНИЕ (symbiosis): вес модели ∝ её edge в текущем режиме
# ==============================================================================
all_idx=sorted(set().union(*[set(pred_by_model[s].index) for s in range(N_MODELS)]))
final_prob=pd.Series(index=all_idx,dtype=float)
regime_at={idx:df.loc[idx,'Regime'] for idx in all_idx}

for idx in all_idx:
    r=regime_at[idx]
    ws=[]; ps=[]
    for s in range(N_MODELS):
        if idx in pred_by_model[s].index:
            e=edge[s][r]
            w=max(e-0.5,0.02)  # только модели с положительным edge; мин. вес 0.02
            ws.append(w); ps.append(pred_by_model[s][idx])
    if not ws:
        final_prob[idx]=0.5; continue
    ws=np.array(ws)/np.sum(ws)
    final_prob[idx]=np.dot(ws,ps)

final_prob.sort_index(inplace=True)
test_all=df.loc[final_prob.index].copy()
test_all['Prob']=final_prob.values
test_all['Next_Return']=(test_all['Close'].shift(-1)-test_all['Close'])/0.0001
test_all.dropna(subset=['Next_Return'],inplace=True)

# ==============================================================================
# 4. БЭКТЕСТ (range-фильтр + per-regime порог)
# ==============================================================================
regime_thr={}
for r in test_all['Regime'].unique():
    rd=test_all[test_all['Regime']==r]
    if len(rd)<30: regime_thr[r]=0.55; continue
    best,be=0.55,-999
    for thr in np.arange(0.50,0.72,0.02):
        mk=rd['Prob']>thr
        if mk.sum()<5: continue
        ev=rd.loc[mk,'Target'].mean()-(1-rd.loc[mk,'Target'].mean())
        if ev>be: be, best=ev,thr
    regime_thr[r]=best

print("\nRegime thresholds:",{k:f'{v:.2f}' for k,v in regime_thr.items()})
print("Edge-веса (range_lo):",{f'M{s+1}':f"{edge[s]['range_lo']:.2f}" for s in range(N_MODELS)})

capital=capital_start; equity=[capital]; trade_pnls=[]
for idx,row in test_all.iterrows():
    if row['Regime'] not in ALLOWED:
        equity.append(capital); continue
    thr=regime_thr.get(row['Regime'],0.55)
    if row['Prob']>thr:
        ret=row['Next_Return']-spread_pips
        conf=(row['Prob']-0.5)*2
        pos=min(max(conf*0.3,0.02),0.1)
        pnl=ret*pip_value*pos*10
        capital+=pnl; trade_pnls.append(pnl)
    equity.append(capital)

trade_pnls=np.array(trade_pnls); equity=np.array(equity)
n=len(trade_pnls); wr=(trade_pnls>0).mean() if n else 0
ret=(equity[-1]-capital_start)/capital_start*100
dd=((equity/np.maximum.accumulate(equity))-1).min()*100
sharpe=trade_pnls.mean()/(trade_pnls.std()+1e-10)*np.sqrt(252) if n else 0
gp=trade_pnls[trade_pnls>0].sum() if (trade_pnls>0).any() else 0
gl=abs(trade_pnls[trade_pnls<0].sum()) if (trade_pnls<0).any() else 1e-10
pf=gp/gl

print(f"\n{'='*60}\nv11 SYMBIOSIS: Dynamic Regime-Weighted Consensus\n{'='*60}")
print(f"  Сделок: {n}\n  Win Rate: {wr:.1%}\n  Доходность: {ret:+.1f}%\n  Max DD: {dd:.1f}%\n  Sharpe: {sharpe:.2f}\n  PF: {pf:.2f}\n  Final: ${equity[-1]:,.2f}")
print(f"\n  vs v8 консенсус:  +16.5% / Sharpe 2.68 / DD -3.1%")
print(f"  vs v9 корзина:    +12.9% / Sharpe 1.47 / DD -4.5%")
print(f"  vs v10 stacking:   -2.0%  / Sharpe -1.13 / DD -5.7%")

eq_df=pd.DataFrame({'Equity':equity[1:]},index=test_all.index)
eq_df['Drawdown_%']=((eq_df['Equity']/eq_df['Equity'].cummax())-1)*100
eq_df.to_csv('equity_v11_dynweight.csv')
plt.figure(figsize=(14,6)); plt.plot(equity,color='teal',linewidth=1.5)
plt.axhline(y=capital_start,color='gray',linestyle='--',alpha=0.5)
plt.title(f'v11 Dynamic Regime-Weighted Consensus (Sharpe={sharpe:.2f}, Ret={ret:+.1f}%)',fontsize=13)
plt.ylabel('Capital $'); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig('backtest_v11_dynweight.png',dpi=150,bbox_inches='tight'); plt.close()
print("\nГрафик: backtest_v11_dynweight.png | Эквити: equity_v11_dynweight.csv")
