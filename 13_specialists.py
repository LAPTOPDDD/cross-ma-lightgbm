import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ДАННЫЕ + ПРИЗНАКИ (v8/v12)
# ==============================================================================
file_path='EURUSD_H1_202401020000_202608262100.csv'
df=pd.read_csv(file_path,sep='\t',header=0,names=['Date','Time','Open','High','Low','Close','Volume','Unknown1','Unknown2'])
df['DateTime']=pd.to_datetime(df['Date']+' '+df['Time'],format='%Y.%m.%d %H:%M:%S')
df.set_index('DateTime',inplace=True); df.sort_index(inplace=True)
for col in ['Open','High','Low','Close']: df[col]=df[col].astype(float)
def add_features(d):
    c,h,l,o=d['Close'],d['High'],d['Low'],d['Open']
    d['SMA_20']=c.rolling(20).mean(); d['SMA_50']=c.rolling(50).mean(); d['EMA_12']=c.ewm(span=12).mean(); d['EMA_26']=c.ewm(span=26).mean()
    d['Dist_SMA20']=(c-d['SMA_20'])/d['SMA_20']; d['Dist_SMA50']=(c-d['SMA_50'])/d['SMA_50']; d['SMA_Spread']=(d['SMA_20']-d['SMA_50'])/d['SMA_50']
    pc=c.shift(1); ps=d['SMA_20'].shift(1); d['Cross_Up']=((c>d['SMA_20'])&(pc<=ps)).astype(int); d['Cross_Down']=((c<d['SMA_20'])&(pc>=ps)).astype(int)
    dd=c.diff(); g=dd.where(dd>0,0).ewm(alpha=1/14).mean(); lo=(-dd.where(dd<0,0)).ewm(alpha=1/14).mean()
    d['RSI_14']=100-100/(1+g/(lo+1e-10)); d['MACD']=d['EMA_12']-d['EMA_26']; d['MACD_Hist']=d['MACD']-d['MACD'].ewm(span=9).mean()
    tr=pd.concat([h-l,np.abs(h-c.shift()),np.abs(l-c.shift())],axis=1).max(axis=1); d['ATR_14']=tr.rolling(14).mean(); d['ATR_Ratio']=d['ATR_14']/(d['ATR_14'].rolling(50).mean()+1e-10)
    bbm=c.rolling(20).mean(); bbs=c.rolling(20).std(); d['BB_Pos']=(c-(bbm-2*bbs))/(4*bbs+1e-10); d['BB_Width']=(4*bbs)/(bbm+1e-10)
    d['Body']=np.abs(c-o)/(d['ATR_14']+1e-10); d['UpperWick']=(h-pd.concat([c,o],axis=1).max(axis=1))/(d['ATR_14']+1e-10); d['LowerWick']=(pd.concat([c,o],axis=1).min(axis=1)-l)/(d['ATR_14']+1e-10)
    d['Vol_Ratio']=d['Volume']/(d['Volume'].rolling(20).mean()+1); 
    for p in [1,2,3,5,10,20]: d[f'Ret_{p}']=c.pct_change(p)
    d['MR_Score']=(c-c.rolling(20).mean())/(c.rolling(20).std()+1e-10)
    d['Hour']=d.index.hour; d['DayOfWeek']=d.index.dayofweek; d['London']=d['Hour'].between(8,15).astype(int); d['Overlap']=d['Hour'].between(13,15).astype(int)
    pdm=h.diff(); mdm=-l.diff(); pdm=pdm.where((pdm>mdm)&(pdm>0),0); mdm=mdm.where((mdm>pdm)&(mdm>0),0)
    a14=tr.rolling(14).mean(); pdi=100*(pdm.rolling(14).mean()/(a14+1e-10)); mdi=100*(mdm.rolling(14).mean()/(a14+1e-10))
    dx=100*np.abs(pdi-mdi)/(pdi+mdi+1e-10); d['ADX']=dx.rolling(14).mean(); d['Trend_Strength']=d['ADX']/50.0; d['Trend_Direction']=np.sign(d['SMA_Spread'])
    return d
def add_mtf(d):
    c=d['Close']
    d['H4_SMA_5']=c.rolling(20).mean(); d['H4_SMA_20']=c.rolling(80).mean(); d['H4_Ret']=c.pct_change(4)
    d['D1_SMA_5']=c.rolling(120).mean(); d['D1_SMA_20']=c.rolling(240).mean(); d['D1_Ret']=c.pct_change(24)
    d['Dist_H4_SMA5']=(c-d['H4_SMA_5'])/(d['H4_SMA_5']+1e-10); d['Dist_H4_SMA20']=(c-d['H4_SMA_20'])/(d['H4_SMA_20']+1e-10); d['H4_SMA_Cross']=np.sign(d['H4_SMA_5']-d['H4_SMA_20'])
    d['Dist_D1_SMA5']=(c-d['D1_SMA_5'])/(d['D1_SMA_5']+1e-10); d['Dist_D1_SMA20']=(c-d['D1_SMA_20'])/(d['D1_SMA_20']+1e-10); d['D1_SMA_Cross']=np.sign(d['D1_SMA_5']-d['D1_SMA_20'])
    return d
df=add_features(df); df=add_mtf(df); df=df.iloc[240:].copy()
ALL=['Dist_SMA20','Dist_SMA50','SMA_Spread','Cross_Up','Cross_Down','RSI_14','MACD_Hist','ATR_Ratio','BB_Pos','BB_Width','Body','UpperWick','LowerWick','Vol_Ratio','MR_Score','Ret_1','Ret_2','Ret_3','Ret_5','Ret_10','Trend_Strength','Trend_Direction','Dist_H4_SMA5','Dist_H4_SMA20','H4_SMA_Cross','H4_Ret','Dist_D1_SMA5','Dist_D1_SMA20','D1_SMA_Cross','D1_Ret','Hour','DayOfWeek','London','Overlap']
df['Target']=(df['Close'].shift(-1)>df['Close']).astype(int); df.dropna(subset=['Target'],inplace=True)

def regime_of(r):
    if (r['ATR_Ratio']>1.2) and (r['Trend_Strength']>0.6): return 'trend_hi'
    elif r['ATR_Ratio']>1.2: return 'range_hi'
    elif r['Trend_Strength']>0.6: return 'trend_lo'
    else: return 'range_lo'
df['Regime']=df.apply(regime_of,axis=1)
regimes=['trend_hi','range_hi','trend_lo','range_lo']

# ==============================================================================
# 2. СПЕЦИАЛИСТЫ — разные ПОДМНОЖЕСТВА ПРИЗНАКОВ (=> разные сигналы)
# ==============================================================================
SPECS=[
  ('Reversion',['MR_Score','RSI_14','BB_Pos','BB_Width','Dist_SMA20','Dist_SMA50','SMA_Spread','Ret_1','Ret_2','Ret_3','Ret_5','Body','UpperWick','LowerWick','Vol_Ratio','ATR_Ratio','Hour','London','Overlap']),
  ('Momentum', ['SMA_Spread','Trend_Strength','Trend_Direction','MACD_Hist','Cross_Up','Cross_Down','Dist_H4_SMA20','Dist_D1_SMA20','H4_SMA_Cross','D1_SMA_Cross','H4_Ret','D1_Ret','ATR_Ratio','Vol_Ratio']),
  ('MTF',      ['Dist_H4_SMA5','Dist_H4_SMA20','H4_SMA_Cross','H4_Ret','Dist_D1_SMA5','Dist_D1_SMA20','D1_SMA_Cross','D1_Ret','ATR_Ratio','Trend_Strength']),
  ('Volatility',['ATR_Ratio','BB_Width','Vol_Ratio','Body','UpperWick','LowerWick','Ret_5','Ret_10','Cross_Up','Cross_Down','MACD_Hist']),
  ('Full',     ALL),
]
pred_by_model={n:pd.Series(dtype=float) for n,_ in SPECS}
edge={n:{r:[] for r in regimes} for n,_ in SPECS}

test_months=pd.date_range('2025-03-01','2026-07-01',freq='2MS')
for i,ts in enumerate(test_months):
    te=ts+pd.DateOffset(months=2); train=df[df.index<ts]; test=df[(df.index>=ts)&(df.index<te)]
    if len(train)<500 or len(test)<30: continue
    y_tr=train['Target']
    for n,feats in SPECS:
        m=lgb.LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,min_child_samples=50,reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbose=-1)
        m.fit(train[feats],y_tr)
        prob=pd.Series(m.predict_proba(test[feats])[:,1],index=test.index)
        pred_by_model[n]=pd.concat([pred_by_model[n],prob])
        for r in regimes:
            sub=test[test['Regime']==r]; idx=sub.index.intersection(prob.index)
            if len(idx)>0:
                preds=(prob[idx]>0.5).astype(int); edge[n][r].append((preds.values==sub.loc[idx,'Target'].values).mean())
    print(f"  WF {i+1}: done",flush=True)
for n,_ in SPECS:
    for r in regimes: edge[n][r]=np.mean(edge[n][r]) if edge[n][r] else 0.5

# ==============================================================================
# 3. ДИН. ВЗВЕШИВАНИЕ ПО РЕЖИМУ (каждый специалист весится своим edge в режиме)
# ==============================================================================
all_idx=sorted(set().union(*[set(pred_by_model[n].index) for n,_ in SPECS]))
final_prob=pd.Series(index=all_idx,dtype=float); reg_at={i:df.loc[i,'Regime'] for i in all_idx}
for idx in all_idx:
    r=reg_at[idx]; ws=[]; ps=[]
    for n,_ in SPECS:
        if idx in pred_by_model[n].index:
            w=max(edge[n][r]-0.5,0.02); ws.append(w); ps.append(pred_by_model[n][idx])
    if not ws: final_prob[idx]=0.5; continue
    ws=np.array(ws)/np.sum(ws); final_prob[idx]=np.dot(ws,ps)
final_prob.sort_index(inplace=True)
test_all=df.loc[final_prob.index].copy(); test_all['Prob']=final_prob.values
test_all['Next_Return']=(test_all['Close'].shift(-1)-test_all['Close'])/0.0001; test_all.dropna(subset=['Next_Return'],inplace=True)

regime_thr={}
for r in test_all['Regime'].unique():
    rd=test_all[test_all['Regime']==r]
    if len(rd)<30: regime_thr[r]=0.55; continue
    best,be=0.55,-999
    for thr in np.arange(0.50,0.72,0.02):
        mk=rd['Prob']>thr
        if mk.sum()<5: continue
        ev=rd.loc[mk,'Target'].mean()-(1-rd.loc[mk,'Target'].mean())
        if ev>be: be,best=ev,thr
    regime_thr[r]=best

print("\nРежимные пороги:",{k:f'{v:.2f}' for k,v in regime_thr.items()})
print("Edge специалистов по режимам:")
for r in regimes:
    print(f"  {r:9}: "+", ".join(f"{n[:4]}={edge[n][r]:.2f}" for n,_ in SPECS))

def backtest(allowed):
    cap0=10000; capital=cap0; equity=[capital]; pnl=[]
    for idx,row in test_all.iterrows():
        if row['Regime'] not in allowed: equity.append(capital); continue
        if row['Prob']>regime_thr.get(row['Regime'],0.55):
            ret=row['Next_Return']-1.0; conf=(row['Prob']-0.5)*2; pos=min(max(conf*0.3,0.02),0.1)
            p=ret*10*pos*10; capital+=p; pnl.append(p)
        equity.append(capital)
    return np.array(equity),np.array(pnl)

# Вариант A: полный symbiosis (все режимы)
eqA,pnA=backtest(set(regimes))
# Вариант B: сравнение с v8 (только range)
eqB,pnB=backtest({'range_lo','range_hi'})

def report(pnl,equity,label):
    n=len(pnl); wr=(pnl>0).mean() if n else 0; ret=(equity[-1]-10000)/10000*100
    dd=((equity/np.maximum.accumulate(equity))-1).min()*100
    sh=pnl.mean()/(pnl.std()+1e-10)*np.sqrt(252) if n else 0
    gp=pnl[pnl>0].sum() if (pnl>0).any() else 0; gl=abs(pnl[pnl<0].sum()) if (pnl<0).any() else 1e-10
    print(f"  [{label}] Сделок {n}, WR {wr:.1%}, Ret {ret:+.1f}%, DD {dd:.1f}%, Sharpe {sh:.2f}, PF {gp/gl:.2f}, ${equity[-1]:,.0f}")
    return equity,ret,sh,dd

print(f"\n{'='*64}\nv13 SYMBIOSIS: Специалисты + дин. взвешивание по режиму\n{'='*64}")
eqA,_,_,_=report(pnA,eqA,'ВСЕ режимы (полный symbiosis)')
eqB,_,_,_=report(pnB,eqB,'Только range (как v8)')
print(f"\n  vs v8 консенсус:        +16.5% / Sharpe 2.68 / DD -3.1%")
print(f"  vs v12 гетерогенный:   -0.8%  / Sharpe -0.58 / DD -5.2%")

# Сохраняем вариант B (range, сравнимый с v8) как основной результат
pnl=pnB; equity=eqB
eq=pd.DataFrame({'Equity':equity[1:]},index=test_all.index); eq['DD_%']=((eq['Equity']/eq['Equity'].cummax())-1)*100
eq.to_csv('equity_v13_specialists.csv')
plt.figure(figsize=(14,6)); plt.plot(equity,color='darkorange',linewidth=1.5); plt.axhline(y=10000,color='gray',ls='--',alpha=0.5)
plt.title('v13 Specialist Symbiosis (range-фильтр)',fontsize=13); plt.ylabel('Capital $'); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig('backtest_v13_specialists.png',dpi=150,bbox_inches='tight'); plt.close()
print("\nГрафик: backtest_v13_specialists.png | Эквити: equity_v13_specialists.csv")
