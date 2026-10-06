import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')

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
    d['Vol_Ratio']=d['Volume']/(d['Volume'].rolling(20).mean()+1)
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
feat=['Dist_SMA20','Dist_SMA50','SMA_Spread','Cross_Up','Cross_Down','RSI_14','MACD_Hist','ATR_Ratio','BB_Pos','BB_Width','Body','UpperWick','LowerWick','Vol_Ratio','MR_Score','Ret_1','Ret_2','Ret_3','Ret_5','Ret_10','Trend_Strength','Trend_Direction','Dist_H4_SMA5','Dist_H4_SMA20','H4_SMA_Cross','H4_Ret','Dist_D1_SMA5','Dist_D1_SMA20','D1_SMA_Cross','D1_Ret','Hour','DayOfWeek','London','Overlap']
df['Target']=(df['Close'].shift(-1)>df['Close']).astype(int); df.dropna(subset=['Target'],inplace=True)
def regime_of(r):
    if (r['ATR_Ratio']>1.2) and (r['Trend_Strength']>0.6): return 'trend_hi'
    elif r['ATR_Ratio']>1.2: return 'range_hi'
    elif r['Trend_Strength']>0.6: return 'trend_lo'
    else: return 'range_lo'
df['Regime']=df.apply(regime_of,axis=1)
ALLOWED={'range_lo','range_hi'}; test_months=pd.date_range('2025-03-01','2026-07-01',freq='2MS')

def run(n_models):
    all_test=[]
    for ts in test_months:
        te=ts+pd.DateOffset(months=2); train=df[df.index<ts]; test=df[(df.index>=ts)&(df.index<te)]
        if len(train)<500 or len(test)<30: continue
        probs=[]
        for s in range(n_models):
            m=lgb.LGBMClassifier(n_estimators=300+s*30,max_depth=3+(s%4),learning_rate=0.03,
                subsample=0.6+(s%5)*0.05,colsample_bytree=0.6+(s%5)*0.05,min_child_samples=50,
                reg_alpha=0.1,reg_lambda=1.0,random_state=s*7,verbose=-1)
            m.fit(train[feat],train['Target']); probs.append(m.predict_proba(test[feat])[:,1])
        res=test.copy(); res['Prob']=np.mean(probs,axis=0); all_test.append(res)
    ta=pd.concat(all_test); ta=ta[~ta.index.duplicated(keep='first')]; ta.sort_index(inplace=True)
    ta['Next_Return']=(ta['Close'].shift(-1)-ta['Close'])/0.0001; ta.dropna(subset=['Next_Return'],inplace=True)
    thr={}
    for r in ta['Regime'].unique():
        rd=ta[ta['Regime']==r]
        if len(rd)<30: thr[r]=0.55; continue
        bt,be=0.55,-999
        for t in np.arange(0.50,0.66,0.02):
            mk=rd['Prob']>t
            if mk.sum()<5: continue
            ev=rd.loc[mk,'Target'].mean()-(1-rd.loc[mk,'Target'].mean())
            if ev>be: be,bt=ev,t
        thr[r]=bt
    cap=10000; eq=[cap]; pn=[]
    for idx,row in ta.iterrows():
        if row['Regime'] not in ALLOWED: eq.append(cap); continue
        if row['Prob']>thr.get(row['Regime'],0.55):
            ret=row['Next_Return']-1.0; conf=(row['Prob']-0.5)*2; pos=min(max(conf*0.3,0.02),0.1)
            p=ret*10*pos*10; cap+=p; pn.append(p)
        eq.append(cap)
    pn=np.array(pn); eq=np.array(eq); n=len(pn)
    ret=(eq[-1]-10000)/10000*100; dd=((eq/np.maximum.accumulate(eq))-1).min()*100
    sh=pn.mean()/(pn.std()+1e-10)*np.sqrt(252) if n else 0
    gp=pn[pn>0].sum() if (pn>0).any() else 0; gl=abs(pn[pn<0].sum()) if (pn<0).any() else 1e-10
    return n,(pn>0).mean(),ret,dd,sh,gp/gl,eq[-1]

print(f"{'N':>3} | {'Trades':>6} | {'WR':>6} | {'Ret%':>7} | {'DD%':>6} | {'Sharpe':>7} | {'PF':>5} | {'Final$':>9}")
for N in [1,3,7,14,21,28,40]:
    n,wr,ret,dd,sh,pf,fin=run(N)
    print(f"{N:>3} | {n:>6} | {wr:>5.1%} | {ret:>+6.1f} | {dd:>5.1f} | {sh:>7.2f} | {pf:>5.2f} | {fin:>9,.0f}")
