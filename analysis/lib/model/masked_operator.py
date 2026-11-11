"""Nutrient-masked, cross-fitted microbial response operator.

Copied from the analysis snapshot that produced the reported results. The only
change is the import of the nutrient masks, which resolves to the adjacent
masks.py instead of the original project tree.


No outcome or participant identity enters prediction features. Mean-response
and nutrient-interaction terms are independently removable and inspectable.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parent
from masks import EXPERT_REVISED_MAC,EXPERT_REVISED_LIPID
SEED=20261012

def splits(f,n=3):
    return list(GroupKFold(n_splits=n,shuffle=True,random_state=SEED).split(f,groups=f.family_group))

def weight(f):
    w=1/f.participant_id.map(f.participant_id.value_counts()).to_numpy(float)
    return w/w.sum()

def centered(a,ids):
    a=np.asarray(a);one=a.ndim==1
    if one:a=a[:,None]
    v=a-pd.DataFrame(a).groupby(np.asarray(ids),sort=False).transform('mean').to_numpy()
    return v.ravel() if one else v

def metrics(f,p):
    p=np.asarray(p);y=f.target.to_numpy(float);w=weight(f);err=y-p
    den=np.sum(w*(y-np.sum(w*y))**2)
    return dict(rmse=float(np.sqrt(w@err**2)),mae=float(w@abs(err)),
        profile_rmse=float(np.sqrt(w@centered(err,f.participant_id)**2)),
        r2=float(1-(w@err**2)/den),pearson=float(np.corrcoef(y,p)[0,1]))

class BaseFeatures:
    def food(self,f):
        a=f[self.food_cols].to_numpy(float)/self.units
        C=f.food__carbohydrate_g.to_numpy(float);F=f.food__fat_g.to_numpy(float)
        P=f.food__protein_g.to_numpy(float);D=f.food__fiber_g.to_numpy(float)
        return np.column_stack([a,np.log1p(a),C/(1+C+P+F),F/(1+C+P+F),D/(1+C),C/50/(1+F/20+P/20)])

    def fit(self,f):
        self.food_cols=sorted(c for c in f if c.startswith('food__'))
        unit={'food__energy_kcal':500,'food__carbohydrate_g':50,'food__protein_g':20,'food__fat_g':20,'food__fiber_g':5,'food__sugar_g':25}
        self.units=np.array([unit[c] for c in self.food_cols])
        self.host_cols=sorted(c for c in f if c.startswith(('person__','context__')))
        self.context_cols=[c for c in self.host_cols if c.startswith('context__')]
        self.context_reference=f[self.context_cols].median().to_dict()
        self.fi=SimpleImputer(strategy='median',keep_empty_features=True).fit(self.food(f))
        self.fs=StandardScaler().fit(self.fi.transform(self.food(f)))
        self.fs.scale_=np.maximum(self.fs.scale_,.2)
        self.hi=SimpleImputer(strategy='median',keep_empty_features=True).fit(f[self.host_cols])
        self.hs=StandardScaler().fit(self.hi.transform(f[self.host_cols]))
        return self

    def transform(self,f,interactions=True):
        x=self.fs.transform(self.fi.transform(self.food(f)))
        h=np.clip(self.hs.transform(self.hi.transform(f[self.host_cols])),-8,8)
        parts=[x,h]
        if interactions:parts.append(np.einsum('bi,bj->bij',x[:,:len(self.food_cols)],h).reshape(len(f),-1)/np.sqrt(max(h.shape[1],1)))
        return np.column_stack(parts)

class BaseModel:
    def __init__(self,kind='ridge',alpha=.1,leaves=7):self.kind=kind;self.alpha=alpha;self.leaves=leaves
    def fit(self,f):
        self.features=BaseFeatures().fit(f);x=self.features.transform(f,self.kind=='ridge')
        if self.kind=='ridge':self.model=Ridge(alpha=self.alpha)
        else:self.model=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=self.leaves,min_samples_leaf=30,learning_rate=.045,l2_regularization=10,early_stopping=False,random_state=SEED)
        w=weight(f);w=w if self.kind=='ridge' else w*len(f)
        self.model.fit(x,f.target,sample_weight=w)
        self.training_people=sorted(f.participant_id.unique());return self
    def predict(self,f):return self.model.predict(self.features.transform(f,self.kind=='ridge'))
    def standardized(self,f):
        f=f.copy()
        for k,v in self.features.context_reference.items():f[k]=v
        return self.predict(f)

BASES={'ridge_001':dict(kind='ridge',alpha=.01),'ridge_01':dict(kind='ridge',alpha=.1),'ridge_1':dict(kind='ridge',alpha=1.),'hgb3':dict(kind='tree',leaves=3),'hgb7':dict(kind='tree',leaves=7)}

def crossfit_base(f,config):
    p=np.empty(len(f))
    for a,b in splits(f):p[b]=BaseModel(**config).fit(f.iloc[a]).predict(f.iloc[b])
    return p

class MicroFeatures:
    def __init__(self,view='log',kernel='linear'):self.view=view;self.kernel=kernel
    def raw(self,f):
        a=f[self.micro_cols].to_numpy(float)
        if self.binary:
            # Presence is never converted into a taxon relative abundance.
            a=np.column_stack([a[:,ix].mean(1) for ix in self.genus_ix])
        elif self.view=='hellinger':a=np.sqrt(a/np.maximum(a.sum(1,keepdims=True),1e-12))
        else:
            a=np.log1p(a*1e6)
            if self.view=='clr':a=a-a.mean(1,keepdims=True)
        if self.function_cols:a=np.column_stack([a,np.log1p(f[self.function_cols].to_numpy(float))])
        return a
    def fit(self,f):
        self.micro_cols=sorted(c for c in f if c.startswith('micro__'))
        self.function_cols=sorted(c for c in f if c.startswith('function__'))
        u=f.drop_duplicates('participant_id');self.binary=np.isin(u[self.micro_cols].to_numpy(),[0,1]).all()
        self.genera=[c.removeprefix('micro__').split()[0].split('_')[0] for c in self.micro_cols]
        self.genus_ix=[[j for j,g in enumerate(self.genera) if g==name] for name in sorted(set(self.genera))]
        if self.binary:self.names=sorted(set(self.genera))
        else:self.names=self.genera
        a=self.raw(u);self.mean=a.mean(0);self.scale=a.std(0)
        # Avoid amplifying rare binary taxa or nearly invariant coordinates.
        self.scale=np.maximum(self.scale,.15 if self.binary else .03)
        z=(a-self.mean)/self.scale
        self.pca=None
        if z.shape[1]>40:
            self.pca=PCA(n_components=min(24,len(u)-3),svd_solver='full').fit(z)
            z=self.pca.transform(z)/np.sqrt(max(self.pca.explained_variance_[0],1e-8))
        self.centers=None
        if self.kernel=='rbf':
            rng=np.random.default_rng(SEED);self.centers=z[rng.choice(len(z),min(24,len(z)),replace=False)]
            r=self.rbf(z);self.rmean=r.mean(0)
        self.training_people=sorted(u.participant_id.unique());return self
    def rbf(self,z):return np.exp(-.5*((z[:,None]-self.centers[None])**2).mean(2))
    def transform(self,f):
        z=np.clip((self.raw(f)-self.mean)/self.scale,-8,8)
        if self.pca is not None:z=self.pca.transform(z)/np.sqrt(max(self.pca.explained_variance_[0],1e-8))
        z=z/np.sqrt(max(z.shape[1],1))
        if self.centers is not None:
            # Centers live in the unnormalised coordinate system.
            r=self.rbf(z*np.sqrt(z.shape[1]))-self.rmean
            z=np.column_stack([z,r/np.sqrt(r.shape[1])])
        return z
    def genus_signal(self,f):
        a=self.raw(f)[:,:len(self.names)]
        return (a-self.mean[:len(self.names)])/self.scale[:len(self.names)]

class OperatorFeatures:
    def __init__(self,view='log',structure='masked',kernel='linear',dose='linear',permute=False):
        self.view=view;self.structure=structure;self.kernel=kernel;self.dose=dose;self.permute=permute
    def exposures(self,f):
        e=f[self.exposure_cols].to_numpy(float)/self.exposure_scale
        return e if self.dose=='linear' else np.log1p(np.maximum(e,0))
    def fit(self,f):
        self.micro=MicroFeatures(self.view,self.kernel).fit(f)
        if self.structure=='unmasked':self.exposure_cols=sorted(c for c in f if c.startswith('food__'))
        else:self.exposure_cols=sorted(c for c in f if c.startswith('exposure__') and c.removeprefix('exposure__') in (EXPERT_REVISED_MAC|EXPERT_REVISED_LIPID))
        a=f.drop_duplicates('food_id')[self.exposure_cols].to_numpy(float)
        self.exposure_scale=np.maximum(np.sqrt((a*a).mean(0)),1e-5)
        self.nutrient_names=[c.removeprefix('exposure__') for c in self.exposure_cols]
        self.B=None
        if self.structure=='prior':
            B=pd.read_parquet(ROOT/'data/nutrient_directions.parquet')
            B=B.reindex(index=self.nutrient_names,columns=self.micro.names,fill_value=0).to_numpy(float)
            B=B-B.mean(1,keepdims=True)
            if self.permute:
                B=B[:,np.random.default_rng(SEED).permutation(B.shape[1])]
            B/=np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-10)
            self.B=B
            u=f.drop_duplicates('participant_id');z=self.micro.genus_signal(u)@self.B.T
            self.prior_mean=z.mean(0);self.prior_scale=np.maximum(z.std(0),.1)
        return self
    def transform(self,f):
        z=self.micro.transform(f);e=self.exposures(f)
        if self.B is None:inter=np.einsum('bi,bj->bij',e,z).reshape(len(f),-1)/np.sqrt(max(e.shape[1],1))
        else:
            cap=(self.micro.genus_signal(f)@self.B.T-self.prior_mean)/self.prior_scale
            inter=e*cap/np.sqrt(max(e.shape[1],1))
        return z,inter

class Operator:
    def __init__(self,features=None,alpha=.1,interaction_weight=1.,mean_weight=1.,profile_weight=0.):
        self.features=features;self.alpha=alpha;self.interaction_weight=interaction_weight;self.mean_weight=mean_weight;self.profile_weight=profile_weight
    def fit(self,f,residual,features=None):
        self.features=features or self.features.fit(f)
        z,t=self.features.transform(f);x=np.column_stack([z*self.mean_weight,t*self.interaction_weight])
        w=weight(f);self.center=x.T@w;self.offset=float(w@residual)
        x=x-self.center;y=residual-self.offset
        gram=x.T@(w[:,None]*x);rhs=x.T@(w*y)
        if self.profile_weight:
            xc=centered(x,f.participant_id);yc=centered(y,f.participant_id)
            gram+=self.profile_weight*xc.T@(w[:,None]*xc);rhs+=self.profile_weight*xc.T@(w*yc)
        self.coef=solve(gram+self.alpha*np.eye(x.shape[1]),rhs,assume_a='pos')
        # A residual offset is shared with the matched host baseline; it is not
        # attributed to microbiome and is deliberately omitted at inference.
        self.nz=z.shape[1];self.training_people=sorted(f.participant_id.unique());return self
    def parts(self,f):
        z,t=self.features.transform(f);a=z*self.mean_weight;b=t*self.interaction_weight
        mean=(a-self.center[:self.nz])@self.coef[:self.nz]
        interaction=(b-self.center[self.nz:])@self.coef[self.nz:]
        return mean,interaction
    def predict(self,f):return sum(self.parts(f))
    def nutrient_terms(self,f):
        z,t=self.features.transform(f);nt=len(self.features.exposure_cols)
        coef=self.coef[self.nz:]*self.interaction_weight
        if self.features.B is None:
            raw=(t*coef).reshape(len(f),nt,-1).sum(2)
        else:raw=t*coef
        return pd.DataFrame(raw,columns=self.features.nutrient_names,index=f.index)

class Combined:
    def __init__(self,base,operator=None):self.base=base;self.operator=operator
    def predict(self,f):return self.base.predict(f)+(0 if self.operator is None else self.operator.predict(f))
    def parts(self,f):return (np.zeros(len(f)),np.zeros(len(f))) if self.operator is None else self.operator.parts(f)
    def response_score(self,f):
        # Fixed context; microbial mean offsets cannot change food order.
        query=f.copy()
        for k,v in self.base.features.context_reference.items():query[k]=v
        r=self.base.predict(query)+self.parts(query)[1]
        return 100*expit(-r)
    def calibrated_fcs(self,f):
        ref=np.clip(f.fcs2.to_numpy(float),.01,99.99)/100
        query=f.copy()
        for k,v in self.base.features.context_reference.items():query[k]=v
        d=np.zeros(len(f)) if self.operator is None else self.operator.nutrient_terms(query).sum(axis=1).to_numpy()
        return 100*expit(np.log(ref/(1-ref))-d)

def feature_configs():
    return {
        'prior_log':dict(view='log',structure='prior'),
        'prior_clr':dict(view='clr',structure='prior'),
        'prior_hellinger':dict(view='hellinger',structure='prior'),
        'masked_log':dict(view='log',structure='masked'),
        'masked_clr':dict(view='clr',structure='masked'),
        'masked_hellinger':dict(view='hellinger',structure='masked'),
        'masked_rbf':dict(view='log',structure='masked',kernel='rbf'),
        'masked_logdose':dict(view='log',structure='masked',dose='log'),
        'unmasked_log':dict(view='log',structure='unmasked'),
    }
