"""Microbiome-conditioned Bayesian calibration of masked nutrient sensitivities."""
from itertools import product
import numpy as np
import pandas as pd
from scipy.linalg import solve
from masked_operator import *
from functional_operator import FunctionalMicro

class ConditionalAdapter:
    def __init__(self,base,food='masked',trait='uniform',bandwidth=1.,mix=.5,mean_mix=0.,noise=1.,dose='linear'):
        self.base=base;self.food=food;self.trait=trait;self.bandwidth=bandwidth
        self.mix=mix;self.mean_mix=mean_mix;self.noise=noise
        self.dose=dose
    def design(self,f):
        a=f[self.columns].to_numpy(float)/self.unit
        if getattr(self,'dose','linear')=='saturating':a=a/(1+a)
        elif getattr(self,'dose','linear')=='log':a=np.log1p(a)
        return np.column_stack([np.ones(len(f)),a])
    def trait_matrix(self,f):
        a=np.clip(self.hs.transform(self.hi.transform(f[self.hcols])),-5,5)/np.sqrt(len(self.hcols))
        if self.trait=='host':return a
        if self.trait=='function':return self.functions.transform(f)
        if self.trait=='taxa':return self.micro.transform(f)
        if self.trait=='host_function':return np.column_stack([a,self.functions.transform(f)])/np.sqrt(2)
        if self.trait=='host_taxa':return np.column_stack([a,self.micro.transform(f)])/np.sqrt(2)
        return np.zeros((len(f),1))
    def fit(self,f,residual):
        self.columns=['food__fiber_g','food__fat_g'] if self.food=='masked' else ['food__carbohydrate_g','food__protein_g','food__fat_g','food__fiber_g']
        self.unit=np.array([5.,20.] if self.food=='masked' else [50.,20.,20.,5.])
        u=f.drop_duplicates('participant_id');self.training_people=sorted(u.participant_id.unique())
        self.hcols=sorted(c for c in f if c.startswith('person__'))
        self.hi=SimpleImputer(strategy='median',keep_empty_features=True).fit(u[self.hcols])
        self.hs=StandardScaler().fit(self.hi.transform(u[self.hcols]))
        self.functions=FunctionalMicro().fit(f)
        self.micro=MicroFeatures('log').fit(f[[c for c in f if not c.startswith('function__')]])
        self.people=u.participant_id.to_numpy();self.traits=self.trait_matrix(u)
        x=self.design(f);d=x.shape[1];sigma=np.eye(d)*.3;mu=np.zeros(d)
        s2=max(np.average(centered(residual,f.participant_id)**2,weights=weight(f)),.01)
        records=[]
        for pid in self.people:
            ix=np.flatnonzero(f.participant_id.to_numpy()==pid);a=x[ix];r=residual[ix]
            records.append((a,r,a.T@a,a.T@r))
        for it in range(25):
            inv=np.linalg.inv(sigma);betas=[];covs=[];variances=[]
            for a,r,gram,rhs in records:
                cov=np.linalg.inv(inv+gram/s2);beta=cov@(inv@mu+rhs/s2)
                betas.append(beta);covs.append(cov);variances.append(np.mean((r-a@beta)**2)+np.trace(a@cov@a.T)/len(r))
            betas=np.asarray(betas);covs=np.asarray(covs);mu=betas.mean(0)
            diff=betas-mu;emp=diff.T@diff/len(diff)+covs.mean(0)
            sigma=.8*emp+.2*np.diag(np.diag(emp))+np.eye(d)*1e-5
            s2=max(np.mean(variances),.01)
        self.mu=mu;self.sigma=sigma;self.betas=betas;self.covs=covs;self.s2=s2;return self
    def prior(self,person):
        if self.trait=='uniform':return self.mu.copy(),self.sigma.copy()
        z=self.trait_matrix(person)[0];d=((self.traits-z)**2).sum(1)
        a=-d/(2*self.bandwidth**2);a-=a.max();w=np.exp(a);w/=w.sum()
        m=w@self.betas;diff=self.betas-m
        cov=np.einsum('n,ni,nj->ij',w,diff,diff)+np.einsum('n,nij->ij',w,self.covs)
        mean=(1-self.mean_mix)*self.mu+self.mean_mix*m
        cov=(1-self.mix)*self.sigma+self.mix*cov
        return mean,cov+np.eye(len(mean))*1e-6
    def predict_history(self,f,k=3,query_after=5):
        rows=[]
        for pid,g in f.sort_values('timestamp').groupby('participant_id',sort=False):
            if len(g)<=query_after:continue
            hist=g.iloc[:k];query=g.iloc[query_after:]
            assert hist.timestamp.max()<query.timestamp.min()
            mu,cov=self.prior(g.iloc[:1]);inv=np.linalg.inv(cov)
            a=self.design(hist);r=hist.target.to_numpy()-self.base.predict(hist)
            postcov=np.linalg.inv(inv+a.T@a/(self.s2*self.noise))
            post=postcov@(inv@mu+a.T@r/(self.s2*self.noise))
            x=self.design(query);p=self.base.predict(query)+x@post
            q=query[['meal_event_id','participant_id','family_group','food_id','meal_id','timestamp','target']].copy()
            q['prediction']=p;q['score']=100*expit(-(self.base.standardized(query)+x[:,1:]@post[1:]))
            q['personal_intercept']=post[0];q['nutrient_adjustment']=x[:,1:]@post[1:];q['history_k']=k
            rows.append(q)
        return pd.concat(rows,ignore_index=True)

def adapter_configs():
    configs={}
    for food in ['masked','all']:
        for noise in [.3,1.,3.]:
            configs[f'{food}_uniform_noise{noise}']=dict(food=food,trait='uniform',noise=noise)
        for trait in ['host','function','taxa','host_function','host_taxa']:
            for bandwidth,mean_mix in product([.3,1.,3.],[0.,.5]):
                key=f'{food}_{trait}_bw{bandwidth}_mean{mean_mix}'
                configs[key]=dict(food=food,trait=trait,bandwidth=bandwidth,mean_mix=mean_mix,mix=.5,noise=1.)
    return configs
