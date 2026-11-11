"""Observed ordinal microbial function scores, with nutrient-channel routing."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from masked_operator import *

MAC_FUNCTIONS={'Butyrate Production Pathways','Gas Production','Methane Gas Production Pathways',
    'Sulfide Gas Production Pathways','Digestive Efficiency','Gut Lining Health',
    'Active Microbial Diversity','Inflammatory Activity'}
LIPID_FUNCTIONS={'Bile Acid Metabolism Pathways','TMA Production Pathways','LPS Biosynthesis Pathways',
    'Inflammatory Activity','Gut Lining Health','Digestive Efficiency'}

class FunctionalMicro:
    def __init__(self,view='ordinal'):self.view=view
    def raw(self,f):
        a=self.imputer.transform(f[self.function_cols])
        if self.view=='onehot':return np.column_stack([a==1,a==3]).astype(float)
        return a
    def fit(self,f):
        u=f.drop_duplicates('participant_id');self.function_cols=sorted(c for c in f if c.startswith('function__'))
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(u[self.function_cols])
        raw=self.raw(u);self.mean=raw.mean(0);self.scale=np.maximum(raw.std(0),.25)
        self.names=[c.removeprefix('function__') for c in self.function_cols]
        if self.view=='onehot':self.names=self.names+self.names
        self.training_people=sorted(u.participant_id.unique());return self
    def transform(self,f):return (self.raw(f)-self.mean)/self.scale/np.sqrt(len(self.names))

class FunctionalFeatures(OperatorFeatures):
    def fit(self,f):
        self.micro=FunctionalMicro(self.view).fit(f)
        if self.structure=='unmasked':self.exposure_cols=sorted(c for c in f if c.startswith('food__'))
        else:self.exposure_cols=sorted(c for c in f if c.startswith('exposure__') and c.removeprefix('exposure__') in (EXPERT_REVISED_MAC|EXPERT_REVISED_LIPID))
        self.nutrient_names=[c.removeprefix('exposure__') for c in self.exposure_cols]
        a=f.drop_duplicates('food_id')[self.exposure_cols].to_numpy(float)
        self.exposure_scale=np.maximum(np.sqrt((a*a).mean(0)),1e-5)
        self.routing=np.ones((len(self.exposure_cols),len(self.micro.names)))
        if self.structure=='prior':
            for i,nutrient in enumerate(self.nutrient_names):
                allowed=MAC_FUNCTIONS if nutrient in EXPERT_REVISED_MAC else LIPID_FUNCTIONS
                self.routing[i]=[n in allowed for n in self.micro.names]
            if self.permute:self.routing=self.routing[:,np.random.default_rng(SEED).permutation(self.routing.shape[1])]
        self.B=None;return self
    def transform(self,f):
        z=self.micro.transform(f);e=self.exposures(f)
        t=np.einsum('bi,bj->bij',e,z)*self.routing[None]
        return z,t.reshape(len(f),-1)/np.sqrt(e.shape[1])

def function_configs():
    return {'prior_ordinal':dict(view='ordinal',structure='prior'),
        'prior_onehot':dict(view='onehot',structure='prior'),
        'masked_ordinal':dict(view='ordinal',structure='masked'),
        'masked_onehot':dict(view='onehot',structure='masked'),
        'unmasked_log':dict(view='ordinal',structure='unmasked')}
