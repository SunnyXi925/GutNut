import numpy as np
import pandas as pd
import pytest
from gmnps.beta_i.nutrient_capacity import NutrientCapacityProjector, signal_rank_report


def data():
    x=pd.DataFrame([[8,2,1],[2,8,2],[6,4,3],[4,6,4]],columns=['a','b','uncovered'],index=['dev1','dev2','dev3','dev4'])
    b=pd.DataFrame([[1,-1,10],[0,0,0],[1,1,2]],index=['contrast','zero','covered_constant'],columns=['a','b','missing'])
    return x,b


def test_zero_sum_coverage_and_unidentifiable():
    x,b=data(); p=NutrientCapacityProjector(b).fit(x)
    np.testing.assert_allclose(p.projection_.sum(axis=1),0,atol=1e-12)
    assert p.coverage_['missing_B_genera']==['missing']
    assert p.coverage_['B_genus_coverage_fraction']==pytest.approx(2/3)
    assert p.diagnostics_.loc['covered_constant','status']=='not_identifiable'
    assert (p.transform(x)[['zero','covered_constant']]==0).all().all()


def test_transform_does_not_fit_on_holdout_or_accept_misaligned_columns():
    x,b=data(); p=NutrientCapacityProjector(b).fit(x)
    center=p.center_.copy(); scale=p.scale_.copy()
    held=x*1000; held.index=['held1','held2','held3','held4']
    p.transform(held)
    np.testing.assert_array_equal(center,p.center_); np.testing.assert_array_equal(scale,p.scale_)
    assert p.fit_sample_ids_==list(x.index)
    with pytest.raises(ValueError,match='order'):
        p.transform(x[['b','a','uncovered']])
    with pytest.raises(ValueError,match='forbidden'):
        p.fit(x,np.array([0,1,0,1]))


def test_projection_varies_across_individuals_and_scale_invariant():
    x,b=data();p=NutrientCapacityProjector(b).fit(x)
    assert p.transform(x)['contrast'].nunique()==4
    np.testing.assert_allclose(p.transform(x),p.transform(x*100),atol=1e-12)
    assert signal_rank_report(p.transform(x))['matrix_rank']==1


def test_mad_falls_back_to_std_for_sparse_signal():
    x=pd.DataFrame([[1,1]]*9+[[10,1]],columns=['a','b'])
    b=pd.DataFrame([[1,-1]],columns=x.columns,index=['nutrient'])
    p=NutrientCapacityProjector(b).fit(x)
    assert p.diagnostics_.loc['nutrient','scale_rule']=='standard_deviation_fallback'
    assert np.isfinite(p.transform(x).to_numpy()).all()
