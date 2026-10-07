import numpy as np
import pytest
from gmnps.scoring.consensus_calibration import DevelopmentConsensusCalibrator, NativeStageSpec


def test_additive_preserves_development_means_and_keeps_test_shift():
    x=np.array([[0.,1.,-1.],[2.,3.,2.],[4.,5.,5.]])
    b=np.array([1.,50.,100.]); ids=['a','b','c']
    model=DevelopmentConsensusCalibrator().fit(x,b,ids,development_ids=['p','q','r'])
    np.testing.assert_allclose(model.transform(x,ids).mean(0),b,atol=1e-12)
    np.testing.assert_allclose(model.transform(x+7,ids).mean(0),b+7,atol=1e-12)
    assert model.transform(x,ids).min()<1
    assert model.transform(x,ids).max()>100
    with pytest.raises(ValueError):model.transform(x,ids[::-1])


def test_logit_mean_identity_endpoint_constraint_and_derivative():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(100,5))*np.array([2,3,4,5,6])
    b=np.array([1.,10.,50.,95.,100.]);ids=list('abcde')
    model=DevelopmentConsensusCalibrator('bounded_logit',10).fit(x,b,ids,development_ids=list(range(100)))
    score=model.transform(x,ids)
    np.testing.assert_allclose(score.mean(0),b,atol=1e-10)
    assert np.min(score)>=1 and np.max(score)<=100
    np.testing.assert_array_equal(score[:,0],np.ones(100))
    np.testing.assert_array_equal(score[:,-1],np.full(100,100.))
    h=1e-4
    fd=(model.transform(x+h,ids)-model.transform(x-h,ids))/(2*h)
    np.testing.assert_allclose(fd,model.derivative(x,ids),atol=1e-9)
    assert (model.transform(x+2,ids).mean(0)[1:-1]>b[1:-1]).all()


def test_invalid_development_ids_and_stage_options_rejected():
    with pytest.raises(ValueError):
        DevelopmentConsensusCalibrator().fit(np.ones((2,2)),[2,3],['a','b'],development_ids=['x','x'])
    with pytest.raises(ValueError):NativeStageSpec(temperature=0)
    with pytest.raises(ValueError):NativeStageSpec(cap=-1)


def test_stage_directional_derivative_and_zero_anchor():
    from types import SimpleNamespace
    from gmnps.scoring.consensus_calibration import NativeStageEngine
    from gmnps.scoring.native_scoring import ATTRIBUTES
    from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES
    rng=np.random.default_rng(91)
    point=np.array([(FCS2_RULES[a].low_points+FCS2_RULES[a].high_points)/2 for a in ATTRIBUTES])
    scorer=SimpleNamespace(points=np.tile(point,(2,1)),weights=np.ones((2,len(point))),
        design=rng.normal(size=(2,3,len(point)))*.1,baseline=np.array([40.,60.]),food_ids=('a','b'))
    engine=NativeStageEngine(scorer)
    signals=rng.normal(size=(4,3));direction=np.array([1.,0.,-.5]);spec=NativeStageSpec(attribute_clip=False,native_clip=False)
    _,analytic,_=engine.evaluate(signals,spec,direction=direction)
    eps=1e-5
    plus,_,_=engine.evaluate(signals+eps*direction,spec)
    minus,_,_=engine.evaluate(signals-eps*direction,spec)
    np.testing.assert_allclose((plus-minus)/(2*eps),analytic,atol=1e-9)
    zero,_,_=engine.evaluate(np.zeros((1,3)),spec)
    np.testing.assert_array_equal(zero,np.zeros((1,2)))
