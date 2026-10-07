from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.uncapped_consensus_scorer import UncappedConsensusScorer
from gmnps.scoring.native_scoring import NUTRIENTS
from gmnps.scoring.native_scoring import ATTRIBUTES
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES


def test_uncapped_public_interface_preserves_order_and_can_exceed_twelve():
    points = np.array([(FCS2_RULES[a].low_points+FCS2_RULES[a].high_points)/2 for a in ATTRIBUTES])
    native = SimpleNamespace(points=points[None], weights=np.ones((1, len(points))),
        design=np.ones((1,len(NUTRIENTS),len(points))), baseline=np.array([50.]),food_ids=('food',))
    scorer = UncappedConsensusScorer(native, fraction=1.)
    x = pd.DataFrame(np.full((2,len(NUTRIENTS)),20.), columns=NUTRIENTS, index=['a','b'])
    scores = scorer.predict(x)
    assert np.max(abs(scores.to_numpy()-50.)) > 12.
    assert scorer.spec.cap is None
    assert np.min(scores.to_numpy()) >= 1 and np.max(scores.to_numpy()) <= 100
    np.testing.assert_array_equal(scorer.predict(x*0).to_numpy(), np.full((2,1),50.))
    with pytest.raises(ValueError):
        scorer.predict(x[x.columns[::-1]])
    with pytest.raises(ValueError):
        scorer.predict(pd.concat([x,x]))


def test_reference_rejects_changed_upstream_calibration_function():
    points = np.array([(FCS2_RULES[a].low_points+FCS2_RULES[a].high_points)/2 for a in ATTRIBUTES])
    native = SimpleNamespace(points=points[None], weights=np.ones((1,len(points))),
        design=np.ones((1,len(NUTRIENTS),len(points)))*.001, baseline=np.array([50.]),food_ids=('food',))
    x = pd.DataFrame(np.tile(np.array([-2., 1., 4.])[:,None],(1,len(NUTRIENTS))),
                     columns=NUTRIENTS, index=['a','b','c'])
    scorer = UncappedConsensusScorer.fit_reference(native,x)
    np.testing.assert_allclose(scorer.predict(x).mean().to_numpy(), [50.], atol=1e-12)
    with pytest.raises(ValueError, match='stage specification'):
        UncappedConsensusScorer(native, calibrator=scorer.calibrator, fraction=0.)
    other = deepcopy(native)
    other.design *= 2
    with pytest.raises(ValueError, match='points/weights/design'):
        UncappedConsensusScorer(other, calibrator=scorer.calibrator)
