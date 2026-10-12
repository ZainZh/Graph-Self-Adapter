"""Keep frozen historical evidence fixtures separate from active verification."""
# This exact yaw-only toy FK fixture is bound by the accepted nominal receipt.
# It is rank deficient for the solver's 3D rotation equalities, not an active
# production regression. Preserve its bytes; the full6DOF replacement owns the
# current check. See HISTORICAL_FIXTURES.json and the accepted package snapshot.
collect_ignore = ['test_cup_v25_current_goalset.py']
