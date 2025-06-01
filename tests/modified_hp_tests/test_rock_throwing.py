import pytest
from counterfact.examples import RockThrowing
from counterfact.definitions import ModifiedHP


class TestAC1ModifiedHPRockThrowing:

    def test_1(self):
        # Correct event and outcome, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "suzy_hits": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_factual(env, event, outcome, state, noise)
        assert result is True

    def test_2(self):
        # Impossible state but should still return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 1,
            "bottle_shatters": 1,
        }
        event = {"suzy_hits": 1, "billy_hits": 1}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_factual(env, event, outcome, state, noise)
        assert result is True

    def test_3(self):
        # Impossible outcome but should still return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 1,
            "bottle_shatters": 0,
        }
        event = {"suzy_throws": 1}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_factual(env, event, outcome, state, noise)
        assert result is True

    def test_4(self):
        # Incorrect event, should return False
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 0,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 0, "suzy_hits": 1}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 0}
        result, info = ac_defn.is_factual(env, event, outcome, state, noise)
        assert result is False

        # Check if the given info is correct
        assert "incorrect_events" in info
        assert "suzy_throws" in info["incorrect_events"]
        assert info["incorrect_events"]["suzy_throws"] is 0

    def test_5(self):
        # Incorrect outcome, should return False
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 0,
            "billy_throws": 1,
            "suzy_hits": 0,
            "billy_hits": 1,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 0}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 0, "billy_throws": 1}
        result, info = ac_defn.is_factual(env, event, outcome, state, noise)
        assert result is False

        # Check if the given info is correct
        assert "incorrect_outcomes" in info
        assert "bottle_shatters" in info["incorrect_outcomes"]
        assert info["incorrect_outcomes"]["bottle_shatters"] is 0


class TestSufficiencyModifiedHPRockThrowing:

    def test_1(self):
        # Event is sufficient but not minimal, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "suzy_hits": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
        assert result is True

    def test_2(self):
        # Event is sufficient but not minimal, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 0,
            "billy_throws": 0,
            "suzy_hits": 0,
            "billy_hits": 0,
            "bottle_shatters": 0,
        }
        event = {"suzy_throws": 0}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 0, "billy_throws": 0}
        result, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
        assert result is True

    def test_3(self):
        # Event is not sufficient, should return False
        # This is also an AC1 violation
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
        assert result is False
        assert "ac2b_alt_outcome" in info
        assert "bottle_shatters" in info["ac2b_alt_outcome"]
        assert info["ac2b_alt_outcome"]["bottle_shatters"] == 1


class TestNecessityModifiedHPRockThrowing:

    def test_1(self):
        # Event is necessary but not minimal, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "billy_throws": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_necessary(env, event, outcome, state, noise)
        assert result is True

        # Check that the correct info was returned
        assert "ac2a_alt_event" in info
        # Check that all keys in the alt event are in the original event
        for key in info["ac2a_alt_event"]:
            assert key in event
        # Check that at least one value in the alt event is not the same as the original event
        assert any([info["ac2a_alt_event"][key] != event[key] for key in event])
        # Check that the alt event is not sufficient for the outcome
        sufficient, _ = ac_defn.is_sufficient(
            env, info["ac2a_alt_event"], outcome, state, noise
        )
        assert sufficient is False

    def test_2(self):
        # Event is necessary and minimal, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_necessary(env, event, outcome, state, noise)
        assert result is True

        # Check that the correct info was returned
        assert "ac2a_alt_event" in info
        # Check that all keys in the alt event are in the original event
        for key in info["ac2a_alt_event"]:
            assert key in event
        # Check that at least one value in the alt event is not the same as the original event
        assert any([info["ac2a_alt_event"][key] != event[key] for key in event])
        # Check that the alt event is not sufficient for the outcome
        sufficient, _ = ac_defn.is_sufficient(
            env, info["ac2a_alt_event"], outcome, state, noise
        )
        assert sufficient is False

    def test_3(self):
        # Event is not necessary, should return False
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_necessary(env, event, outcome, state, noise)
        assert result is False

    def test_4(self):
        # Event is necessary and minimal, should return True
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 0,
            "billy_throws": 0,
            "suzy_hits": 0,
            "billy_hits": 0,
            "bottle_shatters": 0,
        }
        event = {"suzy_throws": 0}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 0, "billy_throws": 0}
        result, info = ac_defn.is_necessary(env, event, outcome, state, noise)
        assert result is True

        # Check that the correct info was returned
        assert "ac2a_alt_event" in info
        # Check that all keys in the alt event are in the original event
        for key in info["ac2a_alt_event"]:
            assert key in event
        # Check that at least one value in the alt event is not the same as the original event
        assert any([info["ac2a_alt_event"][key] != event[key] for key in event])
        # Check that the alt event is not sufficient for the outcome
        sufficient, _ = ac_defn.is_sufficient(
            env, info["ac2a_alt_event"], outcome, state
        )
        assert sufficient is False


class TestMinimalityModifiedHPRockThrowing:

    def test_1(self):
        # Correct event and outcome, suzy_throws = 0 is a minimal actual cause
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 0,
            "billy_throws": 0,
            "suzy_hits": 0,
            "billy_hits": 0,
            "bottle_shatters": 0,
        }
        event = {"suzy_throws": 0}
        outcome = {"bottle_shatters": 0}
        noise = {"suzy_throws": 0, "billy_throws": 0}
        result, info = ac_defn.is_minimal(env, event, outcome, state, noise)
        assert result is True

        # Make sure no smaller cause is reported in the info dict
        assert "ac3_smaller_cause" not in info or not info["ac3_smaller_cause"]

    def test_2(self):
        # Necessary and sufficient cause found but it is not minimal
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "suzy_hits": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_minimal(env, event, outcome, state, noise)
        assert result is False

        # Make sure the smaller cause reported in the info dict is actually smaller
        assert "ac3_smaller_cause" in info
        assert info["ac3_smaller_cause"]
        assert len(info["ac3_smaller_cause"]) < len(event)

        # Make sure the smaller cause is necessary and sufficient
        subevent = info["ac3_smaller_cause"]
        sufficient, _ = ac_defn.is_sufficient(env, subevent, outcome, state, noise)
        assert sufficient is True
        necessary, _ = ac_defn.is_necessary(env, subevent, outcome, state, noise)
        assert necessary is True

    def test_3(self):
        # Minimal necessary and sufficient event found
        env = RockThrowing()
        ac_defn = ModifiedHP()
        state = {
            "suzy_throws": 1,
            "billy_throws": 1,
            "suzy_hits": 1,
            "billy_hits": 0,
            "bottle_shatters": 1,
        }
        event = {"suzy_throws": 1, "billy_hits": 0}
        outcome = {"bottle_shatters": 1}
        noise = {"suzy_throws": 1, "billy_throws": 1}
        result, info = ac_defn.is_minimal(env, event, outcome, state, noise)
        assert result is True
        # Make sure no smaller cause is reported in the info dict
        assert "ac3_smaller_cause" not in info or not info["ac3_smaller_cause"]
