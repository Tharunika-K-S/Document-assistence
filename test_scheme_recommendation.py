"""
GovEase AI - Government Scheme Hybrid Recommendation Engine Test Suite
Tests:
1. Dataset Loading & Preprocessing
2. Mandatory Eligibility Parsing (Age, Income, Gender, Community, State)
3. Eligibility Status Determination (eligible, potentially_eligible, ineligible, insufficient_information)
4. TF-IDF Vectorization & Cosine Similarity Calculation
5. Bilingual Query Expansion (Tamil to English Semantic Mapping)
6. Weighted Multi-Criteria Ranking & Score Breakdown
7. Official Source URL Verification (myscheme.gov.in)
8. Flask API Endpoints (/schemes/, /schemes/api/weights, /schemes/api/questions/<cat>, /schemes/api/match)
"""

import os
import sys
import unittest
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from scheme_recommendation import (
    load_schemes_dataset,
    get_tfidf_engine,
    parse_age_limits,
    parse_income_limit,
    check_scheme_eligibility,
    expand_bilingual_query,
    calculate_hybrid_score,
    recommend_schemes,
    DEFAULT_WEIGHTS,
    CATEGORIES,
    PROFILE_QUESTIONS
)


class TestDatasetAndPreprocessing(unittest.TestCase):
    """Verifies dataset integrity and in-memory TF-IDF index construction."""

    def test_dataset_loads_successfully(self):
        rows = load_schemes_dataset()
        self.assertIsInstance(rows, list, "Schemes dataset should load as a list of dicts")
        self.assertGreaterEqual(len(rows), 4000, f"Expected 4000+ schemes, found {len(rows)}")
        sample = rows[0]
        for key in ['scheme_name', 'categories', 'brief_description', 'benefits', 'eligibility', 'source_url']:
            self.assertIn(key, sample, f"Required key '{key}' missing from scheme record")

    def test_source_urls_are_official(self):
        rows = load_schemes_dataset()
        urls = [r.get('source_url', '') for r in rows if r.get('source_url')]
        self.assertGreater(len(urls), 4000)
        # Verify official domain myscheme.gov.in
        valid_gov_count = sum(1 for u in urls if 'myscheme.gov.in' in u.lower())
        self.assertGreater(valid_gov_count, 3500, "Schemes must originate from official portal myscheme.gov.in")

    def test_tfidf_matrix_cached_and_non_empty(self):
        vec, mat, rows = get_tfidf_engine()
        self.assertIsNotNone(vec, "TF-IDF Vectorizer should not be None")
        self.assertIsNotNone(mat, "TF-IDF Matrix should not be None")
        self.assertEqual(mat.shape[0], len(rows), "TF-IDF rows must equal number of schemes")
        self.assertGreater(mat.shape[1], 1000, "TF-IDF vocabulary should have at least 1,000 terms")


class TestEligibilityParsing(unittest.TestCase):
    """Verifies regex extraction of mandatory eligibility bounds (age, income, gender, community)."""

    def test_age_bounds_parsing(self):
        min_age, max_age = parse_age_limits(
            "Applicants must be between 18 and 45 years of age at the time of application."
        )
        self.assertEqual(min_age, 18)
        self.assertEqual(max_age, 45)

        min_age_senior, max_age_senior = parse_age_limits(
            "Senior citizens aged 60 years and above are eligible for this pension."
        )
        self.assertEqual(min_age_senior, 60)

    def test_income_ceiling_parsing(self):
        income1 = parse_income_limit(
            "Family annual income must not exceed Rs. 2.5 Lakhs from all sources."
        )
        self.assertEqual(income1, 250000)

        income2 = parse_income_limit(
            "The applicant must hold a valid BPL (Below Poverty Line) ration card."
        )
        self.assertEqual(income2, "BPL")


class TestMandatoryEligibilityFiltering(unittest.TestCase):
    """Verifies that mandatory criteria correctly classify profiles and NEVER flag ineligible as eligible."""

    def test_state_mismatch_results_in_ineligible(self):
        scheme_row = {
            'scheme_name': 'Kerala State Paddy Subsidy',
            'state': 'Kerala',
            'level': 'State',
            'eligibility': 'Exclusively for farmers residing in Kerala.',
            'target_beneficiaries': 'Farmers',
            'brief_description': 'Paddy support in Kerala.'
        }
        user_answers = {'state': 'Tamil Nadu', 'age': '30', 'gender': 'female', 'income': '200000'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertEqual(status, 'ineligible')
        self.assertTrue(any('Kerala' in m for m in missing), "Should detail state residency mismatch in explanation")

    def test_age_out_of_bounds_results_in_ineligible(self):
        scheme_row = {
            'scheme_name': 'Old Age Welfare Pension',
            'state': 'Tamil Nadu',
            'level': 'State',
            'eligibility': 'Applicant must be 60 years of age and above.',
            'target_beneficiaries': 'Senior citizens'
        }
        # 25-year-old applying for a 60+ scheme
        user_answers = {'state': 'Tamil Nadu', 'age': '25', 'gender': 'female', 'income': '200000'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertEqual(status, 'ineligible')
        self.assertTrue(any('minimum age of 60' in m for m in missing))

    def test_gender_mismatch_results_in_ineligible(self):
        scheme_row = {
            'scheme_name': 'Destitute Widow Financial Assistance',
            'state': 'Tamil Nadu',
            'level': 'State',
            'target_beneficiaries': 'destitute widows and abandoned women',
            'eligibility': 'Reserved exclusively for women applicants.'
        }
        user_answers = {'state': 'Tamil Nadu', 'age': '30', 'gender': 'male', 'income': '150000'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertEqual(status, 'ineligible')
        self.assertTrue(any('female' in m.lower() for m in missing))

    def test_income_exceeded_results_in_ineligible(self):
        scheme_row = {
            'scheme_name': 'Marginal Farmer Grant',
            'state': 'Tamil Nadu',
            'level': 'State',
            'eligibility': 'Annual family income must not exceed Rs. 2,00,000 from all sources.',
            'target_beneficiaries': 'Low income farmers'
        }
        user_answers = {'state': 'Tamil Nadu', 'age': '30', 'gender': 'female', 'income': '600000'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertEqual(status, 'ineligible')
        self.assertTrue(any('income ceiling exceeded' in m.lower() for m in missing))

    def test_full_criteria_match_results_in_eligible(self):
        scheme_row = {
            'scheme_name': 'Tamil Nadu Small Farmer Mechanization',
            'state': 'Tamil Nadu',
            'level': 'State',
            'eligibility': 'Eligible farmers aged 18 to 60 years with annual family income up to Rs. 3,00,000.',
            'target_beneficiaries': 'Small and marginal farmers',
            'brief_description': 'Farm support'
        }
        user_answers = {'state': 'Tamil Nadu', 'age': '35', 'gender': 'female', 'income': '180000', 'occupation': 'farmer'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertEqual(status, 'eligible')
        self.assertEqual(len(missing), 0)
        self.assertGreaterEqual(len(reasons), 2)

    def test_missing_profile_data_results_in_insufficient_information(self):
        scheme_row = {
            'scheme_name': 'Youth Skill Development Scheme',
            'state': 'Tamil Nadu',
            'level': 'State',
            'eligibility': 'Applicant must be between 18 and 35 years old and hold a family income under 2.5 lakhs.',
            'target_beneficiaries': 'Unemployed youth'
        }
        # User only specified state, but did not provide age or income
        user_answers = {'state': 'Tamil Nadu'}
        status, reasons, missing = check_scheme_eligibility(scheme_row, user_answers)
        self.assertIn(status, ['insufficient_information', 'potentially_eligible'])
        self.assertNotEqual(status, 'eligible', "Must NEVER claim eligible when mandatory criteria are unverified")
        self.assertGreater(len(missing), 0, "Missing verifications must document unverified requirements")


class TestBilingualSemanticExpansion(unittest.TestCase):
    """Verifies Tamil-to-English semantic mapping and keyword expansion."""

    def test_tamil_agriculture_query_expansion(self):
        raw_query = "விவசாயம் டிராக்டர் மானியம்"
        expanded = expand_bilingual_query(raw_query)
        self.assertIn("agriculture", expanded.lower())
        self.assertIn("tractor", expanded.lower())
        self.assertIn("subsidy", expanded.lower())

    def test_tamil_education_query_expansion(self):
        raw_query = "பெண் கல்வி உதவித்தொகை கல்லூரி"
        expanded = expand_bilingual_query(raw_query)
        self.assertIn("scholarship", expanded.lower())
        self.assertIn("education", expanded.lower())

    def test_english_query_preservation(self):
        raw_query = "solar water pump subsidy for small farmers"
        expanded = expand_bilingual_query(raw_query)
        self.assertEqual(raw_query.strip(), expanded.strip())

    def test_empty_query_handling(self):
        self.assertEqual(expand_bilingual_query(""), "")
        self.assertEqual(expand_bilingual_query(None), "")


class TestWeightedScoringAndRanking(unittest.TestCase):
    """Verifies multi-criteria formula: Relevance = w1*Sim + w2*Cat + w3*State + w4*Profile."""

    def test_formula_weights_sum_to_one(self):
        total_weight = sum(DEFAULT_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_hybrid_score_calculation_breakdown(self):
        score, explanation, breakdown = calculate_hybrid_score(
            content_sim=0.80,
            category_match=1.0,
            state_match=1.0,
            profile_match=0.75,
            eligibility_status='eligible',
            weights=DEFAULT_WEIGHTS
        )
        self.assertGreaterEqual(score, 80)
        self.assertEqual(breakdown['content_similarity'], 80.0)
        self.assertEqual(breakdown['category_match'], 100.0)
        self.assertEqual(breakdown['state_match'], 100.0)
        self.assertEqual(breakdown['profile_match'], 75.0)
        self.assertIn("Content Similarity", explanation)

    def test_ineligible_schemes_are_severely_penalized(self):
        score, explanation, breakdown = calculate_hybrid_score(
            content_sim=0.95,
            category_match=1.0,
            state_match=0.0,
            profile_match=0.0,
            eligibility_status='ineligible',
            weights=DEFAULT_WEIGHTS
        )
        self.assertLessEqual(score, 15, "Ineligible schemes must be severely penalized in match score")
        self.assertIn("DISQUALIFIED", explanation)


class TestEndToEndRecommendations(unittest.TestCase):
    """Verifies end-to-end scheme matching with real dataset queries."""

    def test_tamil_tractor_query_recommends_relevant_schemes(self):
        results = recommend_schemes(
            category_key="agriculture",
            answers={"state": "Tamil Nadu", "age": "40", "gender": "male", "income": "200000"},
            free_query="விவசாயம் டிராக்டர் மானியம்",
            limit=5
        )
        self.assertGreater(len(results), 0, "Should return matching schemes for Tamil tractor query")
        top_scheme = results[0]
        self.assertIn('name', top_scheme)
        self.assertIn('match_score', top_scheme)
        self.assertIn('eligibility_status', top_scheme)
        self.assertIn('score_breakdown', top_scheme)
        self.assertIn('source_url', top_scheme)
        # Verify official URL
        self.assertTrue(top_scheme['source_url'].startswith('https://www.myscheme.gov.in/'))

    def test_widow_scheme_not_eligible_for_male(self):
        results = recommend_schemes(
            category_key="women_child",
            answers={"state": "Tamil Nadu", "age": "35", "gender": "male"},
            free_query="widow pension assistance",
            limit=5
        )
        for s in results:
            if 'widow' in s['name'].lower():
                self.assertNotEqual(s['eligibility_status'], 'eligible', "Male applicant must not be marked eligible for widow schemes")

    def test_state_specific_scheme_not_eligible_for_other_states(self):
        results = recommend_schemes(
            category_key="agriculture",
            answers={"state": "Tamil Nadu"},
            free_query="agriculture",
            limit=10
        )
        for s in results:
            if s.get('state') and s['state'].lower() not in ['tamil nadu', 'all india', 'central']:
                self.assertNotEqual(s['eligibility_status'], 'eligible', "Other state scheme must not be eligible for Tamil Nadu resident")


class TestFlaskEndpoints(unittest.TestCase):
    """Verifies Flask HTTP routes and JSON API contracts."""

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_schemes_homepage_route(self):
        res = self.client.get('/schemes/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Government Scheme Recommendation", res.data)
        self.assertIn(b"TF-IDF", res.data)

    def test_weights_api_endpoint(self):
        res = self.client.get('/schemes/api/weights')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('weights', data)
        self.assertEqual(data['weights']['content_similarity'], 0.40)

    def test_questions_api_endpoint(self):
        res = self.client.get('/schemes/api/questions/agriculture')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('category'), 'agriculture')
        self.assertGreater(len(data.get('questions', [])), 0)

    def test_match_api_post_with_tamil_query(self):
        payload = {
            "category": "agriculture",
            "query": "விவசாய டிராக்டர் மானியம்",
            "answers": {
                "state": "Tamil Nadu",
                "age": "38",
                "gender": "female",
                "income": "150000"
            }
        }
        res = self.client.post(
            '/schemes/api/match',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertGreater(len(data.get('results', [])), 0)
        first = data['results'][0]
        self.assertIn('name', first)
        self.assertIn('match_score', first)
        self.assertIn('eligibility_status', first)
        self.assertIn('score_breakdown', first)
        self.assertIn('score_explanation', first)

    def test_match_api_missing_category_returns_400(self):
        res = self.client.post(
            '/schemes/api/match',
            data=json.dumps({"answers": {}}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
