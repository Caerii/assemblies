"""
Tests for language parsing modules.

This module contains tests for the language processing functionality,
including parser brain classes and grammar rules.
"""

import unittest

from src.language.parser import ParserBrain, RussianParserBrain, EnglishParserBrain
from src.language.grammar_rules import LEXEME_DICT, RUSSIAN_LEXEME_DICT
from src.language.language_areas import *
from src.language.readout_methods import ReadoutMethod

class TestLanguageParsing(unittest.TestCase):
    """Test language parsing functionality."""
    
    def test_parser_brain_initialization(self):
        """Test basic parser brain initialization."""
        brain = ParserBrain(p=0.05, lexeme_dict=LEXEME_DICT, 
                           all_areas=AREAS, recurrent_areas=RECURRENT_AREAS,
                           initial_areas=[LEX], readout_rules=ENGLISH_READOUT_RULES)
        self.assertIsNotNone(brain)
        self.assertEqual(len(brain.all_areas), len(AREAS))
        self.assertEqual(len(brain.recurrent_areas), len(RECURRENT_AREAS))

    def test_english_parser_brain(self):
        """Test English parser brain initialization."""
        brain = EnglishParserBrain(p=0.05, LEX_k=20, verbose=False)
        self.assertIsNotNone(brain)
        self.assertIn(LEX, brain.area_by_name)
        self.assertIn(SUBJ, brain.area_by_name)
        self.assertIn(VERB, brain.area_by_name)
        self.assertIn(OBJ, brain.area_by_name)

    def test_russian_parser_brain(self):
        """Test Russian parser brain initialization."""
        brain = RussianParserBrain(p=0.05, LEX_k=10, verbose=False)
        self.assertIsNotNone(brain)
        self.assertIn(LEX, brain.area_by_name)
        self.assertIn(NOM, brain.area_by_name)
        self.assertIn(VERB, brain.area_by_name)
        self.assertIn(ACC, brain.area_by_name)
        self.assertIn(DAT, brain.area_by_name)

    def test_lexeme_dictionaries(self):
        """Test that lexeme dictionaries are properly loaded."""
        self.assertIsInstance(LEXEME_DICT, dict)
        self.assertIsInstance(RUSSIAN_LEXEME_DICT, dict)
        self.assertGreater(len(LEXEME_DICT), 0)
        self.assertGreater(len(RUSSIAN_LEXEME_DICT), 0)

    def test_grammar_rules_structure(self):
        """Test that grammar rules have proper structure."""
        for word, lexeme in LEXEME_DICT.items():
            self.assertIn("index", lexeme)
            self.assertIn("PRE_RULES", lexeme)
            self.assertIn("POST_RULES", lexeme)
            self.assertIsInstance(lexeme["PRE_RULES"], list)
            self.assertIsInstance(lexeme["POST_RULES"], list)

    def test_readout_methods(self):
        """Test readout method enumeration."""
        self.assertEqual(ReadoutMethod.FIXED_MAP_READOUT.value, 1)
        self.assertEqual(ReadoutMethod.FIBER_READOUT.value, 2)
        self.assertEqual(ReadoutMethod.NATURAL_READOUT.value, 3)

    def test_area_definitions(self):
        """Test that area definitions are properly set."""
        self.assertEqual(LEX, "LEX")
        self.assertEqual(SUBJ, "SUBJ")
        self.assertEqual(VERB, "VERB")
        self.assertEqual(OBJ, "OBJ")
        self.assertIn(LEX, AREAS)
        self.assertIn(SUBJ, AREAS)
        self.assertIn(VERB, AREAS)
        self.assertIn(OBJ, AREAS)

if __name__ == '__main__':
    unittest.main()
