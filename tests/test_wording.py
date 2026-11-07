"""Check bilingual narrative and its key final-paper settings."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appcore.text import COPY
class WordingTests(unittest.TestCase):
    def test_punctuation_and_languages(self):
        for key,pair in COPY.items():
            self.assertEqual(len(pair),2,key)
            for text in pair:
                self.assertNotIn(';',text,key);self.assertNotIn('\u2014',text,key)
    def test_reported_experiment(self):
        self.assertIn('0.25°',COPY['mic'][0]);self.assertIn('three epochs',COPY['model_body'][0]);self.assertIn('five epochs',COPY['model_body'][0])
        for value in ['8.089°','8.268°','9.564°']:self.assertIn(value,COPY['finding_body'][0])
        self.assertIn('0.943',COPY['results_synthetic_validation'][0])
        self.assertIn('SIP-derived',COPY['evaluation_body'][0]);self.assertIn('0.174°',COPY['swap'][0])
        self.assertIn('179.34°',COPY['qualitative_totalcapture'][0]);self.assertIn('one-swap local',COPY['swap'][0])
if __name__=='__main__':unittest.main()
