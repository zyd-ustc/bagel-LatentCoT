import unittest

from qwen_latent_cot.upper_bound.atoms import (match_answer, parse_question,
                                               correction_text, score_atoms, image_pass)
from qwen_latent_cot.upper_bound.metrics import mcnemar_exact, evaluate
from qwen_latent_cot.upper_bound.manifests import normalize


class AtomTests(unittest.TestCase):
    def test_count_words_and_digits(self):
        self.assertTrue(match_answer('seven', 'seven'))
        self.assertTrue(match_answer('7', 'seven'))
        self.assertTrue(match_answer('There are seven.', '7'))
        self.assertFalse(match_answer('six', 'seven'))
        self.assertFalse(match_answer('yes', 'seven'))

    def test_yes_no_and_free_text(self):
        self.assertTrue(match_answer('Yes.', 'Yes'))
        self.assertTrue(match_answer('no', 'No'))
        self.assertFalse(match_answer('No.', 'Yes'))
        self.assertTrue(match_answer('green backpack', 'green backpack'))

    def test_verify_count_template(self):
        text = correction_text([dict(question='Are there exactly one bench in the image?', expected='Yes')])
        self.assertEqual(text, 'Ensure: exactly one bench.')
        text = correction_text([dict(question='Are there exactly seven clock in the image?', expected='No')])
        self.assertEqual(text, 'Ensure: not exactly seven clock.')

    def test_parse_and_constraint(self):
        spec = parse_question('How many backpacks are in the image?')
        self.assertEqual(spec['kind'], 'count')
        self.assertEqual(spec['noun'], 'backpacks')
        spec = parse_question('Is the croissant to the left of the dog?')
        self.assertEqual(spec['kind'], 'position')
        self.assertEqual(spec['direction'], 'left')
        text = correction_text([
            dict(question='How many backpacks are in the image?', expected='one'),
            dict(question='Is the croissant to the left of the dog?', expected='Yes'),
        ])
        self.assertIn('exactly one backpack', text)
        self.assertIn('left of the dog', text)

    def test_image_pass(self):
        rows = score_atoms(['one', 'yes'], [['q1', 'one'], ['q2', 'Yes']])
        self.assertTrue(image_pass(rows))
        rows = score_atoms(['two', 'yes'], [['q1', 'one'], ['q2', 'Yes']])
        self.assertFalse(image_pass(rows))


class ManifestTests(unittest.TestCase):
    def test_geneval_shape(self):
        row = dict(geneval_prompt='a teddy bear right of a dog',
                   geneval_include=[{'class': 'teddy bear', 'count': 1},
                                    {'class': 'dog', 'count': 1, 'position': ['right of', 0]}])
        record = normalize(row, 3)
        self.assertEqual(record['prompt'], 'a teddy bear right of a dog')
        questions = [q for q, _ in record['vqa_list']]
        self.assertIn('Are there exactly one teddy bear in the image?', questions)
        self.assertTrue(any('right of the teddy bear' in q for q in questions))

    def test_geneval_pp_shape(self):
        row = dict(category='counting', case_key='abc',
                   benchmark_metadata=dict(prompt='A photo of six clocks',
                                           include=[{'class': 'clock', 'count': 6}],
                                           exclude=[{'class': 'clock', 'count': 7}]))
        record = normalize(row)
        self.assertEqual(record['prompt_id'], 'abc')
        self.assertEqual(record['vqa_list'][0], ['Are there exactly six clock in the image?', 'Yes'])
        self.assertEqual(record['vqa_list'][-1], ['Are there exactly seven clock in the image?', 'No'])

    def test_direct_vqa_shape(self):
        record = normalize(dict(prompt='x', vqa_list=[['q', 'Yes']], prompt_id='p'))
        self.assertEqual(record['prompt_id'], 'p')


class MetricTests(unittest.TestCase):
    def test_mcnemar(self):
        self.assertAlmostEqual(mcnemar_exact(5, 5), 1.0)
        self.assertLess(mcnemar_exact(9, 1), 0.05)

    def test_evaluation_rows(self):
        rows = [dict(base=dict(passed=False, atoms=[dict(question='q', expected='one',
                                                        answer='two', correct=False)]),
                     previews=[dict(index=25, t=0.7, detected_failure=True, atoms=[])],
                     retries=[dict(index=25,
                                   loop=dict(passed=True, atoms=[dict(question='q', correct=True)]),
                                   resample=dict(passed=False, atoms=[dict(question='q', correct=False)]))])]
        report = evaluate(rows, [25])
        self.assertEqual(report['base']['passed'], 0)
        self.assertEqual(report['read_points'][0]['detection']['tp'], 1)
        self.assertEqual(report['read_points'][0]['retry']['loop_success'], 1)


if __name__ == '__main__':
    unittest.main()
