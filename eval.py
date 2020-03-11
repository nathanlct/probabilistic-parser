from PYEVALB import parser as evalb_parser
from PYEVALB import scorer as evalb_scorer


OUTPUT = 'evaluation_data.parser_output'
TARGET = 'evaluation_data.parser_target'


if __name__ == '__main__':
    with open(OUTPUT, 'r') as f:
        outputs = f.readlines()
    with open(TARGET, 'r') as f:
        targets = f.readlines()

    assert len(outputs) == len(targets)

    n_failures = 0
    n_successes = 0
    total_accuracy = 0

    n = len(outputs)
    for target, output in zip(targets, outputs):
        target = target.strip()
        output = output.strip()

        if output == '-':
            n_failures += 1
        else:
            n_successes += 1

            target_tree = evalb_parser.create_from_bracket_string(target[2:-1])
            output_tree = evalb_parser.create_from_bracket_string(output[2:-1])

            s = evalb_scorer.Scorer()
            result = s.score_trees(target_tree, output_tree)

            total_accuracy += result.tag_accracy

    print('successes', n_successes)
    print('failures:', n_failures)
    print('mean accuracy on successes:', total_accuracy / n_successes)