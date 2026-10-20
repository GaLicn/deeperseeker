import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import parse_tools

B = chr(65372)
M = B * 2 + 'DSML' + B * 2
Q = chr(34)
L = chr(123)
R = chr(125)
BODY = L + Q + 'name' + Q + ': ' + Q + 'Bash' + Q + ', ' + Q + 'arguments' + Q + ': ' + L + Q + 'command' + Q + ': ' + Q + 'ls -la' + Q + R + R


def run(opener, closer):
    text = '<' + opener + 'tool_call>' + BODY + '</' + closer + 'tool_call>'
    tools, clean = parse_tools(text)
    return [t['function']['name'] for t in tools], clean


def names_ok(opener, closer):
    names, _ = run(opener, closer)
    return names == ['Bash']


def test_bare():
    assert names_ok('', '')


def test_tight():
    assert names_ok(M, M)


def test_spaced():
    assert names_ok(M + ' ', M + ' ')


def test_halfwidth():
    assert names_ok('||DSML||', '||DSML||')


def test_single_bar():
    assert names_ok('|', '|')


def test_no_leak():
    _, clean = run(M, M)
    assert 'arguments' not in clean
    assert 'ls -la' not in clean


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for t in tests:
        try:
            t()
            print('PASS ' + t.__name__)
        except AssertionError as e:
            bad += 1
            print('FAIL ' + t.__name__ + ': ' + str(e))
    if bad:
        sys.exit(1)
    print(str(len(tests)) + ' passed')


if __name__ == '__main__':
    main()
