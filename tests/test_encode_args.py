import re

def test_start_number_in_encode_args():
    with open('script.js', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find segment / chunk encode arguments array in script.js
    match = re.search(r'const (?:encodeArgs|segArgs)\s*=\s*\[(.*?)\];', content, re.DOTALL)
    assert match is not None, "Encoding args array (segArgs/encodeArgs) not found in script.js"

    args_str = match.group(1)

    # Check if -start_number 0 is present
    assert "'-start_number'" in args_str and "'0'" in args_str, "-start_number 0 is missing in encode args"

if __name__ == "__main__":
    test_start_number_in_encode_args()
