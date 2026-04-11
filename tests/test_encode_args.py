import re

def test_start_number_in_encode_args():
    with open('script.js', 'r') as f:
        content = f.read()

    # Find encodeArgs section
    encode_args_match = re.search(r'const encodeArgs = \[(.*?)\];', content, re.DOTALL)
    assert encode_args_match is not None, "encodeArgs array not found in script.js"

    args_str = encode_args_match.group(1)

    # Check if -start_number 0 is present
    assert "'-start_number'" in args_str and "'0'" in args_str, "-start_number 0 is missing in encodeArgs"

if __name__ == "__main__":
    test_start_number_in_encode_args()
